"""Experiment 1 -- validate the Instruction-Gradient Alignment Law (§4, §9).

Empirical Law 1 states that instruction-embedding similarity ``<p_t, p_j>`` is
monotone in the alignment of task gradients *restricted to the shared occupied
subspace* ``U_j``:

    E[ <grad L_j, grad L_t>_{U_j}  |  <p_t,p_j> = rho ] = phi(rho),

with ``phi`` non-decreasing and a zero-crossing ``rho0``.  This module measures
both sides on real task pairs and reports the correlation and the fitted
``rho0`` -- "this single plot decides the paper; it is cheap."

Weight-space gradient (GPM style)
---------------------------------
For a tracked linear ``y = x W^T`` the loss gradient wrt ``W`` at the shared
pretrained point factorises through hooks::

    grad_W L = sum_tokens (dL/dy)^T x            # shape (out, in)

We capture ``x`` (forward hook) and ``dL/dy`` (backward hook) so no explicit
per-weight autograd on the frozen base is needed.  The subspace-restricted inner
product between two tasks is then

    <G_j, G_t>_{U_j} = < G_j U_j , G_t U_j >_F   (U_j has shape in x r_j)

and we report its cosine.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .lora import iter_instil_layers


class _GradCatcher:
    """Accumulate weight-space gradients ``G = sum (dL/dy)^T x`` per base linear."""

    def __init__(self, base_linears: Dict[str, nn.Linear]):
        self.base_linears = base_linears
        self._x: Dict[str, torch.Tensor] = {}
        self.G: Dict[str, torch.Tensor] = {}
        self._handles = []

    def __enter__(self):
        for name, lin in self.base_linears.items():
            self.G[name] = torch.zeros(lin.out_features, lin.in_features)

            def fwd_hook(mod, inp, out, nm=name):
                self._x[nm] = inp[0].detach()

            def bwd_hook(mod, grad_in, grad_out, nm=name):
                gy = grad_out[0].detach()             # (*, out)
                x = self._x.get(nm)
                if x is None:
                    return
                gy2 = gy.reshape(-1, gy.shape[-1]).float()
                x2 = x.reshape(-1, x.shape[-1]).float()
                self.G[nm] += (gy2.t() @ x2).cpu()    # (out, in)

            self._handles.append(lin.register_forward_hook(fwd_hook))
            self._handles.append(lin.register_full_backward_hook(bwd_hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []


def compute_task_weight_gradients(
    model: nn.Module,
    run_backward: Callable[[], torch.Tensor],
    layer_names: Optional[Sequence[str]] = None,
    reset_adapters: bool = True,
) -> Dict[str, torch.Tensor]:
    """Return ``{layer_name: G}`` accumulated over ``run_backward`` calls.

    ``run_backward`` should perform one forward pass on a batch of the task,
    call ``loss.backward()``, and return the loss (it may be called several
    times to average over batches).  With ``reset_adapters`` the adapters are
    zeroed first so the gradient is measured at the shared pretrained point.
    """
    layers = dict(iter_instil_layers(model))
    if layer_names is not None:
        layers = {n: l for n, l in layers.items() if n in layer_names}
    if reset_adapters:
        for l in layers.values():
            nn.init.zeros_(l.lora_B)  # delta == 0 -> measure at base point

    base = {n: l.base for n, l in layers.items()}
    with _GradCatcher(base) as catcher:
        model.zero_grad(set_to_none=True)
        run_backward()
    return catcher.G


def subspace_cosine(G_j: torch.Tensor, G_t: torch.Tensor, U_j: torch.Tensor) -> float:
    """Cosine of two weight-space gradients restricted to ``span(U_j)``."""
    if U_j.numel() == 0:
        return 0.0
    Gj = (G_j @ U_j)   # (out, r_j)
    Gt = (G_t @ U_j)
    num = (Gj * Gt).sum()
    den = Gj.norm() * Gt.norm()
    if den < 1e-12:
        return 0.0
    return float((num / den).item())


def measure_gradient_alignment(
    prototypes: Sequence[torch.Tensor],
    task_gradients: Sequence[Dict[str, torch.Tensor]],
    task_subspaces: Sequence[Dict[str, torch.Tensor]],
    layer_names: Optional[Sequence[str]] = None,
    aggregate: str = "mean",
) -> Tuple[List[float], List[float], List[Tuple[int, int]]]:
    """Pair up tasks and return ``(similarities, alignments, pairs)``.

    For every ordered pair ``(j, t)`` with ``j != t`` we compute the instruction
    cosine ``<p_t, p_j>`` and the subspace gradient cosine
    ``<G_j, G_t>_{U_j}`` (aggregated over tracked layers).

    Parameters
    ----------
    task_gradients : per task, ``{layer: G}`` from :func:`compute_task_weight_gradients`.
    task_subspaces : per task, ``{layer: U_j}`` occupied bases.
    """
    K = len(prototypes)
    sims: List[float] = []
    aligns: List[float] = []
    pairs: List[Tuple[int, int]] = []
    for j in range(K):
        for t in range(K):
            if j == t:
                continue
            names = layer_names or list(task_gradients[j].keys())
            per_layer = []
            for nm in names:
                if nm not in task_gradients[t] or nm not in task_subspaces[j]:
                    continue
                per_layer.append(subspace_cosine(
                    task_gradients[j][nm], task_gradients[t][nm],
                    task_subspaces[j][nm],
                ))
            if not per_layer:
                continue
            align = sum(per_layer) / len(per_layer) if aggregate == "mean" else max(per_layer)
            sim = float((prototypes[t].flatten() @ prototypes[j].flatten()).item())
            sims.append(sim)
            aligns.append(align)
            pairs.append((j, t))
    return sims, aligns, pairs


def validate_law(similarities: Sequence[float], alignments: Sequence[float]) -> dict:
    """Summarise the Law: Pearson r, sign agreement, and fitted ``rho0``.

    Returns a dict with ``pearson`` (monotonicity strength), ``sign_accuracy``
    (fraction of pairs where high similarity coincides with positive alignment),
    ``rho0`` (the fitted zero-crossing consumed by the gate), and ``n`` pairs.
    """
    from .gate import fit_rho0

    x = torch.tensor(similarities, dtype=torch.float32)
    y = torch.tensor(alignments, dtype=torch.float32)
    n = x.numel()
    out = {"n": int(n), "pearson": 0.0, "sign_accuracy": 0.0, "rho0": 0.0}
    if n < 2:
        return out
    xm, ym = x.mean(), y.mean()
    cov = ((x - xm) * (y - ym)).mean()
    sx = x.std(unbiased=False)
    sy = y.std(unbiased=False)
    out["pearson"] = float((cov / (sx * sy + 1e-12)).item())
    out["rho0"] = fit_rho0(x, y)
    # Sign agreement: does (sim >= rho0) predict (alignment >= 0)?
    pred_pos = x >= out["rho0"]
    actual_pos = y >= 0
    out["sign_accuracy"] = float((pred_pos == actual_pos).float().mean().item())
    return out
