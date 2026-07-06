"""InstilLoRALinear -- the adapted linear layer (paper §5, §8).

Design (why A is frozen and B is trained)
------------------------------------------
The proposal splits each task's low-rank update into a *null-space* part
(provably non-interfering) and a *gated occupied* part (Eq. 1).  We realise this
InfLoRA-style: for task ``t`` the orchestrator builds a **frozen** input basis
``A`` whose rows are

    [  F_t                    |   gamma_{t,j} * U_j  for aligned prior tasks j  ]
       ^ null-space directions      ^ occupied directions, admitted in
         the task cares about         proportion to instruction similarity

and trains only ``B``.  Because the effective update is ``dW = scaling * B @ A``:

* Every null-space row of ``A`` is orthogonal to ``span(U_<t)``, so for any
  ``B`` and any ``x in span(U_<t)`` the null part contributes ``0``  -> the
  update cannot disturb prior tasks through the free subspace.
* An occupied block ``j`` enters ``A`` **only** with weight ``gamma_{t,j}``.
  When ``gamma_{t,j} = 0`` the block is absent, ``A U_j = 0``, and prior task
  ``j`` is provably untouched (Prop. 1 -- exact non-forgetting, the isolation
  special case ``gamma == 0``).  When ``gamma_{t,j} > 0`` the block is admitted
  and cross-task reinforcement is possible (Thm. 1).

Modes (§5.3)
------------
* ``merge`` -- after each task the current ``B @ A`` is *folded* into a single
  running delta ``delta_prev`` (out x in); memory does not grow.
* ``bank``  -- each task's ``(A, B)`` is stored; inference composes them with
  soft routing weights (Alg. 2) for training-free routing and zero-shot blends.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .subspace import SubspaceMemory


class InstilLoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 16,
                 dropout: float = 0.0, name: str = ""):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        self.name = name
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Occupied-subspace memory for this layer (grows one block per task).
        self.memory = SubspaceMemory(self.in_features)

        # ---- Current task adapter: frozen A (buffer), trainable B (param) ----
        # Shapes are set per task by ``set_adapter_basis``; start empty.
        self.register_buffer("lora_A", torch.zeros(0, self.in_features), persistent=False)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, 0))

        # ---- Merge mode: running folded delta (out x in) --------------------
        self.register_buffer("delta_prev", torch.zeros(self.out_features, self.in_features),
                             persistent=True)

        # ---- Bank mode: frozen per-task adapters ----------------------------
        self._bank_A: List[torch.Tensor] = []
        self._bank_B: List[torch.Tensor] = []
        self.register_buffer("_route_weights", None, persistent=False)

        # ---- Bypass: return the pure frozen base (drift-free prototypes) ----
        self._bypass = False

        # ---- Streaming covariance capture for subspace building -------------
        # We accumulate C = sum_x x x^T  (in x in, fixed size) on the fly rather
        # than storing an (N x in) activation matrix, then take its top-r
        # eigenvectors -- avoiding an expensive SVD over all activations
        # (HESTIA-style online statistics).
        self._collecting = False
        self._cov: Optional[torch.Tensor] = None
        self._cov_count = 0
        self._act_budget = 2000

    # -------------------------------------------------------- adapter setup
    def set_adapter_basis(self, A: torch.Tensor) -> None:
        """Install a frozen input basis ``A`` (rows x in) and reset ``B`` to 0."""
        A = A.detach().to(self.delta_prev.device, self.delta_prev.dtype)
        self.lora_A = A
        rows = A.shape[0]
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, rows, device=A.device, dtype=A.dtype)
        )

    @property
    def adapter_rank(self) -> int:
        return self.lora_A.shape[0]

    def current_delta(self) -> torch.Tensor:
        """Effective ``dW`` of the *current* adapter: ``scaling * B @ A``."""
        if self.adapter_rank == 0:
            return torch.zeros(self.out_features, self.in_features,
                               device=self.delta_prev.device, dtype=self.delta_prev.dtype)
        return self.scaling * (self.lora_B @ self.lora_A)

    # --------------------------------------------------------------- merge
    def fold_current_into_prev(self) -> None:
        """Fold ``B @ A`` into the running delta and clear the current adapter."""
        with torch.no_grad():
            self.delta_prev = self.delta_prev + self.current_delta().detach()
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, 0,
                                               device=self.delta_prev.device,
                                               dtype=self.delta_prev.dtype))
        self.lora_A = torch.zeros(0, self.in_features,
                                  device=self.delta_prev.device, dtype=self.delta_prev.dtype)

    # ---------------------------------------------------------------- bank
    def snapshot_to_bank(self) -> None:
        self._bank_A.append(self.lora_A.detach().clone())
        self._bank_B.append(self.lora_B.detach().clone())

    def set_route_weights(self, weights: Optional[torch.Tensor]) -> None:
        self._route_weights = None if weights is None else \
            weights.detach().to(self.delta_prev.device)

    @property
    def bank_size(self) -> int:
        return len(self._bank_A)

    # ------------------------------------------------- activation capture
    def start_collecting(self, budget: int = 2000) -> None:
        self._collecting = True
        self._cov = None
        self._cov_count = 0
        self._act_budget = budget

    def stop_collecting(self):
        """Return ``(C, count)`` -- the accumulated ``in x in`` covariance and
        the number of rows that went into it.  ``C`` stays on the device it was
        accumulated on (typically the GPU) so the eigensolve runs there too."""
        self._collecting = False
        C = self._cov
        n = self._cov_count
        self._cov = None
        self._cov_count = 0
        if C is None:
            return torch.zeros(self.in_features, self.in_features), 0
        return C.float(), n

    def _maybe_capture(self, x: torch.Tensor) -> None:
        if not self._collecting or self._cov_count >= self._act_budget:
            return
        flat = x.reshape(-1, self.in_features).detach().float()
        room = self._act_budget - self._cov_count
        if flat.shape[0] > room:
            flat = flat[:room]
        # Streaming outer-product accumulation: C += X^T X  (fixed in x in).
        contrib = flat.t() @ flat
        if self._cov is None:
            self._cov = contrib
        else:
            self._cov = self._cov + contrib
        self._cov_count += flat.shape[0]

    # ------------------------------------------------------------- forward
    def _delta_out(self, x, A, B):
        # scaling * (dropout(x) A^T) B^T  == x @ (scaling B A)^T
        return (self.lora_dropout(x) @ A.t() @ B.t()) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._bypass:
            return self.base(x)          # pure frozen backbone (no adapters)
        self._maybe_capture(x)
        out = self.base(x)
        # Running merged delta (0 in bank mode).
        if self.delta_prev.abs().sum() > 0:
            out = out + F.linear(x, self.delta_prev)
        # Bank composition (inference-time routing / zero-shot blend).
        if self._route_weights is not None and self.bank_size > 0:
            for wt, A, B in zip(self._route_weights, self._bank_A, self._bank_B):
                if A.shape[0] > 0:
                    out = out + wt * self._delta_out(x, A, B)
            return out
        # Current active adapter (training / merge inference before fold).
        if self.adapter_rank > 0:
            out = out + self._delta_out(x, self.lora_A, self.lora_B)
        return out

    # ----------------------------------------------------------- persistence
    def adapter_state(self) -> dict:
        return {
            "delta_prev": self.delta_prev.detach().cpu(),
            "bank_A": [t.cpu() for t in self._bank_A],
            "bank_B": [t.cpu() for t in self._bank_B],
            "memory": self.memory.state_dict(),
            "scaling": self.scaling,
        }

    def load_adapter_state(self, state: dict) -> None:
        self.delta_prev = state["delta_prev"].to(self.delta_prev.device)
        self._bank_A = [t.to(self.delta_prev.device) for t in state["bank_A"]]
        self._bank_B = [t.to(self.delta_prev.device) for t in state["bank_B"]]
        self.memory.load_state_dict(state["memory"])
        self.scaling = state["scaling"]


# ---------------------------------------------------------------- injection
def _get_submodule(root: nn.Module, path: str) -> nn.Module:
    mod = root
    for part in path.split("."):
        mod = getattr(mod, part)
    return mod


def _set_submodule(root: nn.Module, path: str, new: nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new)


def inject_instil_lora(model: nn.Module, config, verbose: bool = False) -> List[str]:
    """Replace matching ``nn.Linear`` layers with :class:`InstilLoRALinear`.

    A layer is wrapped iff any string in ``config.target_modules`` matches the
    final component of its dotted name (mirroring SAPT's q/v choice).  Base
    weights are frozen; only the adapters' ``B`` train.  Returns wrapped paths.
    """
    to_wrap = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            leaf = name.split(".")[-1]
            if any(t == leaf or t in leaf for t in config.target_modules):
                to_wrap.append(name)
    for name in to_wrap:
        base = _get_submodule(model, name)
        wrapped = InstilLoRALinear(
            base, r=config.lora_r, alpha=config.lora_alpha,
            dropout=config.lora_dropout, name=name,
        )
        _set_submodule(model, name, wrapped)
        if verbose:
            print(f"[instil] wrapped {name} ({base.in_features}->{base.out_features})")
    for n, p in model.named_parameters():
        p.requires_grad = n.endswith("lora_B")
    return to_wrap


def iter_instil_layers(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, InstilLoRALinear):
            yield name, module


from contextlib import contextmanager


@contextmanager
def bypass_adapters(model: nn.Module):
    """Temporarily run the backbone with all adapters disabled (frozen base).

    Used to compute drift-free instruction prototypes (paper §3/§8): the
    encoder should see the *pretrained* representation, not one perturbed by the
    adapters currently installed.
    """
    layers = [l for _, l in iter_instil_layers(model)]
    prev = [l._bypass for l in layers]
    for l in layers:
        l._bypass = True
    try:
        yield
    finally:
        for l, p in zip(layers, prev):
            l._bypass = p
