"""Instil orchestrator -- Algorithms 1 & 2 (paper §5.3, §7).

``Instil`` drives the continual stream over a backbone already injected with
:class:`~instil.lora.InstilLoRALinear` layers.

* :meth:`learn_task` -- Algorithm 1.  Compute the prototype ``p_t``; from a
  pre-pass over the task's inputs build, per layer, the **frozen** gated basis

      A = [ free directions  |  gamma_{t,j} * U_j for aligned prior tasks j ]

  install it, train only ``B``, grow the occupied subspaces, then fold (merge)
  or snapshot (bank) the adapter.
* :meth:`answer` -- Algorithm 2 (Bank).  Embed the query instruction, soft-route
  over stored adapters (Prop. 2) and compose them zero-shot (§5.4).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from .config import InstilConfig
from .encoders import InstructionEncoder
from .gate import InstructionGate
from .lora import InstilLoRALinear, iter_instil_layers
from .logging_utils import get_logger, tqdm_iter
from .update import project_instil_gradients

ForwardFn = Callable[[nn.Module, object], None]


class Instil:
    def __init__(self, model: nn.Module, encoder: InstructionEncoder,
                 config: Optional[InstilConfig] = None):
        self.model = model
        self.encoder = encoder
        self.cfg = config or InstilConfig()
        self.gate = InstructionGate(
            slope_a=self.cfg.gate_slope_a,
            rho0=(self.cfg.rho0 if self.cfg.rho0 is not None else 0.0),
            floor=self.cfg.gate_floor,
        )
        self.prototypes: List[torch.Tensor] = []
        self.instructions: List[str] = []
        self.layers: List[InstilLoRALinear] = [l for _, l in iter_instil_layers(model)]
        if not self.layers:
            raise RuntimeError(
                "No InstilLoRALinear layers found -- call inject_instil_lora(model, cfg) first."
            )

    # --------------------------------------------------------------- gate
    @property
    def num_tasks(self) -> int:
        return len(self.prototypes)

    def gamma_vector(self, p_t: torch.Tensor) -> torch.Tensor:
        """gamma_{t,j} for all prior tasks j (Eq. 1 gate)."""
        return self.gate.gamma(p_t, self.prototypes)

    def maybe_fit_rho0(self, similarities, alignments) -> None:
        from .gate import fit_rho0
        self.gate.rho0 = fit_rho0(torch.as_tensor(similarities),
                                  torch.as_tensor(alignments))

    # ---------------------------------------------------- basis construction
    def _build_basis(self, layer: InstilLoRALinear, cov: torch.Tensor,
                     gamma: torch.Tensor) -> torch.Tensor:
        """Assemble the frozen adapter basis ``A`` (rows x in) for a task.

        ``cov`` is the streaming input covariance ``X^T X`` collected in the
        pre-pass; free directions come from its top eigenvectors (no SVD).
        """
        mem = layer.memory
        dev = cov.device
        cols: List[torch.Tensor] = []
        # Null-space (safe) directions the task cares about, orthogonal to U_<t.
        F = mem.free_directions_cov(cov, rank=self.cfg.lora_r)
        if F.shape[1] > 0:
            cols.append(F.to(dev))
        # Gated occupied directions: block j admitted with weight gamma_j.
        used = 0
        for j, g in enumerate(gamma.tolist()):
            if g < self.cfg.gate_floor:
                continue
            Uj = mem.basis_for(j)
            if Uj.shape[1] == 0:
                continue
            room = self.cfg.subspace_rank_cap - used
            if room <= 0:
                break
            Uj = Uj[:, :room]
            cols.append(float(g) * Uj.to(dev))
            used += Uj.shape[1]
        if not cols:
            # Degenerate: give the task at least one free direction.
            cols.append(torch.eye(layer.in_features, device=dev)[:, : self.cfg.lora_r])
        basis = torch.cat(cols, dim=1)          # (in, R)
        return basis.t().contiguous()           # (R, in) -- rows are directions

    # ------------------------------------------------------- warm start (FWT)
    def _nearest_prior(self, p_t: torch.Tensor) -> Optional[int]:
        if not self.prototypes:
            return None
        d = [float((p_t.flatten() - p.flatten()).norm()) for p in self.prototypes]
        return int(min(range(len(d)), key=lambda i: d[i]))

    def _warm_start(self, layer: InstilLoRALinear, j: int) -> None:
        """Init ``B`` so the current adapter matches nearest prior delta (Thm. 2).

        ``B0 = argmin_B || scaling*B*A - dW_j ||_F = dW_j (scaling*A)^+``.
        """
        if j is None or j >= layer.bank_size or layer.adapter_rank == 0:
            return
        A_j, B_j = layer._bank_A[j], layer._bank_B[j]
        if A_j.shape[0] == 0:
            return
        dW_j = layer.scaling * (B_j @ A_j)               # (out, in)
        A = layer.lora_A                                 # (R, in)
        pinv = torch.linalg.pinv(layer.scaling * A)      # (in, R)
        with torch.no_grad():
            layer.lora_B.data.copy_(dW_j @ pinv)

    # -------------------------------------------------------------- Alg. 1
    def learn_task(
        self,
        instruction: str,
        train_step: Callable[[], None],
        collect_batches: Iterable = (),
        forward_fn: Optional[ForwardFn] = None,
    ) -> torch.Tensor:
        logger = get_logger()
        p_t = self.encoder.encode_one(instruction).float()
        task_id = self.num_tasks
        gamma = self.gamma_vector(p_t)
        if task_id > 0:
            gmax = float(gamma.max()) if gamma.numel() else 0.0
            n_aligned = int((gamma >= self.cfg.gate_floor).sum())
            logger.info(f"task {task_id}: gate admits {n_aligned}/{task_id} prior "
                        f"tasks (max gamma={gmax:.3f}) | mode={self.cfg.mode}")

        # 1) Pre-pass: accumulate this task's input covariance (per layer).
        cov = self._collect_activations(collect_batches, forward_fn)

        # 2) Per layer: build & install the frozen gated basis A (uses U_<t),
        #    then grow the occupied subspace with U_t (§5.1).  Both use the same
        #    covariance, so we do them together and free C immediately -- this
        #    keeps at most one in x in covariance alive at a time on the GPU.
        for layer in self.layers:
            C, n = cov[layer.name]
            A = self._build_basis(layer, C, gamma)     # against U_<t
            layer.set_adapter_basis(A)
            if n > 0:
                layer.memory.add_task_cov(C, n, task_id)   # appends U_t
            else:
                layer.memory.add_task_cov(
                    torch.zeros(layer.in_features, layer.in_features,
                                device=C.device), 0, task_id)
        del cov                                        # release GPU covariances

        # 3) (Bank) forward-transfer warm-start from the nearest prior adapter.
        if self.cfg.mode == "bank" and self.cfg.warm_start and task_id > 0:
            j_star = self._nearest_prior(p_t)
            for layer in self.layers:
                self._warm_start(layer, j_star)

        # 4) Train B (the base and A are frozen).
        self.model.train()
        train_step()

        # 5) Consolidate: fold in place (merge) or snapshot (bank).
        for layer in self.layers:
            if self.cfg.mode == "merge":
                layer.fold_current_into_prev()
            else:
                layer.snapshot_to_bank()

        self.prototypes.append(p_t)
        self.instructions.append(instruction)
        return p_t

    def project_gradients(self) -> None:
        """No-op in this build (gate is structural); safe to call post-backward."""
        project_instil_gradients(self.model)

    def _collect_activations(self, collect_batches, forward_fn) -> Dict[str, tuple]:
        """Return ``{layer_name: (covariance in x in, row_count)}`` for the task."""
        budget = self.cfg.max_activation_samples
        for layer in self.layers:
            layer.start_collecting(budget)
        was_training = self.model.training
        self.model.eval()

        def _all_full():
            return all(l._cov_count >= l._act_budget for l in self.layers)

        with torch.no_grad():
            for batch in tqdm_iter(collect_batches, desc="collect", leave=False):
                if forward_fn is not None:
                    forward_fn(self.model, batch)
                else:
                    self.model(**batch)
                # Each long example fills the row budget fast; once every layer
                # has enough, stop -- no need to sweep the whole train set.
                if _all_full():
                    break
        if was_training:
            self.model.train()
        return {layer.name: layer.stop_collecting() for layer in self.layers}

    # -------------------------------------------------------------- Alg. 2
    def routing_weights(self, p_star: torch.Tensor) -> torch.Tensor:
        """Soft routing weights ``w_t ~ exp(-||p*-p_t||^2 / tau)`` (Alg. 2)."""
        if not self.prototypes:
            return torch.zeros(0)
        P = torch.stack([p.flatten() for p in self.prototypes], dim=0)
        d2 = ((P - p_star.flatten().unsqueeze(0)) ** 2).sum(dim=1)
        logits = -d2 / max(self.cfg.routing_temperature, 1e-6)
        return torch.softmax(logits, dim=0)

    @contextmanager
    def answer(self, instruction: str):
        """Activate the composed adapter for a query instruction (§5.4).

            with instil.answer(query):
                out = model.generate(**inputs)

        For a seen instruction the weights concentrate on its own adapter
        (Prop. 2); for a novel blend the composed delta realises the blend with
        no training.
        """
        if self.cfg.mode != "bank":
            yield None
            return
        p_star = self.encoder.encode_one(instruction).float()
        w = self.routing_weights(p_star)
        for layer in self.layers:
            layer.set_route_weights(w)
        try:
            yield w
        finally:
            for layer in self.layers:
                layer.set_route_weights(None)

    def route_to_task(self, task_id: int):
        """Hard oracle route to a single stored adapter (task-ID upper bound)."""
        w = torch.zeros(self.num_tasks)
        w[task_id] = 1.0
        for layer in self.layers:
            layer.set_route_weights(w)

    def clear_routing(self) -> None:
        for layer in self.layers:
            layer.set_route_weights(None)

    # ----------------------------------------------------------- persistence
    def state_dict(self) -> dict:
        return {
            "config": self.cfg.__dict__,
            "rho0": self.gate.rho0,
            "prototypes": [p.cpu() for p in self.prototypes],
            "instructions": self.instructions,
            "layers": {name: layer.adapter_state()
                       for name, layer in iter_instil_layers(self.model)},
        }

    def load_state_dict(self, state: dict) -> None:
        self.gate.rho0 = state["rho0"]
        self.prototypes = [p.float() for p in state["prototypes"]]
        self.instructions = list(state["instructions"])
        by_name = dict(iter_instil_layers(self.model))
        for name, layer_state in state["layers"].items():
            if name in by_name:
                by_name[name].load_adapter_state(layer_state)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location="cpu"))
