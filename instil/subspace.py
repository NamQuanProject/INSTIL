"""Occupied-subspace bookkeeping -- the GPM half of Instil (paper §5.1, §8).

For every tracked layer we maintain an orthonormal basis ``U`` (shape
``in_features x R``) of the *input-activation* subspace occupied by all tasks
seen so far, plus per-task column ranges so the gate can address one task's
block at a time.

Key objects
-----------
* After task j we collect ~2k input-activation vectors, take the top singular
  directions capturing ``energy_threshold`` of the energy (capped at
  ``rank_cap``), orthogonalise them against the existing basis, and append.
* The free-space projector is ``P^perp = I - U U^T``.  We never materialise the
  ``d x d`` matrix: everything is done with the thin ``in x R`` basis via the
  identity  ``G P^perp = G - (G U) U^T``  (paper §8, "Efficient gated step").

This module is pure linear algebra; it has no notion of instructions or gates.
The gate weights are applied in :mod:`instil.update`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class _TaskBlock:
    """Column range ``[start, end)`` of ``U`` occupied by one task."""
    task_id: int
    start: int
    end: int


class SubspaceMemory:
    """Growing orthonormal basis of occupied input-activation directions.

    One instance per tracked layer.  All tensors are kept on CPU in float32 by
    default (the bases are tiny -- ``in x R`` with ``R`` at most a few hundred).
    """

    def __init__(self, in_features: int,
                 energy_threshold: float = 0.95,
                 rank_cap: int = 16,
                 dtype: torch.dtype = torch.float32,
                 device: str = "cpu"):
        self.in_features = in_features
        self.energy_threshold = energy_threshold
        self.rank_cap = rank_cap
        self.dtype = dtype
        self.device = device
        # U: (in_features, R) orthonormal columns; empty until the first task.
        self.U: torch.Tensor = torch.zeros(in_features, 0, dtype=dtype, device=device)
        self.blocks: List[_TaskBlock] = []

    # ------------------------------------------------------------------ state
    @property
    def rank(self) -> int:
        return self.U.shape[1]

    def block_for(self, task_id: int) -> Optional[_TaskBlock]:
        for b in self.blocks:
            if b.task_id == task_id:
                return b
        return None

    def basis_for(self, task_id: int) -> torch.Tensor:
        """Columns U_j for a single prior task j (shape ``in x r_j``)."""
        b = self.block_for(task_id)
        if b is None:
            return torch.zeros(self.in_features, 0, dtype=self.dtype, device=self.device)
        return self.U[:, b.start:b.end]

    # -------------------------------------------------------------- projectors
    def project_free(self, G: torch.Tensor) -> torch.Tensor:
        """Return ``G P^perp = G - (G U) U^T`` (rows of G live in input space).

        ``G`` has shape ``(*, in_features)``; the projection acts on the last
        (input) dimension.  With an empty basis this is the identity.
        """
        if self.rank == 0:
            return G
        U = self.U.to(G.dtype).to(G.device)
        return G - (G @ U) @ U.t()

    def project_gated(self, G: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """Instruction-gated projection ``G M`` with ``M = I - sum_j (1-gamma_j) U_j U_j^T``.

        This is the linear operator of Eq. (1): the free subspace is kept in
        full, and occupied block ``j`` is kept with coefficient ``gamma_j``.
        ``gamma_j = 0`` recovers pure isolation (GPM/InfLoRA); ``gamma_j = 1``
        is full sharing.

        Parameters
        ----------
        G : (*, in_features) tensor -- the raw update / gradient (input side).
        gamma : (num_prior_tasks,) tensor of gate values, indexed by block order.
        """
        if self.rank == 0:
            return G
        U = self.U.to(G.dtype).to(G.device)
        # Per-column shrink coefficient (1 - gamma_j) broadcast over each block.
        shrink = torch.ones(self.rank, dtype=G.dtype, device=G.device)
        for b, g in zip(self.blocks, gamma.to(G.dtype).to(G.device)):
            shrink[b.start:b.end] = (1.0 - g)
        # G M = G - ((G U) * shrink) U^T
        GU = G @ U                       # (*, R)
        return G - (GU * shrink) @ U.t()

    # ------------------------------------------------- directions (no store)
    @torch.no_grad()
    def _top_directions(self, activations: torch.Tensor, rank: int) -> torch.Tensor:
        """Top right-singular vectors of ``activations`` (shape ``in x k``)."""
        X = activations.to(self.dtype).to(self.device)
        try:
            _, S, Vh = torch.linalg.svd(X, full_matrices=False)
        except Exception:  # pragma: no cover
            _, S, Vh = torch.svd(X)
            Vh = Vh.t()
        V = Vh.t()
        energy = S ** 2
        total = energy.sum().clamp(min=1e-12)
        cumfrac = torch.cumsum(energy, dim=0) / total
        r = int(torch.searchsorted(cumfrac, torch.tensor(self.energy_threshold)).item()) + 1
        r = max(1, min(r, rank, V.shape[1]))
        return V[:, :r]

    @torch.no_grad()
    def free_directions(self, activations: torch.Tensor, rank: int) -> torch.Tensor:
        """Orthonormal top directions of the activations *projected to free space*.

        These are the null-space (safe) rows of the frozen adapter basis ``A``
        for a new task: the directions the task cares about, with everything
        already occupied by prior tasks removed (so they are orthogonal to
        ``span(U_<t)``).  Shape ``in x r_free``.
        """
        V = self._top_directions(activations, rank)
        if self.rank > 0:
            V = V - self.U @ (self.U.t() @ V)   # remove prior-occupied component
        Q, _ = torch.linalg.qr(V)
        # keep only well-conditioned, genuinely-free columns
        keep = []
        for c in range(Q.shape[1]):
            col = Q[:, c]
            if self.rank > 0:
                col = col - self.U @ (self.U.t() @ col)
            if keep:
                Qk = torch.stack(keep, dim=1)
                col = col - Qk @ (Qk.t() @ col)
            n = col.norm()
            if n > 1e-6:
                keep.append(col / n)
        if not keep:
            return torch.zeros(self.in_features, 0, dtype=self.dtype, device=self.device)
        return torch.stack(keep, dim=1)

    # ---------------------------------------------------------- basis growth
    @torch.no_grad()
    def add_task(self, activations: torch.Tensor, task_id: int) -> int:
        """Grow the basis with directions occupied by ``task_id``.

        Parameters
        ----------
        activations : (N, in_features) matrix of collected layer inputs.
        task_id : id of the task these activations belong to.

        Returns the number of new orthonormal columns appended (``r_t``).

        Procedure (GPM):
          1. SVD the (mean-removed) activation matrix.
          2. Keep the smallest set of leading directions whose captured energy
             reaches ``energy_threshold`` (capped at ``rank_cap``).
          3. Remove the component already spanned by the existing basis,
             re-orthonormalise (QR), and append.
        """
        X = activations.to(self.dtype).to(self.device)
        if X.dim() != 2 or X.shape[1] != self.in_features:
            raise ValueError(
                f"expected (N, {self.in_features}) activations, got {tuple(X.shape)}"
            )
        # Right singular vectors of X == eigenvectors of the input covariance.
        # X = U S V^T ;  the V columns (in-space) are the directions we store.
        try:
            _, S, Vh = torch.linalg.svd(X, full_matrices=False)
        except Exception:  # pragma: no cover - numerical fallback
            _, S, Vh = torch.svd(X)
            Vh = Vh.t()
        V = Vh.t()                      # (in_features, k)
        energy = (S ** 2)
        total = energy.sum().clamp(min=1e-12)
        cumfrac = torch.cumsum(energy, dim=0) / total
        r = int(torch.searchsorted(cumfrac, torch.tensor(self.energy_threshold)).item()) + 1
        r = max(1, min(r, self.rank_cap, V.shape[1]))
        candidate = V[:, :r]            # (in_features, r)

        # Orthogonalise against the existing basis, then re-orthonormalise.
        if self.rank > 0:
            candidate = candidate - self.U @ (self.U.t() @ candidate)
        # QR gives an orthonormal basis of the (residual) column space.
        Q, _ = torch.linalg.qr(candidate)
        # Drop numerically-zero columns (already covered by prior tasks).
        keep = []
        for c in range(Q.shape[1]):
            col = Q[:, c]
            if self.rank > 0:
                col = col - self.U @ (self.U.t() @ col)
            if len(keep) > 0:
                Qk = torch.stack(keep, dim=1)
                col = col - Qk @ (Qk.t() @ col)
            n = col.norm()
            if n > 1e-6:
                keep.append(col / n)
        if not keep:
            # Fully covered already: register an empty block for bookkeeping.
            self.blocks.append(_TaskBlock(task_id, self.rank, self.rank))
            return 0
        newcols = torch.stack(keep, dim=1)
        start = self.rank
        self.U = torch.cat([self.U, newcols], dim=1)
        self.blocks.append(_TaskBlock(task_id, start, self.U.shape[1]))
        return newcols.shape[1]

    # ----------------------------------------------------------- persistence
    def state_dict(self) -> dict:
        return {
            "in_features": self.in_features,
            "energy_threshold": self.energy_threshold,
            "rank_cap": self.rank_cap,
            "U": self.U.cpu(),
            "blocks": [(b.task_id, b.start, b.end) for b in self.blocks],
        }

    def load_state_dict(self, state: dict) -> None:
        self.in_features = state["in_features"]
        self.energy_threshold = state["energy_threshold"]
        self.rank_cap = state["rank_cap"]
        self.U = state["U"].to(self.dtype).to(self.device)
        self.blocks = [_TaskBlock(*t) for t in state["blocks"]]
