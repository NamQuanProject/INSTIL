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

    # ---------------------------------------- efficient top-r eigendecomposition
    @torch.no_grad()
    def _top_eig(self, C: torch.Tensor, k: int):
        """Top-``k`` eigenpairs of a symmetric PSD matrix ``C`` (in x in).

        Uses LOBPCG -- O(in^2 * k) for the k leading directions -- instead of a
        full SVD/eigh, following the streaming, low-order-statistics philosophy
        of HESTIA's OnlineDiagonalGMM.  Falls back to a dense ``eigh`` for small
        matrices or if LOBPCG fails to converge.

        Returns ``(eigvals desc (k,), eigvecs (in, k))`` with eigvals clamped >=0.
        """
        C = C.to(self.dtype).to(self.device)
        d = C.shape[0]
        k = max(1, min(k, d))
        # Tiny ridge keeps the matrix strictly PD for LOBPCG stability.
        jitter = 1e-6 * (C.diagonal().mean().abs() + 1e-12)
        use_dense = (d <= 64) or (k >= d - 1)
        if not use_dense:
            try:
                Cj = C + jitter * torch.eye(d, dtype=self.dtype, device=self.device)
                evals, evecs = torch.lobpcg(Cj, k=k, largest=True)
                return evals.clamp(min=0.0), evecs
            except Exception:  # pragma: no cover - fall back to dense
                use_dense = True
        evals, evecs = torch.linalg.eigh(C)          # ascending
        evals = evals.flip(0)[:k].clamp(min=0.0)
        evecs = evecs.flip(1)[:, :k]
        return evals, evecs

    def _rank_for_energy(self, evals: torch.Tensor, total_energy: torch.Tensor,
                         rank: int) -> int:
        """Smallest #directions whose captured energy reaches the threshold."""
        total = total_energy.clamp(min=1e-12)
        cumfrac = torch.cumsum(evals, dim=0) / total
        r = int(torch.searchsorted(cumfrac, torch.tensor(
            self.energy_threshold, dtype=cumfrac.dtype, device=cumfrac.device)).item()) + 1
        return max(1, min(r, rank, evals.numel()))

    def _orthonormalize_against_prior(self, V: torch.Tensor) -> torch.Tensor:
        """Modified Gram-Schmidt of ``V``'s columns against ``self.U`` and each
        other; drops numerically-zero (already-covered) columns."""
        keep = []
        for c in range(V.shape[1]):
            col = V[:, c]
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

    # ---------------------------------------- covariance-based primary methods
    @torch.no_grad()
    def free_directions_cov(self, C: torch.Tensor, rank: int) -> torch.Tensor:
        """Free-space (null) directions from a streaming covariance ``C=X^T X``.

        The top directions of the task's inputs, with everything already
        occupied by prior tasks removed (orthogonal to ``span(U_<t)``).  These
        are the safe rows of the frozen adapter basis ``A`` (shape ``in x r``).
        """
        C = C.to(self.dtype).to(self.device)
        if C.abs().sum() == 0:
            return torch.zeros(self.in_features, 0, dtype=self.dtype, device=self.device)
        _, V = self._top_eig(C, rank)
        return self._orthonormalize_against_prior(V)

    @torch.no_grad()
    def add_task_cov(self, C: torch.Tensor, count: int, task_id: int) -> int:
        """Grow the occupied basis from a streaming covariance ``C = X^T X``.

        Energy is read directly from the eigenvalues and ``trace(C)`` (== total
        energy), so no full spectrum is needed.  Returns #new columns appended.
        """
        C = C.to(self.dtype).to(self.device)
        if C.shape != (self.in_features, self.in_features):
            raise ValueError(
                f"expected ({self.in_features},{self.in_features}) covariance, "
                f"got {tuple(C.shape)}")
        if count <= 0 or C.abs().sum() == 0:
            # Empty/degenerate task: register an empty block for bookkeeping.
            self.blocks.append(_TaskBlock(task_id, self.rank, self.rank))
            return 0
        evals, V = self._top_eig(C, self.rank_cap)
        r = self._rank_for_energy(evals, C.diagonal().sum(), self.rank_cap)
        newcols = self._orthonormalize_against_prior(V[:, :r])
        if newcols.shape[1] == 0:
            self.blocks.append(_TaskBlock(task_id, self.rank, self.rank))
            return 0
        start = self.rank
        self.U = torch.cat([self.U, newcols], dim=1)
        self.blocks.append(_TaskBlock(task_id, start, self.U.shape[1]))
        return newcols.shape[1]

    # ----------------------------------- raw-activation wrappers (compatibility)
    @staticmethod
    def covariance(activations: torch.Tensor) -> torch.Tensor:
        """``X^T X`` for an ``(N, in)`` activation matrix (the only reduction kept)."""
        X = activations.float()
        return X.t() @ X

    @torch.no_grad()
    def free_directions(self, activations: torch.Tensor, rank: int) -> torch.Tensor:
        return self.free_directions_cov(self.covariance(activations), rank)

    @torch.no_grad()
    def add_task(self, activations: torch.Tensor, task_id: int) -> int:
        """Grow the basis from a raw ``(N, in)`` activation matrix (builds ``X^T X``)."""
        if activations.dim() != 2 or activations.shape[1] != self.in_features:
            raise ValueError(
                f"expected (N, {self.in_features}) activations, got {tuple(activations.shape)}")
        return self.add_task_cov(self.covariance(activations),
                                 activations.shape[0], task_id)

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
