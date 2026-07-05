"""The instruction gate  gamma_{t,j} = sigma(a<p_t,p_j> + b)  (paper §5.2).

This is the whole trick: a scalar, training-free, a-priori quantity -- the
cosine similarity of two task instructions -- decides how much of task t's
update is allowed to reinforce prior task j's occupied subspace.

* ``a > 0`` is the reinforcement sharpness.
* ``b = -a * rho0`` fixes the zero-crossing so that ``gamma > 0.5`` exactly when
  ``<p_t, p_j> >= rho0`` (reinforce only instruction-aligned tasks; Eq. 1).
* ``rho0`` is the zero-crossing of the Instruction-Gradient Alignment Law
  (Empirical Law 1, §4).  It is either supplied or fitted from a handful of
  early tasks (:func:`fit_rho0`).

Setting ``a -> inf`` recovers a hard threshold; ``gamma == 0`` on every block
recovers pure isolation (O-LoRA / GPM / InfLoRA) -- the ``gamma == 0`` special
case of Corollary 1.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch


class InstructionGate:
    """Callable gate producing per-prior-task admission coefficients."""

    def __init__(self, slope_a: float = 10.0, rho0: float = 0.0,
                 floor: float = 1e-3):
        assert slope_a > 0, "gate slope a must be > 0 (monotone in similarity)"
        self.a = float(slope_a)
        self.rho0 = float(rho0)
        self.floor = float(floor)

    @property
    def b(self) -> float:
        # Bias fixed so the sigmoid crosses 0.5 at similarity == rho0.
        return -self.a * self.rho0

    def gamma(self, p_t: torch.Tensor, prototypes: Sequence[torch.Tensor]) -> torch.Tensor:
        """Return ``gamma`` of shape ``(len(prototypes),)`` for prior tasks.

        ``p_t`` and each prior prototype are assumed L2-normalised, so the inner
        product is a cosine similarity in ``[-1, 1]``.
        """
        if len(prototypes) == 0:
            return torch.zeros(0)
        P = torch.stack([p.flatten() for p in prototypes], dim=0)  # (J, e)
        sims = P @ p_t.flatten()                                   # (J,)
        g = torch.sigmoid(self.a * sims + self.b)
        # Hard floor -> exact isolation on weakly-aligned blocks (keeps the
        # non-forgetting guarantee numerically exact, Prop. 1).
        g = torch.where(g < self.floor, torch.zeros_like(g), g)
        return g

    def __repr__(self) -> str:
        return f"InstructionGate(a={self.a}, rho0={self.rho0}, b={self.b:.3f})"


def fit_rho0(similarities: torch.Tensor, alignments: torch.Tensor) -> float:
    """Estimate the Law's zero-crossing rho0 from paired measurements (§4).

    Given instruction similarities ``<p_t,p_j>`` and *measured* gradient
    alignments ``<grad L_j, grad L_t>_{U_j}`` across task pairs, fit a simple
    monotone (linear) ``phi`` and return the similarity at which the fitted
    alignment crosses zero.  This is the cheap, paper-deciding calibration of
    §9 experiment 1.

    Falls back to 0.0 if the data is degenerate.
    """
    x = similarities.flatten().float()
    y = alignments.flatten().float()
    if x.numel() < 2 or torch.allclose(x, x.mean()):
        return 0.0
    # Least-squares line y = m x + c.
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom.abs() < 1e-12:
        return 0.0
    m = ((x - xm) * (y - ym)).sum() / denom
    c = ym - m * xm
    if m.abs() < 1e-12:
        return float(xm.item())
    rho0 = (-c / m).item()
    # Clamp to the valid cosine range.
    return float(max(-1.0, min(1.0, rho0)))
