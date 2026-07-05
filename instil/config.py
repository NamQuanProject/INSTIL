"""Configuration for Instil (paper §5, §8)."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class InstilConfig:
    """All Instil hyper-parameters in one place.

    Defaults follow the "Implementation Details" section (§8): LoRA rank 8-16
    on the attention q, v projections, occupied subspaces capturing >=0.95 of
    the input-activation energy, and a gate fitted from the Law on the first
    few tasks.
    """

    # ---- LoRA adapter (§8) -------------------------------------------------
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    # Substring match against module names, mirroring SAPT (q, v projections).
    target_modules: List[str] = field(default_factory=lambda: ["q", "v"])

    # ---- Occupied-subspace bookkeeping (§5.1, §8) --------------------------
    # Keep top singular directions until this fraction of input-activation
    # energy is captured, capped at ``subspace_rank_cap`` directions per task.
    energy_threshold: float = 0.95
    subspace_rank_cap: int = 16
    # How many input-activation rows to collect per task per tracked layer.
    max_activation_samples: int = 2000

    # ---- Instruction gate  gamma = sigma(a<p_t,p_j> + b)  (§5.2) -----------
    # b is fixed so that gamma > 0.5 only when <p_t,p_j> >= rho0, i.e. b = -a*rho0.
    gate_slope_a: float = 10.0
    # If ``rho0`` is None it is fitted from the Law (§4) on the first
    # ``law_fit_tasks`` tasks; otherwise this fixed value is used.
    rho0: Optional[float] = None
    law_fit_tasks: int = 3
    # Hard floor: gates below this are clamped to exactly 0 (pure isolation on
    # that block) so the non-forgetting guarantee is numerically exact.
    gate_floor: float = 1e-3

    # ---- Mode (§5.3) -------------------------------------------------------
    # "merge": one shared adapter updated in place (memory does not grow).
    # "bank":  additionally store tiny per-task deltas for training-free
    #          routing (Alg. 2) and zero-shot composition (§5.4).
    mode: str = "bank"

    # ---- Routing / composition (§5.4) --------------------------------------
    routing_temperature: float = 0.1  # tau in  w_t ~ exp(-||p*-p_t||^2 / tau)

    # ---- Forward transfer warm-start (§5.4, Thm. 2) ------------------------
    # Bank mode only: warm-start a new task from the nearest prior adapter.
    warm_start: bool = True

    # ---- Prototypes --------------------------------------------------------
    l2_normalize_prototypes: bool = True

    def __post_init__(self):
        assert self.mode in {"merge", "bank"}, f"unknown mode {self.mode!r}"
        assert 0.0 < self.energy_threshold <= 1.0
        assert self.lora_r > 0
