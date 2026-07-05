"""Instil: Instruction-Anchored Continual Learning.

Reference implementation of the method described in ``ideas/instil.pdf``:

    Instil — The Instruction Manifold Predicts Weight-Space Conflict.

Instil splits each task's low-rank update into a *null-space* part (provably
non-interfering) and an *occupied-space* part admitted in proportion to
instruction compatibility.  This turns subspace *isolation* (O-LoRA / GPM /
InfLoRA) into *certified transfer*: non-negative backward transfer (Thm. 1),
a forward-transfer bound (Thm. 2), a non-forgetting floor (Prop. 1), weak
dominance over isolation (Cor. 1), and never-forgetting routing (Prop. 2).

The package is intentionally backbone-agnostic.  It plugs into any
``torch.nn.Module`` that contains ``nn.Linear`` layers (T5, LLaMA, ...) and it
mirrors the LoRA conventions used by the bundled SAPT codebase (LoRA on the
attention ``q, v`` projections).

Module map (paper section in brackets):

    encoders.py   frozen instruction encoder E -> unit prototypes p_t        [§3, §8]
    gate.py       instruction gate  gamma = sigma(a<p_t,p_j> + b)            [§5.2]
    subspace.py   GPM occupied-subspace bookkeeping U_j, projector P^perp    [§5.1]
    lora.py       InstilLoRALinear (SAPT-compatible A/B) + injection         [§8]
    update.py     instruction-gated gradient projection (Eq. 1)             [§5.2]
    instil.py     Instil orchestrator: LearnTask / Answer, Merge & Bank      [§5.3, §7]
    metrics.py    OP / Forgetting / BWT / FWT (matches SAPT score.py)        [§9]
    law.py        Experiment 1 -- validate the Instruction-Gradient Law      [§4, §9]
    trainer.py    thin continual-training loop wiring it all together        [§7]
"""

from .config import InstilConfig
from .gate import InstructionGate, fit_rho0
from .subspace import SubspaceMemory
from .encoders import InstructionEncoder, MeanPooledBackboneEncoder, HashingEncoder
from .lora import InstilLoRALinear, inject_instil_lora, iter_instil_layers
from .instil import Instil
from .metrics import continual_metrics, sapt_metrics
from .law import measure_gradient_alignment, validate_law

__all__ = [
    "InstilConfig",
    "InstructionGate",
    "fit_rho0",
    "SubspaceMemory",
    "InstructionEncoder",
    "MeanPooledBackboneEncoder",
    "HashingEncoder",
    "InstilLoRALinear",
    "inject_instil_lora",
    "iter_instil_layers",
    "Instil",
    "continual_metrics",
    "sapt_metrics",
    "measure_gradient_alignment",
    "validate_law",
]

__version__ = "0.1.0"
