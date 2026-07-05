"""Unit tests for Instil's structural guarantees.

Run with:   pytest tests/            (or)   python tests/test_core.py

These cover the pieces whose correctness is *provable*, so they should pass
exactly (up to float tolerance):

  * metrics  -- OP / Forgetting / BWT / FWT match hand computation
  * gate     -- monotone in instruction similarity, crosses 0.5 at rho0, floors
  * subspace -- free directions are orthogonal to the occupied basis (the core
                of Prop. 1), and the free projector annihilates occupied vectors
  * end2end  -- MERGE mode with the gate off leaves a prior task's loss unchanged
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from instil.metrics import continual_metrics
from instil.gate import InstructionGate, fit_rho0
from instil.subspace import SubspaceMemory


def test_metrics():
    R = [[80., 0., 0.],
         [70., 85., 0.],
         [60., 80., 90.]]
    m = continual_metrics(R)
    assert abs(m["OP"] - (60 + 80 + 90) / 3) < 1e-6
    assert abs(m["Forgetting"] - 12.5) < 1e-6           # (20 + 5)/2
    assert abs(m["BWT"] - (-25.0 / 3)) < 1e-6           # [-20,-5,0]/3
    assert abs(m["FWT"] - (85.0 - 50.94)) < 1e-6        # mean(diag) - baseline
    # with an explicit per-task baseline
    m2 = continual_metrics(R, single_task_baseline=[80., 85., 90.])
    assert abs(m2["FWT"] - 0.0) < 1e-6                  # diag == baseline


def test_gate_monotone_and_floor():
    e2 = lambda v: torch.tensor(v, dtype=torch.float32)
    p_t = e2([1.0, 0.0])
    priors = [e2([1.0, 0.0]), e2([0.0, 1.0]), e2([-1.0, 0.0])]  # cos = 1, 0, -1
    gate = InstructionGate(slope_a=10.0, rho0=0.0, floor=1e-3)
    g = gate.gamma(p_t, priors)
    assert g[0] > g[1] > g[2]                 # monotone in similarity
    assert abs(g[1] - 0.5) < 1e-3             # crosses 0.5 at rho0
    assert g[2] == 0.0                        # floored to exact isolation


def test_fit_rho0():
    sims = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    aligns = sims.clone()                      # perfectly monotone, crosses 0 at 0
    assert abs(fit_rho0(sims, aligns) - 0.0) < 1e-5
    assert abs(fit_rho0(sims, sims - 0.5) - 0.5) < 1e-5   # zero-crossing shifted


def test_subspace_free_orthogonality():
    torch.manual_seed(0)
    d = 6
    mem = SubspaceMemory(d, energy_threshold=0.999, rank_cap=6)
    # Task 0 occupies span(e0, e1).
    acts0 = torch.randn(200, 2) @ torch.eye(6)[:2]     # rows in span(e0,e1)
    mem.add_task(acts0, 0)
    assert mem.rank >= 2
    # A new task cares about span(e1, e2); its free directions must be ⊥ U_0.
    acts1 = torch.randn(200, 2) @ torch.eye(6)[1:3]
    F = mem.free_directions(acts1, rank=4)
    leak = (mem.U.t() @ F).abs().max().item()
    assert leak < 1e-5, f"free directions leak into occupied space: {leak}"
    # The free projector annihilates any occupied vector (core of Prop. 1).
    u = mem.U[:, 0]
    assert mem.project_free(u.unsqueeze(0)).abs().max().item() < 1e-5


def test_end_to_end_non_forgetting():
    import torch.nn as nn
    from instil import InstilConfig, Instil, inject_instil_lora, HashingEncoder
    from instil.trainer import ContinualTrainer

    torch.manual_seed(0)
    d_in, d_out, k = 16, 4, 2

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(d_in, d_out, bias=False)
            nn.init.zeros_(self.proj.weight)

        def forward(self, x):
            return self.proj(x)

    def make(seed):
        g = torch.Generator().manual_seed(seed)
        S = torch.linalg.qr(torch.randn(d_in, k, generator=g))[0]
        W = torch.randn(d_out, d_in, generator=g)
        def sample(n):
            Z = torch.randn(n, k, generator=g)
            X = Z @ S.t()
            return X, X @ W.t()
        return sample

    cfg = InstilConfig(lora_r=k, lora_alpha=2 * k, target_modules=["proj"],
                       mode="merge", energy_threshold=0.999, rho0=2.0)  # gate off
    model = TinyNet()
    inject_instil_lora(model, cfg)
    instil = Instil(model, HashingEncoder(64), cfg)

    def mse(m, b):
        x, y = b
        return ((m(x) - y) ** 2).mean()

    trainer = ContinualTrainer(instil, mse, epochs=200, lr=1e-2, device="cpu",
                               grad_clip=None, forward_fn=lambda m, b: m(b[0]))

    s0, s1 = make(1), make(2)
    Xtr0, Ytr0 = s0(128); Xte0, Yte0 = s0(128)
    Xtr1, Ytr1 = s1(128)
    tr0 = [(Xtr0[i:i+32], Ytr0[i:i+32]) for i in range(0, 128, 32)]
    tr1 = [(Xtr1[i:i+32], Ytr1[i:i+32]) for i in range(0, 128, 32)]

    trainer.learn_task("task zero summarize", tr0, collect_loader=tr0)
    before = float(((model(Xte0) - Yte0) ** 2).mean())
    trainer.learn_task("task one classify", tr1, collect_loader=tr1)
    after = float(((model(Xte0) - Yte0) ** 2).mean())
    assert abs(after - before) < 1e-3, f"forgetting: {before} -> {after}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL TESTS PASSED")
