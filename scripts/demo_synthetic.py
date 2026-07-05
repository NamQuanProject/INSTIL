#!/usr/bin/env python
"""Self-contained CPU demo & self-test of Instil (no downloads, torch only).

It builds a stream of synthetic low-rank regression tasks, each with a short
natural-language "instruction", and exercises the whole method:

  * MERGE mode with the gate forced OFF  -> verifies Prop. 1 (exact
    non-forgetting): a prior task's loss is *unchanged* after later tasks.
  * BANK mode with the real instruction gate -> verifies Prop. 2 (routing picks
    the right adapter) and §5.4 zero-shot composition of a blended instruction.
  * Prints the instruction-gate matrix gamma_{t,j} and the CL metrics.

Run:  python scripts/demo_synthetic.py
Exit code 0 and "ALL CHECKS PASSED" means the structural guarantees hold.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from instil import InstilConfig, Instil, inject_instil_lora, HashingEncoder
from instil.trainer import ContinualTrainer

torch.manual_seed(0)

D_IN, D_OUT, K = 24, 6, 3           # input dim, output dim, per-task subspace rank
N_TRAIN, N_TEST = 256, 128

# Two clusters of instructions; within a cluster the words overlap (high cosine).
TASKS = [
    ("summarize the paragraph into a short abstract summary", "A"),
    ("paraphrase and rewrite the text passage into a summary", "A"),
    ("classify the sentence sentiment as positive or negative label", "B"),
    ("label the review sentiment positive negative classification", "B"),
]


class TinyNet(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out, bias=False)
        nn.init.zeros_(self.proj.weight)   # base delta starts at 0 (adapter-only)

    def forward(self, x):
        return self.proj(x)


def make_task(seed):
    g = torch.Generator().manual_seed(seed)
    S = torch.linalg.qr(torch.randn(D_IN, K, generator=g))[0]   # input subspace
    W = torch.randn(D_OUT, D_IN, generator=g) * 0.5             # target map
    def sample(n):
        Z = torch.randn(n, K, generator=g)
        X = Z @ S.t()                    # inputs live in span(S) (rank K)
        Y = X @ W.t()
        return X, Y
    return sample


def loaders(sample, bs=32):
    Xtr, Ytr = sample(N_TRAIN)
    Xte, Yte = sample(N_TEST)
    train = [(Xtr[i:i+bs], Ytr[i:i+bs]) for i in range(0, N_TRAIN, bs)]
    test = (Xte, Yte)
    return train, test


def mse_loss(model, batch):
    x, y = batch
    return ((model(x) - y) ** 2).mean()


@torch.no_grad()
def eval_loss(model, test):
    x, y = test
    return float(((model(x) - y) ** 2).mean())


def build(mode, rho0):
    cfg = InstilConfig(
        lora_r=K, lora_alpha=2 * K, target_modules=["proj"], mode=mode,
        energy_threshold=0.999, subspace_rank_cap=8, rho0=rho0,
        gate_slope_a=12.0, warm_start=(mode == "bank"),
    )
    model = TinyNet(D_IN, D_OUT)
    inject_instil_lora(model, cfg)
    encoder = HashingEncoder(dim=128)
    instil = Instil(model, encoder, cfg)
    trainer = ContinualTrainer(instil, mse_loss, epochs=250, lr=1e-2,
                               device="cpu", grad_clip=None,
                               forward_fn=lambda m, b: m(b[0]))
    return model, instil, trainer


def forward_collect(m, b):
    m(b[0])


# --------------------------------------------------------------------------- #
def demo_non_forgetting():
    print("\n[1] MERGE mode, gate OFF  ->  Prop. 1 (exact non-forgetting)")
    model, instil, trainer = build("merge", rho0=2.0)  # rho0=2 => gamma==0 always
    samples = [make_task(100 + i) for i in range(len(TASKS))]
    tests = []
    baseline = []  # loss on task j right after learning task j
    R = []
    for i, ((instr, _), sample) in enumerate(zip(TASKS, samples)):
        train, test = loaders(sample)
        tests.append(test)
        trainer.learn_task(instr, train, collect_loader=train)
        baseline.append(eval_loss(model, test))
        R.append([eval_loss(model, tests[j]) for j in range(i + 1)]
                 + [0.0] * (len(TASKS) - i - 1))
    print("    diag (loss right after learning each task):",
          [f"{b:.4f}" for b in baseline])
    final = [eval_loss(model, t) for t in tests]
    print("    final loss per task after full stream:     ",
          [f"{f:.4f}" for f in final])
    drift = max(abs(final[j] - baseline[j]) for j in range(len(TASKS) - 1))
    print(f"    max backward drift on earlier tasks: {drift:.2e}  (expect ~0)")
    assert drift < 1e-3, f"non-forgetting violated: drift={drift}"
    print("    OK: earlier-task losses are unchanged by later tasks.")


def demo_gate_matrix():
    print("\n[2] Instruction gate  gamma_{t,j} = sigma(a<p_t,p_j> + b)")
    enc = HashingEncoder(dim=128)
    protos = [enc.encode_one(instr) for instr, _ in TASKS]
    from instil.gate import InstructionGate
    gate = InstructionGate(slope_a=12.0, rho0=0.3)
    print("       (rows=new task t, cols=prior task j)")
    for t in range(len(TASKS)):
        g = gate.gamma(protos[t], protos[:t])
        row = " ".join(f"{v:.2f}" for v in g.tolist())
        cl = TASKS[t][1]
        print(f"    t={t} [{cl}]  gamma={row or '(none)'}")
    print("    (within-cluster pairs get high gamma, cross-cluster ~0)")


def demo_bank_routing():
    print("\n[3] BANK mode  ->  Prop. 2 (routing) + zero-shot composition")
    model, instil, trainer = build("bank", rho0=0.3)
    samples = [make_task(100 + i) for i in range(len(TASKS))]
    tests = []
    for (instr, _), sample in zip(TASKS, samples):
        train, test = loaders(sample)
        tests.append(test)
        trainer.learn_task(instr, train, collect_loader=train)

    # Routing: answering with task i's instruction should recover task i.
    print("    routing weights argmax per task instruction:")
    ok = True
    R = []
    for i, (instr, _) in enumerate(TASKS):
        p = instil.encoder.encode_one(instr)
        w = instil.routing_weights(p)
        pick = int(w.argmax())
        ok = ok and (pick == i)
        with instil.answer(instr):
            row = [eval_loss(model, tests[j]) for j in range(len(TASKS))]
        R.append(row)
        print(f"    task {i}: argmax={pick} (want {i}), w={[round(x,2) for x in w.tolist()]}")
    assert ok, "routing did not recover the correct adapter for a seen task"
    # Diagonal (routed to own task) should be much lower than off-diagonal.
    diag = sum(R[i][i] for i in range(len(TASKS))) / len(TASKS)
    off = sum(R[i][j] for i in range(len(TASKS)) for j in range(len(TASKS)) if i != j)
    off /= (len(TASKS) * (len(TASKS) - 1))
    print(f"    mean routed-to-own loss {diag:.4f}  vs  mean cross loss {off:.4f}")
    assert diag < off, "routing to the correct task should beat cross-task"

    # Composition: a blended instruction of the two cluster-A tasks.
    blend = "summarize and paraphrase the passage into a short abstract"
    p = instil.encoder.encode_one(blend)
    w = instil.routing_weights(p)
    print(f"    zero-shot blend '{blend[:40]}...' -> weights "
          f"{[round(x,2) for x in w.tolist()]}")
    massA = w[0].item() + w[1].item()
    print(f"    mass on cluster-A adapters (tasks 0,1): {massA:.2f} (expect > 0.5)")
    assert massA > 0.5, "composition did not concentrate on the right cluster"
    print("    OK: routing is exact and composition targets the right cluster.")


def main():
    demo_non_forgetting()
    demo_gate_matrix()
    demo_bank_routing()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
