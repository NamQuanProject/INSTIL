#!/usr/bin/env python
"""Experiment 1 (§9): validate the Instruction-Gradient Alignment Law on SuperNI.

For each task we (a) build its occupied subspace ``U_j`` per tracked layer from
input activations, and (b) measure the weight-space gradient ``G_j`` at the
shared pretrained point.  We then scatter instruction similarity ``<p_t,p_j>``
against the subspace gradient cosine ``<G_j,G_t>_{U_j}`` and report the Pearson
correlation, the sign accuracy, and the fitted zero-crossing ``rho0`` -- the
"single plot that decides the paper".

Outputs ``law_points.csv`` (sim,alignment,task_j,task_t) and ``law_summary.json``.
A PNG scatter is written too if matplotlib is available.

Example
-------
    python scripts/run_law_experiment.py \
        --model_name_or_path t5-large --data_dir data \
        --task_order taskA,taskB,taskC,... --max_batches 4 --output_dir law_out
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from instil import (InstilConfig, inject_instil_lora, iter_instil_layers,
                    MeanPooledBackboneEncoder)
from instil.law import (compute_task_weight_gradients, measure_gradient_alignment,
                        validate_law)
from instil.data_superni import load_superni_task, make_loaders


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--benchmark", default="SuperNI")
    p.add_argument("--task_order", required=True)
    p.add_argument("--target_modules", default="q,v")
    p.add_argument("--max_batches", type=int, default=4,
                   help="batches per task for gradient/activation estimation")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_source_length", type=int, default=512)
    p.add_argument("--max_target_length", type=int, default=50)
    p.add_argument("--output_dir", default="law_out")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    os.makedirs(args.output_dir, exist_ok=True)
    task_order = args.task_order.split(",")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path).to(args.device)
    cfg = InstilConfig(target_modules=args.target_modules.split(","))
    inject_instil_lora(model, cfg, verbose=False)
    layer_names = [n for n, _ in iter_instil_layers(model)]
    encoder = MeanPooledBackboneEncoder(model, tokenizer,
                                        max_length=args.max_source_length,
                                        device=args.device)

    prototypes, task_grads, task_subspaces = [], [], []

    for i, name in enumerate(task_order):
        print(f"[law] task {i}: {name}")
        task = load_superni_task(args.data_dir, name, args.benchmark)
        prototypes.append(encoder.encode_one(task["instruction"]).float())
        train_loader, _, _ = make_loaders(
            task, tokenizer, batch_size=args.batch_size,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length, device=args.device,
        )
        batches = []
        it = iter(train_loader)
        for _ in range(args.max_batches):
            try:
                batches.append(next(it))
            except StopIteration:
                break

        # (a) occupied subspaces U_j
        for _, layer in iter_instil_layers(model):
            layer.start_collecting(2000)
        model.eval()
        with torch.no_grad():
            for b in batches:
                model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                      labels=b["labels"])
        subs = {}
        for n, layer in iter_instil_layers(model):
            C, cnt = layer.stop_collecting()
            m = layer.memory
            if cnt > 0:
                m.add_task_cov(C, cnt, i)
            subs[n] = m.basis_for(i)
        task_subspaces.append(subs)

        # (b) weight-space gradient G_j at the shared pretrained point
        def run_backward():
            model.train()
            for b in batches:
                loss = model(**b).loss / len(batches)
                loss.backward()
        task_grads.append(
            compute_task_weight_gradients(model, run_backward, layer_names,
                                          reset_adapters=True))

    sims, aligns, pairs = measure_gradient_alignment(
        prototypes, task_grads, task_subspaces, layer_names)
    summary = validate_law(sims, aligns)
    print("\n==== Instruction-Gradient Alignment Law ====")
    print(json.dumps(summary, indent=2))

    with open(os.path.join(args.output_dir, "law_points.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["similarity", "alignment", "task_j", "task_t"])
        for (s, a, (j, t)) in zip(sims, aligns, pairs):
            w.writerow([s, a, task_order[j], task_order[t]])
    with open(os.path.join(args.output_dir, "law_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 4))
        plt.scatter(sims, aligns, alpha=0.6)
        plt.axhline(0, color="grey", lw=0.8)
        plt.axvline(summary["rho0"], color="red", ls="--",
                    label=f"rho0={summary['rho0']:.3f}")
        plt.xlabel("instruction similarity  <p_t, p_j>")
        plt.ylabel("gradient alignment  <gL_j, gL_t>_{U_j}")
        plt.title(f"Instil Law: r={summary['pearson']:.2f}, n={summary['n']}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "law_scatter.png"), dpi=150)
        print(f"[law] scatter saved to {args.output_dir}/law_scatter.png")
    except Exception as e:  # pragma: no cover
        print(f"[law] (matplotlib unavailable, skipping plot: {e})")


if __name__ == "__main__":
    main()
