#!/usr/bin/env python
"""Run Instil on the SuperNI CIT benchmark with a T5 backbone.

This is the real-data counterpart of SAPT's ``gen_script_superni_t5.py`` +
``run_t5.py`` pipeline, but end-to-end in one process: it streams the task
order, learns each task with the instruction-gated update (Eq. 1), and after
every task evaluates *all* seen tasks (Bank routing) to fill the lower-triangular
result matrix R, from which OP / Forgetting / BWT / FWT are computed (§9).

Example
-------
    python scripts/run_instil_t5.py \
        --model_name_or_path t5-large \
        --data_dir SAPT/CL_Benchmark \
        --task_order task1572_samsum_summary,task363_sst2_polarity_classification,... \
        --mode bank --lora_r 8 --lora_alpha 16 --epochs 5 \
        --output_dir logs_and_outputs/instil_superni

Use ``--max_train`` / ``--max_eval`` / a tiny model (``t5-small``) for a quick
smoke test on CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from instil import (InstilConfig, Instil, inject_instil_lora,
                    MeanPooledBackboneEncoder, continual_metrics)
from instil.trainer import ContinualTrainer
from instil.data_superni import load_superni_task, make_loaders
from instil.textscore import corpus_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--data_dir", default="SAPT/CL_Benchmark")
    p.add_argument("--benchmark", default="SuperNI")
    p.add_argument("--task_order", required=True, help="comma-separated task names")
    p.add_argument("--mode", default="bank", choices=["bank", "merge"])
    p.add_argument("--metric", default="rougeL", choices=["rougeL", "exact_match"])
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--target_modules", default="q,v")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--max_source_length", type=int, default=512)
    p.add_argument("--max_target_length", type=int, default=50)
    p.add_argument("--gate_slope_a", type=float, default=10.0)
    p.add_argument("--rho0", type=float, default=None,
                   help="fix the gate zero-crossing; omit to use 0.0 / Law fit")
    p.add_argument("--no_warm_start", action="store_true")
    p.add_argument("--max_train", type=int, default=None)
    p.add_argument("--max_eval", type=int, default=200)
    p.add_argument("--output_dir", default="logs_and_outputs/instil_superni")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def generate(model, tokenizer, eval_loader, device, max_new_tokens):
    model.eval()
    preds = []
    for batch in eval_loader:
        gen = model.generate(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            max_new_tokens=max_new_tokens, num_beams=1,
        )
        preds.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return preds


def eval_task(instil, model, tokenizer, task, eval_loader, args):
    """Evaluate one seen task via Bank routing (or the shared adapter in merge)."""
    references = [e.target for e in eval_loader.dataset]
    if instil.cfg.mode == "bank":
        with instil.answer(task["instruction"]):
            preds = generate(model, tokenizer, eval_loader, args.device,
                             args.max_target_length)
    else:
        preds = generate(model, tokenizer, eval_loader, args.device,
                         args.max_target_length)
    return corpus_score(preds, references, args.metric)


def main():
    args = parse_args()
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    os.makedirs(args.output_dir, exist_ok=True)
    task_order = args.task_order.split(",")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    cfg = InstilConfig(
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=args.target_modules.split(","),
        mode=args.mode, gate_slope_a=args.gate_slope_a, rho0=args.rho0,
        warm_start=not args.no_warm_start,
    )
    wrapped = inject_instil_lora(model, cfg, verbose=True)
    print(f"[instil] wrapped {len(wrapped)} linear layers")

    encoder = MeanPooledBackboneEncoder(model, tokenizer,
                                        max_length=args.max_source_length,
                                        device=args.device)
    instil = Instil(model, encoder, cfg)

    def loss_fn(m, batch):
        return m(**batch).loss

    def forward_fn(m, batch):  # used for the activation-collection pass
        m(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
          labels=batch["labels"])

    trainer = ContinualTrainer(instil, loss_fn, epochs=args.epochs, lr=args.lr,
                               device=args.device, forward_fn=forward_fn,
                               log_every=10)

    # Cache task data + eval loaders as we go.
    tasks, eval_loaders = [], []
    R = []  # lower-triangular result matrix

    for i, name in enumerate(task_order):
        print(f"\n===== Task {i}: {name} =====")
        task = load_superni_task(args.data_dir, name, args.benchmark)
        train_loader, eval_loader, _ = make_loaders(
            task, tokenizer, batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
            device=args.device, max_train=args.max_train, max_eval=args.max_eval,
        )
        tasks.append(task)
        eval_loaders.append(eval_loader)

        trainer.learn_task(task["instruction"], train_loader, collect_loader=train_loader)

        # Evaluate all tasks seen so far -> row i of R.
        row = []
        for j in range(i + 1):
            score = eval_task(instil, model, tokenizer, tasks[j], eval_loaders[j], args)
            row.append(score)
            print(f"    R[{i}][{j}] ({task_order[j]}): {score:.2f}")
        row.extend([0.0] * (len(task_order) - i - 1))
        R.append(row)

    metrics = continual_metrics(R)
    print("\n==== Instil continual-learning metrics ====")
    print(json.dumps(metrics, indent=2))

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"task_order": task_order, "R": R, "metrics": metrics}, f, indent=2)
    instil.save(os.path.join(args.output_dir, "instil_state.pt"))
    with open(os.path.join(args.output_dir, "task_order.txt"), "w") as f:
        f.write(",".join(task_order))
    print(f"[instil] saved to {args.output_dir}")


if __name__ == "__main__":
    main()
