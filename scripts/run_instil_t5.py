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
        --data_dir data \
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
from instil.logging_utils import setup_file_logger, tqdm_iter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--benchmark", default="SuperNI")
    p.add_argument("--task_order", required=True, help="comma-separated task names")
    p.add_argument("--mode", default="bank", choices=["bank", "merge"])
    p.add_argument("--metric", default="rougeL", choices=["rougeL", "exact_match"])
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--target_modules", default="q,v")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--min_steps", type=int, default=300,
                   help="floor on optimisation steps per task (fixes small/undertrained tasks)")
    p.add_argument("--num_beams", type=int, default=4,
                   help="beam search width at eval (lifts ROUGE on generative tasks)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--max_source_length", type=int, default=512)
    p.add_argument("--max_target_length", type=int, default=128,
                   help="raise for summarization tasks whose references exceed 50 tokens")
    p.add_argument("--gate_slope_a", type=float, default=10.0)
    p.add_argument("--rho0", type=float, default=None,
                   help="fix the gate zero-crossing; omit to use 0.0 / Law fit")
    p.add_argument("--no_warm_start", action="store_true")
    p.add_argument("--max_train", type=int, default=None)
    p.add_argument("--max_eval", type=int, default=200)
    p.add_argument("--output_dir", default="logs_and_outputs/instil_superni")
    p.add_argument("--log_dir", default="logs")
    p.add_argument("--run_name", default=None, help="log/run name (default: from output_dir)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def generate(model, tokenizer, eval_loader, device, max_new_tokens, num_beams=4, desc="eval"):
    model.eval()
    preds = []
    for batch in tqdm_iter(eval_loader, desc=desc, leave=False):
        gen = model.generate(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            max_new_tokens=max_new_tokens, num_beams=num_beams,
        )
        preds.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return preds


def eval_task(instil, model, tokenizer, task, eval_loader, args, desc="eval"):
    """Evaluate one seen task via Bank routing (or the shared adapter in merge)."""
    references = [e.target for e in eval_loader.dataset]
    if instil.cfg.mode == "bank":
        with instil.answer(task["instruction"]):
            preds = generate(model, tokenizer, eval_loader, args.device,
                             args.max_target_length, args.num_beams, desc=desc)
    else:
        preds = generate(model, tokenizer, eval_loader, args.device,
                         args.max_target_length, args.num_beams, desc=desc)
    return corpus_score(preds, references, args.metric)


def main():
    args = parse_args()
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    os.makedirs(args.output_dir, exist_ok=True)
    run_name = args.run_name or (os.path.basename(args.output_dir.rstrip("/")) or "instil")
    logger, logfile = setup_file_logger(run_name=run_name, log_dir=args.log_dir)
    logger.info(f"run={run_name} | args: {vars(args)}")
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
    wrapped = inject_instil_lora(model, cfg, verbose=False)
    logger.info(f"wrapped {len(wrapped)} linear layers ({cfg.target_modules})")

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
                               log_every=10, min_steps=args.min_steps)

    # Cache task data + eval loaders as we go.
    tasks, eval_loaders = [], []
    R = []  # lower-triangular result matrix

    task_bar = tqdm_iter(range(len(task_order)), desc="tasks", total=len(task_order),
                         leave=True)
    for i in task_bar:
        name = task_order[i]
        logger.info(f"===== Task {i}/{len(task_order)-1}: {name} =====")
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

        trainer.learn_task(task["instruction"], train_loader,
                           collect_loader=train_loader, desc=f"task{i}:{name[:18]}")

        # Evaluate all tasks seen so far -> row i of R.
        row = []
        for j in range(i + 1):
            score = eval_task(instil, model, tokenizer, tasks[j], eval_loaders[j],
                              args, desc=f"eval t{j}")
            row.append(score)
            logger.info(f"    R[{i}][{j}] {task_order[j]:<45} {args.metric}={score:.2f}")
        row.extend([0.0] * (len(task_order) - i - 1))
        R.append(row)
        seen = continual_metrics(R)
        logger.info(f"    running metrics after task {i}: "
                    f"OP={seen['OP']:.2f} BWT={seen['BWT']:.2f} "
                    f"FWT={seen['FWT']:.2f} Fgt={seen['Forgetting']:.2f}")

    metrics = continual_metrics(R)
    logger.info("==== Instil continual-learning metrics ====")
    logger.info(json.dumps(metrics, indent=2))

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"task_order": task_order, "R": R, "metrics": metrics}, f, indent=2)
    instil.save(os.path.join(args.output_dir, "instil_state.pt"))
    with open(os.path.join(args.output_dir, "task_order.txt"), "w") as f:
        f.write(",".join(task_order))
    logger.info(f"saved outputs to {args.output_dir} | full log at {logfile}")


if __name__ == "__main__":
    main()
