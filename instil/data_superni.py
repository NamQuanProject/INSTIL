"""SuperNI / Long-Sequence data loading for Instil (matches SAPT's CL_Benchmark).

Each SuperNI task directory holds ``train.json`` / ``dev.json`` / ``test.json``.
The JSON has a ``Definition`` (list; ``Definition[0]`` is the natural-language
instruction ``s_t``) and ``Instances`` (list of ``{"input", "output"}``).

We build seq2seq examples where the model input is ``definition + input`` (so it
conditions on the instruction, per §3 "instructions available at train and
test") and the target is the output.  The *prototype* ``p_t`` is computed from
the **definition alone** -- that is the a-priori signal the gate consumes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_superni_task(root: str, task_name: str, benchmark: str = "SuperNI") -> dict:
    """Return ``{instruction, train, dev, test}`` for one task.

    ``train/dev/test`` are lists of ``(input, output)`` tuples.
    """
    tdir = os.path.join(root, benchmark, task_name)
    out = {"instruction": "", "train": [], "dev": [], "test": []}
    for split in ("train", "dev", "test"):
        path = os.path.join(tdir, f"{split}.json")
        if not os.path.exists(path):
            continue
        d = _read_json(path)
        if not out["instruction"]:
            definition = d.get("Definition", [""])
            out["instruction"] = definition[0] if isinstance(definition, list) else str(definition)
        pairs = []
        for inst in d.get("Instances", []):
            output = inst.get("output", "")
            if isinstance(output, list):  # some tasks store a list of valid answers
                output = output[0] if output else ""
            pairs.append((inst.get("input", ""), output))
        out[split] = pairs
    return out


@dataclass
class SeqExample:
    source: str
    target: str


class Seq2SeqTaskDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], instruction: str,
                 prepend_instruction: bool = True):
        self.instruction = instruction
        self.examples = []
        for inp, out in pairs:
            src = f"{instruction}\n\n{inp}" if prepend_instruction else inp
            self.examples.append(SeqExample(src, out))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


class T5Collator:
    """Tokenise a batch of :class:`SeqExample` into T5 model kwargs."""

    def __init__(self, tokenizer, max_source_length: int = 512,
                 max_target_length: int = 50, device: Optional[str] = None):
        self.tok = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.device = device

    def __call__(self, batch: List[SeqExample]) -> Dict[str, torch.Tensor]:
        sources = [e.source for e in batch]
        targets = [e.target for e in batch]
        model_inputs = self.tok(
            sources, max_length=self.max_source_length, truncation=True,
            padding=True, return_tensors="pt",
        )
        labels = self.tok(
            targets, max_length=self.max_target_length, truncation=True,
            padding=True, return_tensors="pt",
        )["input_ids"]
        # Ignore pad tokens in the loss (HF convention).
        labels[labels == self.tok.pad_token_id] = -100
        model_inputs["labels"] = labels
        if self.device is not None:
            model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
        return model_inputs


def make_loaders(task: dict, tokenizer, batch_size: int = 8,
                 eval_batch_size: int = 32, max_source_length: int = 512,
                 max_target_length: int = 50, device: Optional[str] = None,
                 max_train: Optional[int] = None, max_eval: Optional[int] = 200,
                 prepend_instruction: bool = True):
    """Build ``(train_loader, eval_loader, collator)`` for a task dict."""
    collate = T5Collator(tokenizer, max_source_length, max_target_length, device)
    train_pairs = task["train"][:max_train] if max_train else task["train"]
    eval_pairs = task["test"] or task["dev"]
    if max_eval:
        eval_pairs = eval_pairs[:max_eval]
    train_ds = Seq2SeqTaskDataset(train_pairs, task["instruction"], prepend_instruction)
    eval_ds = Seq2SeqTaskDataset(eval_pairs, task["instruction"], prepend_instruction)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate)
    eval_loader = DataLoader(eval_ds, batch_size=eval_batch_size, shuffle=False,
                             collate_fn=collate)
    return train_loader, eval_loader, collate
