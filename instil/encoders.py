"""Frozen instruction encoders E: instruction text -> unit prototype p_t.

Paper §3 / §8.  The prototype lives in a *shared, frozen, drift-free* semantic
manifold and is available *before* any gradient is seen -- this is what makes
the instruction a legitimate a-priori predictor of weight-space conflict.

Two encoders are provided:

* ``MeanPooledBackboneEncoder`` -- the paper default: mean-pooled last-hidden
  state of the instruction pushed through the *frozen backbone* (zero extra
  models).
* ``HashingEncoder`` -- a dependency-light bag-of-words / feature-hashing
  encoder.  Deterministic and CPU-only; handy for tests, the synthetic demo,
  and ablating "encoder choice" (§9) without loading a large model.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import List, Optional, Sequence

import torch


class InstructionEncoder:
    """Interface: map a list of instruction strings to (N, e) unit prototypes."""

    dim: int

    @torch.no_grad()
    def encode(self, instructions: Sequence[str]) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    @torch.no_grad()
    def encode_one(self, instruction: str) -> torch.Tensor:
        return self.encode([instruction])[0]


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class MeanPooledBackboneEncoder(InstructionEncoder):
    """Mean-pooled last-hidden-state of the instruction through a frozen model.

    Works with either an encoder-decoder (T5: uses ``.encoder``) or a decoder
    backbone (LLaMA).  The backbone is used purely as a *frozen* feature
    extractor; no parameters are trained here (paper §8, "zero extra models").
    """

    def __init__(self, model, tokenizer, max_length: int = 512, device=None,
                 l2_normalize: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device or next(model.parameters()).device
        self.l2_normalize = l2_normalize
        # Prefer the encoder tower for seq2seq models.
        self._encoder_module = getattr(model, "encoder", model)
        self.dim = getattr(model.config, "d_model",
                           getattr(model.config, "hidden_size", None))

    @torch.no_grad()
    def encode(self, instructions: Sequence[str]) -> torch.Tensor:
        from contextlib import nullcontext
        try:
            from .lora import bypass_adapters, iter_instil_layers
            has_adapters = any(True for _ in iter_instil_layers(self.model))
            ctx = bypass_adapters(self.model) if has_adapters else nullcontext()
        except Exception:
            ctx = nullcontext()

        was_training = self.model.training
        self.model.eval()
        batch = self.tokenizer(
            list(instructions),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        with ctx:
            out = self._encoder_module(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
        hidden = out.last_hidden_state  # (N, L, e)
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # (N, L, 1)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts  # mean pooling over real tokens
        if was_training:
            self.model.train()
        pooled = pooled.float().cpu()
        return _l2_normalize(pooled) if self.l2_normalize else pooled


class HashingEncoder(InstructionEncoder):
    """Deterministic feature-hashing bag-of-words encoder (no dependencies).

    Not meant to compete with a real LM encoder -- it exists so the whole
    pipeline (gate, subspace, routing, Law experiment) can be exercised on CPU
    with zero downloads.  Two instructions that share vocabulary get a high
    cosine similarity, which is exactly the signal the gate consumes.
    """

    _token_re = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = 256, l2_normalize: bool = True):
        self.dim = dim
        self.l2_normalize = l2_normalize

    def _hash(self, token: str) -> int:
        h = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(h[:4], "little") % self.dim

    @torch.no_grad()
    def encode(self, instructions: Sequence[str]) -> torch.Tensor:
        vecs = torch.zeros(len(instructions), self.dim, dtype=torch.float32)
        for i, text in enumerate(instructions):
            for tok in self._token_re.findall(text.lower()):
                idx = self._hash(tok)
                # signed hashing reduces collision bias
                sign = 1.0 if (self._hash(tok + "#sign") % 2 == 0) else -1.0
                vecs[i, idx] += sign * (1.0 / math.sqrt(1 + idx % 7))
        return _l2_normalize(vecs) if self.l2_normalize else vecs
