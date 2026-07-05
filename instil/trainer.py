"""A thin continual-training loop wiring Instil to a dataloader (paper §7).

This is the ergonomic path: you provide a model (already injected), an encoder,
a per-task ``(instruction, train_loader, collect_loader)`` stream, an optimizer
factory, and a ``loss_fn(model, batch) -> loss``.  The trainer builds the
``train_step`` closure that runs the epochs *and* inserts the gated gradient
projection (Eq. 1) in the right place -- right after ``backward()`` and before
``optimizer.step()``.

You are free to ignore this and drive :meth:`Instil.learn_task` yourself (e.g.
from a HuggingFace ``Trainer``): just call ``instil.project_gradients()`` after
each ``loss.backward()``.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from .instil import Instil


OptimizerFactory = Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer]
LossFn = Callable[[nn.Module, object], torch.Tensor]


class ContinualTrainer:
    def __init__(self, instil: Instil, loss_fn: LossFn,
                 optimizer_factory: Optional[OptimizerFactory] = None,
                 epochs: int = 1, grad_clip: Optional[float] = 1.0,
                 lr: float = 3e-4, device: Optional[str] = None,
                 forward_fn=None, log_every: int = 0):
        self.instil = instil
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.forward_fn = forward_fn
        self.log_every = log_every
        self.optimizer_factory = optimizer_factory or (
            lambda params: torch.optim.AdamW(params, lr=lr)
        )

    def _trainable_params(self):
        return [p for p in self.instil.model.parameters() if p.requires_grad]

    def _make_train_step(self, train_loader) -> Callable[[], None]:
        def train_step():
            optimizer = self.optimizer_factory(self._trainable_params())
            step = 0
            for _ in range(self.epochs):
                for batch in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    loss = self.loss_fn(self.instil.model, batch)
                    loss.backward()
                    # ---- the Instil update (Eq. 1) -------------------------
                    self.instil.project_gradients()
                    if self.grad_clip:
                        torch.nn.utils.clip_grad_norm_(
                            self._trainable_params(), self.grad_clip)
                    optimizer.step()
                    step += 1
                    if self.log_every and step % self.log_every == 0:
                        print(f"    step {step:5d}  loss {float(loss):.4f}")
        return train_step

    def learn_task(self, instruction: str, train_loader, collect_loader=None):
        collect = collect_loader if collect_loader is not None else train_loader
        return self.instil.learn_task(
            instruction=instruction,
            train_step=self._make_train_step(train_loader),
            collect_batches=collect,
            forward_fn=self.forward_fn,
        )

    def run_stream(self, stream: Iterable[Tuple[str, object, object]]) -> List[torch.Tensor]:
        """Run a full stream of ``(instruction, train_loader, collect_loader)``."""
        protos = []
        for i, item in enumerate(stream):
            if len(item) == 3:
                instruction, train_loader, collect_loader = item
            else:
                instruction, train_loader = item
                collect_loader = None
            print(f"[instil] === task {i}: {instruction[:60]!r} ===")
            protos.append(self.learn_task(instruction, train_loader, collect_loader))
        return protos
