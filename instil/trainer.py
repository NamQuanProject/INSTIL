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

import math
from typing import Callable, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from .instil import Instil
from .logging_utils import get_logger, tqdm_bar


OptimizerFactory = Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer]
LossFn = Callable[[nn.Module, object], torch.Tensor]


class ContinualTrainer:
    def __init__(self, instil: Instil, loss_fn: LossFn,
                 optimizer_factory: Optional[OptimizerFactory] = None,
                 epochs: int = 1, grad_clip: Optional[float] = 1.0,
                 lr: float = 3e-4, device: Optional[str] = None,
                 forward_fn=None, log_every: int = 0, min_steps: int = 0):
        self.instil = instil
        self.loss_fn = loss_fn
        self.epochs = epochs
        # Floor on optimisation steps per task: small tasks (few hundred
        # examples) otherwise get only ~80-100 updates and stay undertrained.
        self.min_steps = min_steps
        self.grad_clip = grad_clip
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.forward_fn = forward_fn
        self.log_every = log_every
        self.optimizer_factory = optimizer_factory or (
            lambda params: torch.optim.AdamW(params, lr=lr)
        )

    def _trainable_params(self):
        return [p for p in self.instil.model.parameters() if p.requires_grad]

    def _make_train_step(self, train_loader, desc: str = "train") -> Callable[[], None]:
        logger = get_logger()

        def train_step():
            optimizer = self.optimizer_factory(self._trainable_params())
            try:
                steps_per_epoch = len(train_loader)
            except TypeError:
                steps_per_epoch = None
            # Raise the epoch count so every task gets at least ``min_steps``
            # updates -- small tasks otherwise converge nowhere near their loss.
            n_epochs = self.epochs
            if self.min_steps and steps_per_epoch:
                n_epochs = max(self.epochs,
                               math.ceil(self.min_steps / steps_per_epoch))
            total = n_epochs * steps_per_epoch if steps_per_epoch else None
            bar = tqdm_bar(total=total, desc=desc, leave=False)
            step, running = 0, 0.0
            for epoch in range(n_epochs):
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
                    lv = float(loss)
                    running += lv
                    bar.update(1)
                    bar.set_postfix(epoch=epoch + 1, loss=f"{lv:.4f}")
                    if self.log_every and step % self.log_every == 0:
                        logger.info(f"  [{desc}] step {step:5d}/{total or '?'} "
                                    f"epoch {epoch + 1}/{n_epochs} "
                                    f"loss {lv:.4f} (avg {running / step:.4f})")
            bar.close()
            if step:
                logger.info(f"  [{desc}] done: {step} steps, "
                            f"final loss {running / step:.4f} (avg)")
        return train_step

    def learn_task(self, instruction: str, train_loader, collect_loader=None,
                   desc: str = "train"):
        collect = collect_loader if collect_loader is not None else train_loader
        return self.instil.learn_task(
            instruction=instruction,
            train_step=self._make_train_step(train_loader, desc=desc),
            collect_batches=collect,
            forward_fn=self.forward_fn,
        )

    def run_stream(self, stream: Iterable[Tuple[str, object, object]]) -> List[torch.Tensor]:
        """Run a full stream of ``(instruction, train_loader, collect_loader)``."""
        logger = get_logger()
        protos = []
        for i, item in enumerate(stream):
            if len(item) == 3:
                instruction, train_loader, collect_loader = item
            else:
                instruction, train_loader = item
                collect_loader = None
            logger.info(f"=== task {i}: {instruction[:60]!r} ===")
            protos.append(self.learn_task(instruction, train_loader, collect_loader,
                                          desc=f"task{i}"))
        return protos
