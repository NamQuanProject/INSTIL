"""Gradient hook (kept for API compatibility) -- paper §5.2.

In this implementation the instruction gate is baked **structurally** into the
frozen adapter basis ``A`` (see :mod:`instil.lora`): the null-space rows are
orthogonal to ``span(U_<t)`` and each occupied block ``j`` enters ``A`` only
with weight ``gamma_{t,j}``.  Consequently ``dW = scaling * B @ A`` satisfies the
non-forgetting floor (Prop. 1) for *any* ``B``, and no gradient surgery is
required during the optimiser loop.

``project_instil_gradients`` is therefore a no-op; it remains so callers written
against the "project the gradient after backward()" convention keep working.
"""

from __future__ import annotations


def project_instil_gradients(model) -> None:  # noqa: D401 - intentional no-op
    """No-op: the gate is enforced by construction of the frozen basis A."""
    return None
