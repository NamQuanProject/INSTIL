"""Continual-learning metrics: OP / Forgetting / BWT / FWT (paper §9).

The definitions match SAPT's ``score.py`` so numbers are directly comparable.

Let ``R`` be the lower-triangular result matrix where ``R[i][j]`` is the score
on task ``j`` *after* training through task ``i`` (``i >= j``).  With ``K`` tasks:

    OP  (Cl)  = mean_j R[K-1][j]                         final average performance
    Fgt       = mean_{j<K-1} ( max_{i<K-1} R[i][j] - R[K-1][j] )
    BWT       = mean_j ( R[K-1][j] - R[j][j] )
    FWT       = mean_j ( R[j][j] - b_j )

``b_j`` is the reference score of task ``j`` trained *individually* (isolated
single-task fine-tuning).  SAPT hard-codes a single pooled baseline constant
(50.94); we accept a per-task baseline vector and fall back to that constant so
either convention works.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

SAPT_SINGLE_TASK_BASELINE = 50.94


def continual_metrics(
    R: Sequence[Sequence[float]],
    single_task_baseline: Optional[Sequence[float]] = None,
) -> dict:
    """Compute {OP, Forgetting, BWT, FWT} from a result matrix.

    Parameters
    ----------
    R : lower-triangular ``K x K`` list-of-lists; ``R[i][j]`` valid for ``j<=i``.
        Upper-triangle entries (future tasks) are ignored.
    single_task_baseline : optional length-``K`` vector of isolated single-task
        scores ``b_j`` for FWT.  If ``None``, the SAPT pooled constant is used.
    """
    K = len(R)
    if K == 0:
        return {"OP": 0.0, "Forgetting": 0.0, "BWT": 0.0, "FWT": 0.0}

    last = R[K - 1]
    OP = sum(last[j] for j in range(K)) / K

    if K > 1:
        fgt = []
        for j in range(K - 1):
            history = [R[i][j] for i in range(j, K - 1)]  # i in [j, K-2]
            fgt.append(max(history) - last[j])
        Forgetting = sum(fgt) / len(fgt)
    else:
        Forgetting = 0.0

    diag = [R[i][i] for i in range(K)]
    BWT = sum(last[i] - diag[i] for i in range(K)) / K

    if single_task_baseline is None:
        FWT = sum(diag) / K - SAPT_SINGLE_TASK_BASELINE
    else:
        assert len(single_task_baseline) == K
        FWT = sum(diag[i] - single_task_baseline[i] for i in range(K)) / K

    return {"OP": OP, "Forgetting": Forgetting, "BWT": BWT, "FWT": FWT}


# Backwards-friendly alias mirroring SAPT's function name / key names.
def sapt_metrics(scores_array: Sequence[Sequence[float]],
                 baseline: float = SAPT_SINGLE_TASK_BASELINE) -> dict:
    """Same numbers as SAPT ``cal_continue_learning_metrics`` (keys Cl/Fgt/Fwt/Bwt)."""
    m = continual_metrics(scores_array, single_task_baseline=None)
    # respect a custom pooled baseline if provided
    K = len(scores_array)
    diag = [scores_array[i][i] for i in range(K)] if K else []
    fwt = (sum(diag) / K - baseline) if K else 0.0
    return {"Cl": m["OP"], "Fgt": m["Forgetting"], "Fwt": fwt, "Bwt": m["BWT"]}
