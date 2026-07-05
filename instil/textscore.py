"""Lightweight text scorers (ROUGE-L F1, exact match) -- no external deps.

SAPT reports ``rougeL`` for SuperNI and ``exact_match`` for the Long-Sequence
benchmark.  We reimplement both with a whitespace tokeniser so evaluation runs
without the heavy ``rouge`` package; numbers track SAPT closely for these short
generations.  Scores are returned on a 0-100 scale to match ``score.py``.
"""

from __future__ import annotations

from typing import List, Sequence


def _tokens(s: str) -> List[str]:
    return s.strip().lower().split()


def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[m]


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 (0-100)."""
    p, r = _tokens(prediction), _tokens(reference)
    if not p and not r:
        return 100.0
    if not p or not r:
        return 0.0
    lcs = _lcs(p, r)
    if lcs == 0:
        return 0.0
    prec = lcs / len(p)
    rec = lcs / len(r)
    return 100.0 * (2 * prec * rec) / (prec + rec)


def exact_match(prediction: str, reference: str) -> float:
    return 100.0 if prediction.strip().lower() == reference.strip().lower() else 0.0


def corpus_score(predictions: Sequence[str], references: Sequence[str],
                 metric: str = "rougeL") -> float:
    assert len(predictions) == len(references)
    if not predictions:
        return 0.0
    fn = rouge_l if metric == "rougeL" else exact_match
    return sum(fn(p, r) for p, r in zip(predictions, references)) / len(predictions)
