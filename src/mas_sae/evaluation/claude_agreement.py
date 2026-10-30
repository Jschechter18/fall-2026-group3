"""Agreement between Claude's four-class labels and a reference labeling (e.g. PR #153).

The reference is a comparison point, not ground truth: disagreements are listed for human
adjudication.  Claude rows that ended in a technical error are counted in an ``error``
column, never silently dropped.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

from mas_sae.evaluation.claude_validator import VALID_LABELS

ERROR = "error"


def confusion(pairs: Iterable[tuple[str, str | None]]) -> dict[str, dict[str, int]]:
    """``{reference: {predicted: n}}`` over the four labels; a missing prediction is ``error``."""
    matrix = {ref: {pred: 0 for pred in (*VALID_LABELS, ERROR)} for ref in VALID_LABELS}
    for reference, predicted in pairs:
        if reference not in matrix:
            raise ValueError(f"unknown reference label {reference!r}")
        matrix[reference][predicted if predicted in VALID_LABELS else ERROR] += 1
    return matrix


def per_class(matrix: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int | None]]:
    """Precision, recall and F1 per label (errors count against recall, not precision)."""
    out = {}
    for label in VALID_LABELS:
        tp = matrix[label][label]
        predicted = sum(matrix[ref][label] for ref in VALID_LABELS)
        support = sum(matrix[label].values())
        precision = tp / predicted if predicted else None
        recall = tp / support if support else None
        if precision is None or recall is None:
            f1 = None
        elif precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        out[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support,
                      "predicted": predicted}
    return out


def cohen_kappa(matrix: dict[str, dict[str, int]]) -> float | None:
    """Cohen's kappa over rows with a four-class prediction (technical errors excluded)."""
    n = sum(matrix[r][p] for r in VALID_LABELS for p in VALID_LABELS)
    if not n:
        return None
    observed = sum(matrix[label][label] for label in VALID_LABELS) / n
    expected = sum(sum(matrix[label][p] for p in VALID_LABELS) * sum(matrix[r][label] for r in VALID_LABELS)
                   for label in VALID_LABELS) / n ** 2
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def weighted_accuracy(matrix: dict[str, dict[str, int]], population: dict[str, int]) -> float | None:
    """Accuracy re-weighted to the population's label mix (for class-stratified samples)."""
    total = sum(population.get(label, 0) for label in VALID_LABELS)
    recalls = per_class(matrix)
    if not total or any(recalls[label]["recall"] is None for label in VALID_LABELS if population.get(label)):
        return None
    return sum(population.get(label, 0) / total * (recalls[label]["recall"] or 0.0) for label in VALID_LABELS)


def binary_side_agreement(pairs: Iterable[tuple[str, str | None]]) -> dict[str, int | float | None]:
    """Among episodes both sides call adoption/retention: how often they pick the same side."""
    both = [(r, p) for r, p in pairs if r in ("adoption", "retention") and p in ("adoption", "retention")]
    same = sum(r == p for r, p in both)
    return {"n": len(both), "same": same, "rate": same / len(both) if both else None}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for k successes out of n."""
    if not n:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def label_distribution(labels: Iterable[str | None]) -> dict:
    """Counts, shares and 95% intervals per label; a missing label counts as ``error``."""
    labels = list(labels)
    n = len(labels)
    counts = Counter(label if label in VALID_LABELS else ERROR for label in labels)
    out = {"n": n, **{label: {"count": counts[label], "share": counts[label] / n if n else None,
                              "ci95": wilson(counts[label], n)} for label in (*VALID_LABELS, ERROR)}}
    binary = counts["adoption"] + counts["retention"]
    out["retention_share_of_binary"] = {"retention": counts["retention"], "binary": binary,
                                        "share": counts["retention"] / binary if binary else None,
                                        "ci95": wilson(counts["retention"], binary)}
    return out
