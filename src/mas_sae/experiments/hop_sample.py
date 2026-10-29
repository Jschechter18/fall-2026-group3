"""Seeded samples of Natural episodes, allocated to strata in proportion to their size.

Unlike ``adoption_experiment.sample_rows`` (equal allocation across hop x correctness
strata, an enriched design), these keep the pool's stratum mix so label rates can be
read as estimates for it.  ``proportional_hop_sample`` draws a fresh sample by hop
group; ``stratified_top_up`` grows an existing cohort so that cohort + additions is
proportional across any stratum key.  Selection never looks at A2, behavior or
lexical labels.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Callable

from mas_sae.data.musique import hop_group


def proportional_allocation(populations: dict[str, int], size: int) -> dict[str, int]:
    """Largest-remainder allocation of *size* across groups (ties broken by group name)."""
    total = sum(populations.values())
    if size <= 0 or size > total:
        raise ValueError(f"size must be in 1..{total}, got {size}")
    quotas = {g: size * n / total for g, n in populations.items()}
    counts = {g: int(q) for g, q in quotas.items()}
    by_remainder = sorted(populations, key=lambda g: (-(quotas[g] - counts[g]), g))
    for g in by_remainder[:size - sum(counts.values())]:
        counts[g] += 1
    return counts


def proportional_hop_sample(
    rows: list[dict[str, Any]], size: int, seed: int, exclude_ids: set[str] = frozenset(),
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Draw *size* rows, proportional by ``hop_group``, skipping ``exclude_ids``.

    Returns ``(selected, allocation)`` where allocation maps each hop group to
    its ``population`` (after exclusion) and ``selected`` count.  Rows are
    returned unmodified.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["question_id"] not in exclude_ids:
            groups[hop_group(row["question_id"])].append(row)
    counts = proportional_allocation({g: len(m) for g, m in groups.items()}, size)

    rng = random.Random(seed)
    selected = []
    for g in sorted(groups):
        members = sorted(groups[g], key=lambda r: r["question_id"])
        rng.shuffle(members)
        selected.extend(members[:counts[g]])
    rng.shuffle(selected)
    allocation = {g: {"population": len(groups[g]), "selected": counts[g]} for g in sorted(groups)}
    return selected, allocation


def top_up_allocation(populations: dict[str, int], have: dict[str, int], total: int) -> dict[str, int]:
    """Units to add per stratum so the cohort grows to *total* as close to proportional as it can.

    One unit at a time goes to the open stratum furthest below its quota
    ``total * population / sum(populations)`` (ties broken by stratum name).
    Strata the cohort already over-covers receive nothing.
    """
    pool = sum(populations.values())
    start = sum(have.values())
    if not start <= total <= pool:
        raise ValueError(f"total must be in {start}..{pool}, got {total}")
    quota = {s: total * n / pool for s, n in populations.items()}
    current = {s: have.get(s, 0) for s in populations}
    for _ in range(total - start):
        open_strata = [s for s in sorted(populations) if current[s] < populations[s]]
        current[max(open_strata, key=lambda s: quota[s] - current[s])] += 1
    return {s: current[s] - have.get(s, 0) for s in populations}


def stratified_top_up(
    rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], str], cohort_ids: set[str],
    additional: int, seed: int, exclude_ids: set[str] = frozenset(),
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Draw *additional* new rows so that cohort + new is proportional across ``key`` strata.

    *rows* is the full pool; rows in *exclude_ids* are neither counted nor drawn.
    Every cohort question must be in the pool.  Returns ``(new_rows, strata)``
    where strata maps each stratum to its population, cohort count, added count,
    cumulative total and proportional quota.  Rows are returned unmodified.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["question_id"] not in exclude_ids:
            groups[key(row)].append(row)
    pooled = {r["question_id"] for members in groups.values() for r in members}
    if not cohort_ids <= pooled:
        raise ValueError(f"{len(cohort_ids - pooled)} cohort question(s) are not in the pool")
    have = Counter(key(r) for members in groups.values() for r in members if r["question_id"] in cohort_ids)
    total = len(cohort_ids) + additional
    added = top_up_allocation({s: len(m) for s, m in groups.items()}, have, total)

    rng = random.Random(seed)
    selected = []
    for s in sorted(groups):
        available = sorted((r for r in groups[s] if r["question_id"] not in cohort_ids),
                           key=lambda r: r["question_id"])
        rng.shuffle(available)
        selected.extend(available[:added[s]])
    rng.shuffle(selected)
    pool = len(pooled)
    strata = {s: {"population": len(groups[s]), "cohort": have[s], "added": added[s],
                  "total": have[s] + added[s], "quota": round(total * len(groups[s]) / pool, 2)}
              for s in sorted(groups)}
    return selected, strata
