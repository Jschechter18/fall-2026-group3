"""Paired analysis of two collections of the same questions under one change (#174, #156).

The rules below are fixed before a collection runs, so missing data cannot be handled after seeing results.
Each assigned question gets one outcome per arm:

* a Claude label (adoption, retention, agreement, out) for a collected episode the validator labeled;
* ``out`` for a question the collection excluded by a protocol rule (``exclusions.jsonl``: no usable blind
  Critic answer or recommendation, or no controlled target). That is model behavior, as in PR #153;
* ``not_collected`` when the run has neither an episode nor a protocol exclusion (it never finished the
  question), and ``label_failed`` when the validator gave no label after its retries. These are technical
  failures, never counted as Out.

The binary outcome is A+R (adoption or retention). Three analyses:

* ``primary``: questions where neither arm has a technical failure;
* ``intention_to_treat``: every assigned question, a technical failure counted as not A+R;
* ``treated_only``: primary questions whose intended change was applied (e.g. competitors that passed checks).

Each gives both arms' A+R share with a 95% Wilson interval, the paired 2x2 table, the discordant pairs, the
change in percentage points with Newcombe's hybrid score interval for paired proportions, and the exact
two-sided McNemar p-value.
"""
import math
from collections import Counter

from mas_sae.data.musique import hop_type
from mas_sae.evaluation.claude_agreement import wilson
from mas_sae.evaluation.claude_validator import VALID_LABELS

PROTOCOL_EXCLUSION_STAGES = ("critic_blind", "critic_recommendation", "controlled_target")
TECHNICAL = ("not_collected", "label_failed")
OUTCOMES = (*VALID_LABELS, *TECHNICAL)
BINARY = ("adoption", "retention")


def by_question(rows: list[dict], source: str) -> dict[str, dict]:
    """Index rows by ``question_id``, refusing repeats."""
    indexed = {}
    for row in rows:
        if row["question_id"] in indexed:
            raise ValueError(f"{source}: repeated question_id {row['question_id']!r}")
        indexed[row["question_id"]] = row
    return indexed


def arm_outcomes(assigned: list[str], interactions: list[dict], exclusions: list[dict],
                 labels: list[dict]) -> dict[str, str]:
    """The outcome of every assigned question in one arm, by the rules in the module docstring."""
    episodes = by_question(interactions, "interactions")
    excluded = by_question(exclusions, "exclusions")
    labeled = by_question(labels, "labels")
    if len(set(assigned)) != len(assigned):
        raise ValueError("the assigned question list repeats a question")
    for name, rows in (("interactions", episodes), ("exclusions", excluded), ("labels", labeled)):
        extra = sorted(rows.keys() - set(assigned))
        if extra:
            raise ValueError(f"{name}: {len(extra)} question(s) were not assigned, e.g. {extra[:3]}")
    both = sorted(episodes.keys() & excluded.keys())
    if both:
        raise ValueError(f"{len(both)} question(s) are both collected and excluded, e.g. {both[:3]}")
    stages = sorted({row["stage"] for row in excluded.values()} - set(PROTOCOL_EXCLUSION_STAGES))
    if stages:
        raise ValueError(f"unknown collection exclusion stages {stages}; classify them before analysing")
    unmatched = sorted(labeled.keys() - episodes.keys())
    if unmatched:
        raise ValueError(f"{len(unmatched)} label(s) have no collected episode, e.g. {unmatched[:3]}")
    unlabeled = sorted(episodes.keys() - labeled.keys())
    if unlabeled:
        raise ValueError(f"{len(unlabeled)} episode(s) have no label row (incomplete label run), e.g. {unlabeled[:3]}")
    outcomes = {}
    for qid in assigned:
        if qid in excluded:
            outcomes[qid] = "out"
        elif qid not in episodes:
            outcomes[qid] = "not_collected"
        else:
            label = labeled[qid]
            if label["episode_id"] != episodes[qid]["episode_id"]:
                raise ValueError(f"{qid!r}: the label is for episode {label['episode_id']!r}, "
                                 f"not {episodes[qid]['episode_id']!r}")
            if label["behavior_label"] is not None and label["behavior_label"] not in VALID_LABELS:
                raise ValueError(f"{qid!r}: unknown label {label['behavior_label']!r}")
            outcomes[qid] = label["behavior_label"] or "label_failed"
    return outcomes


def exact_mcnemar(gained: int, lost: int) -> float:
    """Two-sided exact McNemar p-value: a binomial test of the discordant pairs at 1/2."""
    n = gained + lost
    if not n:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(gained, lost) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_difference_ci(both: int, treated_only: int, baseline_only: int, neither: int,
                         z: float = 1.96) -> tuple[float, float] | None:
    """Interval for the treated minus baseline share on paired binary data.

    Newcombe's hybrid score interval (1998, method 10): Wilson intervals for each share, combined with
    the correlation phi, whose numerator AD - BC becomes max(AD - BC - n/2, 0) when positive (Altman et al.,
    Statistics with Confidence, 2000).
    """
    n = both + treated_only + baseline_only + neither
    if not n:
        return None
    p1, p2 = (both + treated_only) / n, (both + baseline_only) / n
    l1, u1 = wilson(both + treated_only, n, z)
    l2, u2 = wilson(both + baseline_only, n, z)
    margins = (both + treated_only) * (baseline_only + neither) * (both + baseline_only) * (treated_only + neither)
    cross = both * neither - treated_only * baseline_only
    numerator = cross if cross < 0 else max(cross - n / 2, 0)
    phi = numerator / math.sqrt(margins) if margins else 0.0
    low = p1 - p2 - math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    high = p1 - p2 + math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return max(-1.0, low), min(1.0, high)


def paired_test(baseline: dict[str, bool], treated: dict[str, bool], ids: list[str]) -> dict:
    """Both arms' A+R shares, the paired table, the change with its interval and the exact McNemar p."""
    cells = Counter((treated[q], baseline[q]) for q in ids)
    both, treated_only = cells[True, True], cells[True, False]
    baseline_only, neither = cells[False, True], cells[False, False]
    n = len(ids)
    arms = {name: {"binary": k, "share": k / n if n else None, "ci95": wilson(k, n)}
            for name, k in (("baseline", both + baseline_only), ("treated", both + treated_only))}
    ci = paired_difference_ci(both, treated_only, baseline_only, neither)
    return {"n": n, **arms,
            "table": {"both": both, "treated_only": treated_only, "baseline_only": baseline_only, "neither": neither},
            "discordant": {"gained": treated_only, "lost": baseline_only},
            "change_pp": 100 * (treated_only - baseline_only) / n if n else None,
            "change_ci95_pp": [100 * ci[0], 100 * ci[1]] if ci else None,
            "exact_mcnemar_p": exact_mcnemar(treated_only, baseline_only)}


def analyses(baseline: dict[str, str], treated: dict[str, str], assigned: list[str],
             treated_ids: list[str] | None = None) -> dict:
    """The primary, intention-to-treat and (with ``treated_ids``) treated-only analyses."""
    if set(baseline) != set(assigned) or set(treated) != set(assigned):
        raise ValueError("both arms need an outcome for exactly the assigned questions")
    if treated_ids is not None:
        if len(set(treated_ids)) != len(treated_ids):
            raise ValueError("the treated question list repeats a question")
        if set(treated_ids) - set(assigned):
            raise ValueError("treated questions must be assigned questions")
    hit = {name: {q: arm[q] in BINARY for q in assigned} for name, arm in (("baseline", baseline), ("treated", treated))}
    primary = [q for q in assigned if baseline[q] not in TECHNICAL and treated[q] not in TECHNICAL]
    result = {"primary": paired_test(hit["baseline"], hit["treated"], primary),
              "intention_to_treat": paired_test(hit["baseline"], hit["treated"], list(assigned))}
    if treated_ids is not None:
        chosen = set(treated_ids)
        result["treated_only"] = paired_test(hit["baseline"], hit["treated"], [q for q in primary if q in chosen])
    result["technical_failures"] = {name: dict(Counter(arm[q] for q in assigned if arm[q] in TECHNICAL))
                                    for name, arm in (("baseline", baseline), ("treated", treated))}
    return result


def hop_of(question_id: str) -> str:
    """Hop type such as ``3hop1``, ignoring a ``source:`` namespace."""
    return hop_type(question_id.split(":", 1)[-1])


def correctness_cell(episode: dict | None) -> str:
    """Solver A1 and natural Critic correctness, e.g. ``A1 wrong / Critic right``; no episode for exclusions."""
    if episode is None:
        return "no episode"
    side = lambda flag: "right" if flag is True else "wrong" if flag is False else "unknown"
    return f"A1 {side(episode['solver_attempt_1_correct'])} / Critic {side(episode.get('critic_feedback_correct'))}"


def breakdown(outcomes: dict[str, str], episodes: dict[str, dict], key) -> dict[str, dict[str, int]]:
    """Outcome counts within each group given by ``key(question_id, episode)``; descriptive only."""
    groups: dict[str, Counter] = {}
    for qid, outcome in outcomes.items():
        groups.setdefault(key(qid, episodes.get(qid)), Counter())[outcome] += 1
    return {group: {o: counts[o] for o in OUTCOMES if counts[o]} for group, counts in sorted(groups.items())}
