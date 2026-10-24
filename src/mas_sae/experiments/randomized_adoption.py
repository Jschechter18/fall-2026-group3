"""Seeded, question-level order randomization for labeled candidate evaluation."""
from collections import Counter, defaultdict
import random

from mas_sae.data.musique import hop_group
from mas_sae.evaluation.scoring import answer_matches
from mas_sae.experiments.anonymized_adoption import ORDERS, run_anonymization
from mas_sae.experiments.controlled_adoption import summarize_controlled

PROTOCOL = "adoption_randomized_labeled_order_v1"


def assign_orders(rows, examples, seed):
    """Block-randomize by hop and A1 correctness, before observing any A2.

    Same question always gets the same order across critic arms. Counts differ
    by at most one within blocks and in the full question sample. Assignment
    is stable under reordering inputs, using only A1 and pinned dataset fields.
    """
    if type(seed) is not int:
        raise ValueError("Order seed must be an integer")
    if not rows or len({r["question_id"] for r in rows}) != len(rows):
        raise ValueError("Question sample must be nonempty and unique")
    groups = defaultdict(list)
    for row in rows:
        example = examples[row["question_id"]]
        correct = answer_matches(row["solver_attempt_1"], example["answer"], example.get("answer_aliases", []))
        groups[(hop_group(row["question_id"]), correct)].append(row["question_id"])
    rng = random.Random(seed)
    extra = rng.randrange(2)
    assignment = {}
    for key in sorted(groups):
        ids = sorted(groups[key])
        rng.shuffle(ids)
        first_count = len(ids) // 2
        if len(ids) % 2:
            first_count += extra
            extra = 1 - extra
        values = [ORDERS[0]] * first_count + [ORDERS[1]] * (len(ids) - first_count)
        rng.shuffle(values)
        assignment.update(zip(ids, values))
    return assignment


def order_only_baseline(rows):
    """Fixed first-candidate predictor, with no fitting or held-out-data claims."""
    tp = tn = fp = fn = 0
    for row in rows:
        prediction = row["candidate_order"] == "critic_first"
        actual = row["adopted_target"]
        tp += prediction and actual
        fp += prediction and not actual
        tn += not prediction and not actual
        fn += not prediction and actual
    positive, negative = tp + fn, tn + fp
    return {"n": len(rows), "prediction": "adopt if critic_first; otherwise do not adopt",
            "accuracy": (tp + tn) / len(rows) if rows else None,
            "balanced_accuracy": ((tp / positive) + (tn / negative)) / 2 if positive and negative else None,
            "confusion_matrix": {"true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn},
            "always_adopt_accuracy": positive / len(rows) if rows else None,
            "never_adopt_accuracy": negative / len(rows) if rows else None}


def summarize_randomized_order(records, assignments, seed):
    completed = [r for r in records if r["status"] == "completed"]
    strict = [r for r in completed if r["adopted_target"] != r["retained_a1"]]
    return {"protocol": PROTOCOL, "order_seed": seed, "assignment_unit": "question",
            "assigned_question_counts": dict(Counter(assignments.values())),
            "completed_trial_order_counts": dict(Counter(r["candidate_order"] for r in completed)),
            "completed": len(completed), "skipped": len(records) - len(completed),
            "distinct_completed_questions": len({r["question_id"] for r in completed}),
            "adopted_target": sum(r["adopted_target"] for r in completed),
            "retained_a1": sum(r["retained_a1"] for r in completed),
            "other_outputs": sum(not r["adopted_target"] and not r["retained_a1"] for r in completed),
            "arms": summarize_controlled(records)["arms"],
            "by_order": {order: summarize_controlled([r for r in records if r["candidate_order"] == order])["arms"] for order in ORDERS},
            "order_only_baseline": {"all_completed_adoption_vs_nonadoption": order_only_baseline(completed),
                                    "strict_adoption_vs_retention": order_only_baseline(strict)},
            "note": "Seeded assignment precedes A2 and is balanced over questions, not forced solver outcomes. All arms of a question share its assigned order. Target failures and repeated arms can unbalance completed trial counts. The order-only baseline is a fixed first-candidate heuristic, not a trained or held-out probe. Keep all variants of a question in the same research partition."}


def run_randomized_order(rows, examples, sources, solver, output_dir, seed=42, provenance=None):
    assignments = assign_orders(rows, examples, seed)
    return run_anonymization(rows, examples, sources, solver, output_dir, provenance=provenance,
                             assigned_orders=assignments, order_seed=seed)
