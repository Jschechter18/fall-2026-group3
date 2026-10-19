"""Cross source-label visibility with answer order on frozen controlled targets."""
import csv
import json
import logging
from pathlib import Path

from mas_sae.agents.solver import Solver
from mas_sae.evaluation.behavior_labels import classify_candidate
from mas_sae.evaluation.scoring import answer_matches, score_revision
from mas_sae.experiments.controlled_adoption import ARMS, summarize_controlled
from mas_sae.experiments.artifacts import sha256_text
from mas_sae.experiments.records import append_jsonl
from mas_sae.experiments.verification_adoption import RESULT_COLUMNS, result_csv_row, validate_sources

logger = logging.getLogger(__name__)
PROTOCOL = "adoption_anonymization_v1"
CONDITIONS = ("labeled", "anonymous")
ORDERS = ("a1_first", "critic_first")
CELLS = tuple((condition, order) for condition in CONDITIONS for order in ORDERS)
CANDIDATE_TEMPLATE = (
    "Answer the question using only the paragraphs provided.\n"
    "Consider the two candidate answers below. You may use either candidate, "
    "provide another answer, or state that the evidence is insufficient.\n"
    "Respond with the final answer text only, no explanation.\n\n"
    "{paragraphs}\n\nQuestion: {question}\n"
    "Candidate A{label_a}: {answer_a}\n"
    "Candidate B{label_b}: {answer_b}\n"
    "Final answer:"
)
LABELS = {"a1": " (your previous answer)", "critic": " (the reviewer's recommendation)"}


class CandidateSolver(Solver):
    def build_candidate_prompt(self, question, paragraphs, a1, target, condition, order):
        if condition not in CONDITIONS or order not in ORDERS:
            raise ValueError("Unknown anonymity condition or candidate order")
        origins = ("a1", "critic") if order == "a1_first" else ("critic", "a1")
        answers = {"a1": a1, "critic": target}
        return CANDIDATE_TEMPLATE.format(
            paragraphs=self.format_paragraphs(paragraphs), question=question,
            answer_a=answers[origins[0]], answer_b=answers[origins[1]],
            label_a=LABELS[origins[0]] if condition == "labeled" else "",
            label_b=LABELS[origins[1]] if condition == "labeled" else "",
        )

    def compare_candidates(self, question, paragraphs, a1, target, condition, order):
        return self._generate(self.build_candidate_prompt(question, paragraphs, a1, target, condition, order))


def summarize_anonymization(records):
    index = {(r["question_id"], r["arm"], r["instruction_condition"], r["candidate_order"]): r
             for r in records if r["status"] == "completed"}
    by_cell = {}
    for condition, order in CELLS:
        subset = [r for r in records if r["instruction_condition"] == condition and r["candidate_order"] == order]
        by_cell[f"{condition}_{order}"] = summarize_controlled(subset)["arms"]
    label_effects = {}
    order_effects = {}
    for arm in ARMS:
        label_effects[arm] = {}
        order_effects[arm] = {}
        for order in ORDERS:
            pairs = [(r, index[(qid, arm, "anonymous", order)]) for (qid, a, c, o), r in index.items()
                     if a == arm and c == "labeled" and o == order and (qid, arm, "anonymous", order) in index]
            label_effects[arm][order] = {
                "completed_pairs": len(pairs),
                "labeled_adopted": sum(a["adopted_target"] for a, b in pairs),
                "anonymous_adopted": sum(b["adopted_target"] for a, b in pairs),
                "labeled_retained": sum(a["retained_a1"] for a, b in pairs),
                "anonymous_retained": sum(b["retained_a1"] for a, b in pairs),
                "adopted_to_retained": sum(a["adopted_target"] and b["retained_a1"] for a, b in pairs),
                "retained_to_adopted": sum(a["retained_a1"] and b["adopted_target"] for a, b in pairs),
                "accuracy_gains": sum(not a["a2_correct"] and b["a2_correct"] for a, b in pairs),
                "accuracy_losses": sum(a["a2_correct"] and not b["a2_correct"] for a, b in pairs),
            }
        for condition in CONDITIONS:
            pairs = [(r, index[(qid, arm, condition, "critic_first")]) for (qid, a, c, o), r in index.items()
                     if a == arm and c == condition and o == "a1_first" and (qid, arm, condition, "critic_first") in index]
            order_effects[arm][condition] = {
                "completed_pairs": len(pairs),
                "target_adopted_a1_first": sum(a["adopted_target"] for a, b in pairs),
                "target_adopted_critic_first": sum(b["adopted_target"] for a, b in pairs),
                "adoption_changes_with_order": sum(a["adopted_target"] != b["adopted_target"] for a, b in pairs),
                "retained_a1_both_orders": sum(a["retained_a1"] and b["retained_a1"] for a, b in pairs),
                "adopted_target_both_orders": sum(a["adopted_target"] and b["adopted_target"] for a, b in pairs),
                "selected_candidate_a_both_orders": sum(a["selected_candidate"] == b["selected_candidate"] == "A" for a, b in pairs),
                "selected_candidate_b_both_orders": sum(a["selected_candidate"] == b["selected_candidate"] == "B" for a, b in pairs),
            }
    return {"protocol": PROTOCOL, "by_condition_and_order": by_cell,
            "paired_label_effects": label_effects, "paired_order_effects": order_effects,
            "note": "Same frozen A1 and recommendation in each cell. Conditions differ only in source labels; both orders are tested. Cells are repeated trials, not independent questions. All completed outputs remain in denominators; source skips are preserved. No historical baseline is treated as a matched layout control."}


EXTRA_COLUMNS = ["candidate_order", "candidate_a", "candidate_b", "candidate_a_source", "candidate_b_source",
                 "selected_candidate", "source_feedback", "evaluation_prompt"]
PAIR_BASE = ["question_id", "arm", "question", "ground_truth", "solver_attempt_1", "critic_advocated_answer", "source_a2"]
PAIR_COLUMNS = PAIR_BASE + [f"{c}_{o}_{field}" for c, o in CELLS for field in
                           ("status", "solver_attempt_2", "adopted_target", "retained_a1", "a2_correct", "selected_candidate")]


def run_anonymization(rows, examples, sources, solver, output_dir, provenance=None, assigned_orders=None, order_seed=None):
    validate_sources(rows, examples, sources)
    if assigned_orders is not None:
        if set(assigned_orders) != {r["question_id"] for r in rows} or any(order not in ORDERS for order in assigned_orders.values()):
            raise ValueError("Every sampled question must have exactly one valid assigned order")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    metadata = {**(provenance or {}), "protocol": PROTOCOL, "candidate_template": CANDIDATE_TEMPLATE,
                "source_labels": LABELS, "conditions": CONDITIONS, "orders": ORDERS}
    if assigned_orders is not None:
        from mas_sae.experiments.randomized_adoption import PROTOCOL as RANDOMIZED_PROTOCOL
        metadata.update(protocol=RANDOMIZED_PROTOCOL, conditions=("labeled",), order_seed=order_seed,
                        assignment_unit="question", assignment_strata="hop_group x recomputed_a1_correctness")
        # Save the entire assignment before generating any revised answer.
        (output / "order_assignments.json").write_text(json.dumps(assigned_orders, indent=2, sort_keys=True) + "\n")
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    records = []
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as rf, \
         (output / "comparisons.csv").open("w", newline="", encoding="utf-8") as pf:
        writer = csv.DictWriter(rf, fieldnames=RESULT_COLUMNS + EXTRA_COLUMNS)
        pair_writer = csv.DictWriter(pf, fieldnames=PAIR_COLUMNS + (["assigned_order"] if assigned_orders is not None else []))
        writer.writeheader()
        pair_writer.writeheader()
        for i, source in enumerate(sources):
            example = examples[source["question_id"]]
            pair_row = {key: source.get(key, "") for key in PAIR_BASE}
            pair_row["source_a2"] = source.get("solver_attempt_2", "")
            cells = CELLS if assigned_orders is None else (("labeled", assigned_orders[source["question_id"]]),)
            if assigned_orders is not None:
                pair_row["assigned_order"] = assigned_orders[source["question_id"]]
            for condition, order in cells:
                record = {key: source[key] for key in (
                    "question_id", "arm", "question", "ground_truth", "answer_aliases", "context",
                    "solver_attempt_1", "stratum", "target_metadata", "status")}
                record.update(protocol=metadata["protocol"], instruction_condition=condition, candidate_order=order,
                              a1_correct=answer_matches(source["solver_attempt_1"], example["answer"], example.get("answer_aliases", [])))
                if source["status"] == "completed":
                    a1, target = source["solver_attempt_1"], source["critic_advocated_answer"]
                    first, second = ("a1", "critic") if order == "a1_first" else ("critic", "a1")
                    answers = {"a1": a1, "critic": target}
                    prompt = solver.build_candidate_prompt(source["question"], example["paragraphs"], a1, target, condition, order)
                    a2 = solver.compare_candidates(source["question"], example["paragraphs"], a1, target, condition, order)
                    scores = score_revision(a2, a1, target, example["answer"], example.get("answer_aliases", []),
                                            accept_gold_aliases=source["arm"] == "correct_correction")
                    adopted, retained = scores["adopted_target"], scores["retained_a1"]
                    selected = ("A" if first == "critic" else "B") if adopted else (("A" if first == "a1" else "B") if retained else "neither")
                    record.update(critic_advocated_answer=target, critic_noncommittal=False,
                                  source_feedback=source["critic_feedback"], source_a2=source["solver_attempt_2"],
                                  candidate_a=answers[first], candidate_b=answers[second],
                                  candidate_a_source=first, candidate_b_source=second, selected_candidate=selected,
                                  evaluation_prompt=prompt, solver_attempt_2=a2,
                                  generation=dict(getattr(solver, "last_generation", None) or {}),
                                  prompt_sha256=sha256_text(prompt),
                                  a2_correct=scores["a2_correct"],
                                  adopted_target=adopted, retained_a1=retained,
                                  matches_source_a2=answer_matches(a2, source["solver_attempt_2"]))
                    record["candidate_label"] = classify_candidate(record)
                append_jsonl(output / "episodes.jsonl", record)
                flat = result_csv_row(record)
                flat.update({key: record.get(key, "") for key in EXTRA_COLUMNS})
                writer.writerow(flat)
                rf.flush()
                for field in ("status", "solver_attempt_2", "adopted_target", "retained_a1", "a2_correct", "selected_candidate"):
                    pair_row[f"{condition}_{order}_{field}"] = record.get(field, "")
                records.append(record)
            pair_writer.writerow(pair_row)
            pf.flush()
            if (i + 1) % 10 == 0 or i + 1 == len(sources):
                logger.info("Candidate evaluation: %d/%d saved question/arm records processed", i + 1, len(sources))
    if assigned_orders is None:
        summary = summarize_anonymization(records)
    else:
        from mas_sae.experiments.randomized_adoption import summarize_randomized_order
        summary = summarize_randomized_order(records, assigned_orders, order_seed)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
