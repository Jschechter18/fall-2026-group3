"""Paired protocol comparison using saved Natural Critic recommendations.

Coherence screening uses only A1 and Critic fields. Historical A2, eligibility,
combined audit candidate_status and correctness never decide inclusion.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from mas_sae.agents.critic import Critic
from mas_sae.agents.solver import Solver
from mas_sae.evaluation.behavior import textual_relation
from mas_sae.evaluation.scoring import answer_matches
from mas_sae.experiments.adoption_experiment import stratum
from mas_sae.experiments.anonymized_adoption import CANDIDATE_TEMPLATE
from mas_sae.experiments.randomized_adoption import assign_orders, order_only_baseline
from mas_sae.experiments.records import append_jsonl

SAMPLE_PROTOCOL = "adoption_natural_sample_v1"
PROTOCOL = "adoption_natural_paired_v1"
LABEL_VERSION = "natural_adoption_lexical_v1"
CONDITIONS = ("original", "labeled_randomized")


def screen_natural(rows, audit_rows):
    """Join #115 train coherence to original inputs; fail on stale/missing joins.

    'consistent' is absence of detected warnings, not human-confirmed coherence.
    Textually different answers can still be semantically equivalent.
    """
    audit = {}
    for entry in audit_rows:
        if entry["partition"] != "train":
            raise ValueError("Natural screening requires a train-only audit")
        qid = entry["question_id"]
        if qid in audit:
            raise ValueError(f"Duplicate audit question: {qid}")
        audit[qid] = entry
    selected, counts = [], Counter()
    for row in rows:
        qid = row["question_id"]
        if qid not in audit:
            raise ValueError(f"Missing audit question: {qid}")
        entry = audit[qid]
        feedback = Critic.parse_natural(row["critic_raw_output"])
        for key, expected in (("solver_a1", row["solver_attempt_1"]),
                              ("delivered_verdict", feedback.verdict or ""),
                              ("delivered_advocated_answer", feedback.advocated_answer or ""),
                              ("delivered_explanation", feedback.explanation)):
            if entry[key] != expected:
                raise ValueError(f"Stale audit input: {qid}, {key}")
        if feedback.to_solver_text() != row["critic_feedback"]:
            raise ValueError(f"Saved feedback differs from parsed delivery: {qid}")
        if feedback.advocated_answer != row.get("critic_advocated_answer"):
            raise ValueError(f"Saved recommendation differs from parsed delivery: {qid}")
        if entry["coherence"] not in ("consistent", "uncertain", "inconsistent", "unusable"):
            raise ValueError(f"Unknown coherence: {qid}")
        if entry["coherence"] != "consistent":
            counts[entry["coherence"]] += 1
        elif feedback.noncommittal or not feedback.advocated_answer:
            counts["missing_recommendation"] += 1
        elif textual_relation(row["solver_attempt_1"], feedback.advocated_answer) != "different":
            counts["no_clear_lexical_conflict"] += 1
        else:
            selected.append({**row, "natural_screen": {
                "coherence": entry["coherence"], "coherence_reasons": entry["coherence_reasons"],
                "eligibility": "automatic_screen_only_not_human_confirmed",
            }})
    return selected, {"input_questions": len(rows), "screen_pass": len(selected), "excluded": dict(counts)}


def summarize_natural(records):
    def counts(rows):
        return dict(trials=len(rows), questions=len({r["question_id"] for r in rows}),
                    adopted=sum(r["adopted_target"] for r in rows),
                    retained_a1=sum(r["retained_a1"] for r in rows),
                    other=sum(r["response_class"] == "other" for r in rows),
                    ambiguous_overlap=sum(r["response_class"] == "ambiguous_overlap" for r in rows),
                    a2_correct=sum(r["a2_correct"] for r in rows),
                    adopted_wrong=sum(r["adopted_target"] and not r["recommendation_correct"] for r in rows),
                    wrong_recommendations=sum(not r["recommendation_correct"] for r in rows),
                    adopted_correct=sum(r["adopted_target"] and r["recommendation_correct"] for r in rows),
                    correct_recommendations=sum(r["recommendation_correct"] for r in rows),
                    reached_token_budget=sum(bool(r["generation"].get("reached_token_budget")) for r in rows))
    by_condition = {c: counts([r for r in records if r["instruction_condition"] == c]) for c in CONDITIONS}
    index = {(r["question_id"], r["instruction_condition"]): r for r in records}
    pairs = [(r, index[(r["question_id"], CONDITIONS[1])]) for r in records if r["instruction_condition"] == CONDITIONS[0]]
    candidate = [b for _, b in pairs]
    return {"protocol": PROTOCOL, "label_version": LABEL_VERSION, "conditions": by_condition,
            "paired": {"questions": len(pairs),
                       "response_transitions": dict(Counter(f"{a['response_class']} -> {b['response_class']}" for a, b in pairs)),
                       "accuracy_gains": sum(not a["a2_correct"] and b["a2_correct"] for a, b in pairs),
                       "accuracy_losses": sum(a["a2_correct"] and not b["a2_correct"] for a, b in pairs),
                       "original_matches_historical_a2": sum(a["matches_historical_a2"] for a, _ in pairs)},
            "by_candidate_order": {o: counts([r for r in candidate if r["candidate_order"] == o]) for o in ("a1_first", "critic_first")},
            "order_only_baseline": order_only_baseline(candidate),
            "note": "Natural recommendations, separate from controlled targets. Original receives full feedback; candidates receive answer only. Automatic lexical labels and eligibility require semantic review. No activation capture."}


def run_natural(rows, examples, solver, candidate_solver, output_dir, seed=42, provenance=None):
    """Freeze order before inference, replay only A2, and save paired outputs."""
    for row in rows:
        example = examples[row["question_id"]]
        if row.get("critic_condition") != "natural" or row.get("natural_screen", {}).get("coherence") != "consistent":
            raise ValueError("Expected screened Natural episodes")
        if row["question"] != example["question"] or row["ground_truth"] != example["answer"]:
            raise ValueError("Sample differs from pinned dataset")
        if textual_relation(row["solver_attempt_1"], row["critic_advocated_answer"]) != "different":
            raise ValueError("Natural candidates must differ")
    if solver.REVISE_PROMPT_V1 != Solver.REVISE_PROMPT_V1:
        raise ValueError("Unexpected original revision template")
    assignments = assign_orders(rows, examples, seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    metadata = {**(provenance or {}), "protocol": PROTOCOL, "label_version": LABEL_VERSION,
                "order_seed": seed, "revision_template": Solver.REVISE_PROMPT_V1,
                "candidate_template": CANDIDATE_TEMPLATE, "captures_activations": False}
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "order_assignments.json").write_text(json.dumps(assignments, indent=2) + "\n")
    records, pairs = [], []
    with (output / "results.csv").open("w", newline="") as file:
        writer = None
        for row in rows:
            qid = row["question_id"]
            ex = examples[qid]
            target, a1 = row["critic_advocated_answer"], row["solver_attempt_1"]
            correct = answer_matches(target, ex["answer"], ex.get("answer_aliases", []))
            pair = {"question_id": qid, "candidate_order": assignments[qid]}
            for condition in CONDITIONS:
                if condition == "original":
                    prompt = solver.build_revise_prompt(row["question"], ex["paragraphs"], a1, row["critic_feedback"])
                    agent = solver
                else:
                    prompt = candidate_solver.build_candidate_prompt(row["question"], ex["paragraphs"], a1, target, "labeled", assignments[qid])
                    agent = candidate_solver
                a2 = agent._generate(prompt)
                adopted = answer_matches(a2, target, ex.get("answer_aliases", []) if correct else ())
                retained = answer_matches(a2, a1)
                response = "ambiguous_overlap" if adopted and retained else "adopted" if adopted else "retained_a1" if retained else "other"
                record = dict(protocol=PROTOCOL, label_version=LABEL_VERSION, recommendation_source="natural",
                              question_id=qid, arm="natural", instruction_condition=condition, status="completed",
                              candidate_order=assignments[qid], question=row["question"],
                              context=Solver.format_paragraphs(ex["paragraphs"]), ground_truth=ex["answer"],
                              answer_aliases=ex.get("answer_aliases", []), solver_attempt_1=a1,
                              critic_advocated_answer=target, critic_feedback=row["critic_feedback"],
                              historical_a2=row.get("solver_attempt_2"), evaluation_prompt=prompt,
                              prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                              solver_attempt_2=a2, adopted_target=adopted, retained_a1=retained,
                              response_class=response, recommendation_correct=correct,
                              a1_correct=answer_matches(a1, ex["answer"], ex.get("answer_aliases", [])),
                              a2_correct=answer_matches(a2, ex["answer"], ex.get("answer_aliases", [])),
                              matches_historical_a2=answer_matches(a2, row.get("solver_attempt_2") or ""),
                              stratum=stratum(row), natural_screen=row["natural_screen"],
                              generation=dict(getattr(agent, "last_generation", None) or {}))
                append_jsonl(output / "episodes.jsonl", record)
                if writer is None:
                    writer = csv.DictWriter(file, fieldnames=list(record))
                    writer.writeheader()
                writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in record.items()})
                file.flush()
                records.append(record)
                for key in ("solver_attempt_2", "response_class", "a2_correct", "adopted_target", "retained_a1"):
                    pair[f"{condition}_{key}"] = record[key]
            pairs.append(pair)
    with (output / "comparisons.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    summary = summarize_natural(records)
    with (output / "slide_table.csv").open("w", newline="") as file:
        table = [{"recommendation_source": "natural", "method": c, **v} for c, v in summary["conditions"].items()]
        writer = csv.DictWriter(file, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
