"""Paired original versus evidence-verification instructions on frozen feedback."""
import csv
import hashlib
import json
import logging
from pathlib import Path

from mas_sae.agents.solver import Solver
from mas_sae.evaluation.behavior_labels import classify_candidate
from mas_sae.evaluation.scoring import answer_matches
from mas_sae.experiments.artifacts import sha256_file
from mas_sae.experiments.controlled_adoption import ARMS, PROTOCOL as SOURCE_PROTOCOL, summarize_controlled
from mas_sae.experiments.records import append_jsonl, read_jsonl

logger = logging.getLogger(__name__)
PROTOCOL = "adoption_evidence_verification_v1"
CONDITIONS = ("original", "evidence_verification")
VERIFICATION_INSTRUCTION = (
    "Evaluate the reviewer's recommendation against the provided passages. "
    "The reviewer may be mistaken. Change your previous answer when the "
    "passages support a correction. Otherwise, retain your answer if it "
    "remains supported. If neither answer is supported, give the best "
    "supported answer or state that the evidence is insufficient.\n"
)


class EvidenceVerificationSolver(Solver):
    """The original revision template plus one verification instruction."""
    REVISE_PROMPT_V1 = Solver.REVISE_PROMPT_V1.replace(
        "Respond with the final answer only, no explanation.\n",
        VERIFICATION_INSTRUCTION + "Respond with the final answer only, no explanation.\n",
    )


def load_controlled_source(directory, packet, sample_sha256):
    """Require matching sample/model/context settings and a completed source run."""
    directory = Path(directory)
    metadata = json.loads((directory / "provenance.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    if metadata.get("protocol") != SOURCE_PROTOCOL or summary.get("protocol") != SOURCE_PROTOCOL:
        raise ValueError("Expected a completed controlled_adoption_v1 run")
    for key, expected in (("sample_sha256", sample_sha256), ("solver", packet["solver"]),
                          ("dataset", packet["dataset"]), ("revise_template", Solver.REVISE_PROMPT_V1)):
        if metadata.get(key) != expected:
            raise ValueError(f"Controlled run differs from replay configuration: {key}")
    records = read_jsonl(directory / "episodes.jsonl")
    return records, {name: sha256_file(directory / name)
                     for name in ("episodes.jsonl", "provenance.json", "summary.json")}


def validate_sources(rows, examples, sources):
    """Check every source record before inference; never regenerate targets."""
    wanted = {r["question_id"]: r for r in rows}
    if len(wanted) != len(rows):
        raise ValueError("Duplicate questions in sample")
    keys = [(r["question_id"], r["arm"]) for r in sources]
    if len(set(keys)) != len(keys) or set(keys) != {(qid, arm) for qid in wanted for arm in ARMS}:
        raise ValueError("Controlled records must contain exactly one of each arm per sampled question")
    for source in sources:
        qid = source["question_id"]
        row, example = wanted[qid], examples[qid]
        if source.get("protocol") != SOURCE_PROTOCOL:
            raise ValueError(f"{qid}: unexpected source protocol")
        if source["status"] != "completed" and not source["status"].startswith("skipped_"):
            raise ValueError(f"{qid}: unfinished source record")
        for key, expected in (("question", example["question"]), ("ground_truth", example["answer"]),
                              ("solver_attempt_1", row["solver_attempt_1"]),
                              ("context", Solver.format_paragraphs(example["paragraphs"])),
                              ("answer_aliases", example.get("answer_aliases", []))):
            if source.get(key) != expected:
                raise ValueError(f"{qid}: frozen source differs from pinned input: {key}")
        if row["question"] != example["question"] or row["ground_truth"] != example["answer"]:
            raise ValueError(f"{qid}: sample differs from pinned dataset")
        if source["status"] == "completed":
            if not source.get("critic_advocated_answer") or not source.get("critic_feedback"):
                raise ValueError(f"{qid}: completed source lacks a recommendation")
            if source.get("critic_advocated_answer") != source.get("target_metadata", {}).get("answer"):
                raise ValueError(f"{qid}: source recommendation and target disagree")


RESULT_COLUMNS = [
    "question_id", "arm", "instruction_condition", "status", "skip_reason", "question",
    "context", "ground_truth", "answer_aliases", "solver_attempt_1", "a1_correct",
    "critic_advocated_answer", "critic_feedback", "target_source", "target_type_check",
    "solver_attempt_2", "adopted_target", "retained_a1", "a2_correct", "solver_response",
    "corrected", "harmed", "source_a2", "matches_source_a2", "generated_tokens",
    "reached_token_budget", "prompt_sha256",
]
PAIR_COLUMNS = [
    "question_id", "arm", "question", "context", "ground_truth", "solver_attempt_1",
    "a1_correct", "critic_advocated_answer", "critic_feedback", "source_a2",
] + [f"{condition}_{field}" for condition in CONDITIONS for field in
     ("status", "solver_attempt_2", "solver_response", "adopted_target", "retained_a1", "a2_correct")] + [
    "adopted_to_retained", "retained_to_adopted", "accuracy_gain", "accuracy_loss",
]


def result_csv_row(record):
    out = {key: record.get(key, "") for key in RESULT_COLUMNS}
    out["answer_aliases"] = json.dumps(record.get("answer_aliases", []), ensure_ascii=False)
    out["target_source"] = record.get("target_metadata", {}).get("source", "")
    out["target_type_check"] = record.get("target_metadata", {}).get("type_check", "")
    out["skip_reason"] = record.get("target_metadata", {}).get("reason", "")
    out["solver_response"] = record.get("candidate_label", {}).get("solver_response", "")
    for key in ("generated_tokens", "reached_token_budget"):
        out[key] = record.get("generation", {}).get(key, "")
    if record["status"] == "completed":
        out["corrected"] = not record["a1_correct"] and record["a2_correct"]
        out["harmed"] = record["a1_correct"] and not record["a2_correct"]
    return out


def pair_csv_row(original, verification):
    out = {key: original.get(key, "") for key in PAIR_COLUMNS[:10]}
    for condition, record in zip(CONDITIONS, (original, verification)):
        for field in ("status", "solver_attempt_2", "adopted_target", "retained_a1", "a2_correct"):
            out[f"{condition}_{field}"] = record.get(field, "")
        out[f"{condition}_solver_response"] = record.get("candidate_label", {}).get("solver_response", "")
    if original["status"] == verification["status"] == "completed":
        out.update(adopted_to_retained=original["adopted_target"] and verification["retained_a1"],
                   retained_to_adopted=original["retained_a1"] and verification["adopted_target"],
                   accuracy_gain=not original["a2_correct"] and verification["a2_correct"],
                   accuracy_loss=original["a2_correct"] and not verification["a2_correct"])
    return out


def summarize_verification(records):
    conditions = {condition: summarize_controlled([r for r in records if r["instruction_condition"] == condition])["arms"]
                  for condition in CONDITIONS}
    index = {(r["question_id"], r["arm"], r["instruction_condition"]): r for r in records}
    comparisons = {}
    for arm in ARMS:
        pairs = [(r, index[(r["question_id"], arm, "evidence_verification")]) for r in records
                 if r["arm"] == arm and r["instruction_condition"] == "original" and r["status"] == "completed"]
        pairs = [(a, b) for a, b in pairs if b["status"] == "completed"]
        comparisons[arm] = {
            "completed_pairs": len(pairs),
            "original_adopted": sum(a["adopted_target"] for a, b in pairs),
            "verification_adopted": sum(b["adopted_target"] for a, b in pairs),
            "original_retained": sum(a["retained_a1"] for a, b in pairs),
            "verification_retained": sum(b["retained_a1"] for a, b in pairs),
            "adopted_to_retained": sum(a["adopted_target"] and b["retained_a1"] for a, b in pairs),
            "retained_to_adopted": sum(a["retained_a1"] and b["adopted_target"] for a, b in pairs),
            "accuracy_gains": sum(not a["a2_correct"] and b["a2_correct"] for a, b in pairs),
            "accuracy_losses": sum(a["a2_correct"] and not b["a2_correct"] for a, b in pairs),
            "original_matches_source_a2": sum(a["matches_source_a2"] for a, b in pairs),
        }
    return {"protocol": PROTOCOL, "conditions": conditions, "paired_comparisons": comparisons,
            "note": "Same saved feedback and A1 under both instructions. All completed trials remain in denominators. Scores use existing automatic answer matching. Source skips are preserved; no new targets are generated."}


def run_verification(rows, examples, sources, solvers, output_dir, provenance=None):
    validate_sources(rows, examples, sources)
    if set(solvers) != set(CONDITIONS):
        raise ValueError("Both original and evidence_verification solvers are required")
    if solvers["original"].REVISE_PROMPT_V1 != Solver.REVISE_PROMPT_V1 or solvers["evidence_verification"].REVISE_PROMPT_V1 != EvidenceVerificationSolver.REVISE_PROMPT_V1:
        raise ValueError("Unexpected revision template for instruction condition")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    metadata = {**(provenance or {}), "protocol": PROTOCOL,
                "revision_templates": {name: agent.REVISE_PROMPT_V1 for name, agent in solvers.items()}}
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    records = []
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as results_file, \
         (output / "comparisons.csv").open("w", newline="", encoding="utf-8") as pairs_file:
        writer = csv.DictWriter(results_file, fieldnames=RESULT_COLUMNS)
        pairs_writer = csv.DictWriter(pairs_file, fieldnames=PAIR_COLUMNS)
        writer.writeheader()
        pairs_writer.writeheader()
        for i, source in enumerate(sources):
            pair = []
            example = examples[source["question_id"]]
            for condition in CONDITIONS:
                solver = solvers[condition]
                record = {key: source[key] for key in (
                    "question_id", "arm", "question", "ground_truth", "answer_aliases", "context",
                    "solver_attempt_1", "stratum", "target_metadata", "status")}
                record.update(protocol=PROTOCOL, instruction_condition=condition,
                              a1_correct=answer_matches(source["solver_attempt_1"], example["answer"], example.get("answer_aliases", [])))
                if source["status"] == "completed":
                    record.update(critic_feedback=source["critic_feedback"],
                                  critic_advocated_answer=source["critic_advocated_answer"], critic_noncommittal=False,
                                  source_a2=source["solver_attempt_2"])
                    prompt = solver.build_revise_prompt(source["question"], example["paragraphs"], source["solver_attempt_1"], source["critic_feedback"])
                    a2 = solver.revise(source["question"], example["paragraphs"], source["solver_attempt_1"], source["critic_feedback"])
                    record.update(solver_attempt_2=a2, generation=dict(getattr(solver, "last_generation", None) or {}),
                                  prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                                  a2_correct=answer_matches(a2, example["answer"], example.get("answer_aliases", [])),
                                  adopted_target=answer_matches(a2, source["critic_advocated_answer"], example.get("answer_aliases", []) if source["arm"] == "correct_correction" else ()),
                                  retained_a1=answer_matches(a2, source["solver_attempt_1"]),
                                  matches_source_a2=answer_matches(a2, source["solver_attempt_2"]))
                    record["candidate_label"] = classify_candidate(record)
                append_jsonl(output / "episodes.jsonl", record)
                writer.writerow(result_csv_row(record))
                results_file.flush()
                records.append(record)
                pair.append(record)
            pairs_writer.writerow(pair_csv_row(*pair))
            pairs_file.flush()
            if (i + 1) % 10 == 0 or i + 1 == len(sources):
                logger.info("Verification comparison: %d/%d saved question/arm records processed", i + 1, len(sources))
    summary = summarize_verification(records)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
