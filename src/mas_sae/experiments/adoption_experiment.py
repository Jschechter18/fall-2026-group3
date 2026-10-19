"""Paired feedback ablations with frozen A1 and unchanged Solver prompts.

Automatic behavior labels are diagnostic candidates, never human ground truth.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random

from mas_sae.agents.solver import Solver
from mas_sae.data.musique import hop_group
from mas_sae.evaluation.behavior_labels import classify_candidate
from mas_sae.evaluation.scoring import answer_matches
from mas_sae.experiments.records import append_jsonl

ARMS = ("original", "no_verdict", "answer_only", "no_feedback")
VERDICTS = (
    "The reviewer agrees with your previous answer.",
    "The reviewer disagrees with your previous answer.",
    "The reviewer provided feedback without a clear agree or disagree verdict.",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def stratum(row):
    def flag(value):
        return str(value) if type(value) is bool else "unknown"
    return " / ".join((hop_group(row["question_id"]),
                       "a1=" + flag(row.get("solver_attempt_1_correct")),
                       "critic=" + flag(row.get("critic_feedback_correct"))))


def sample_rows(rows, partitions_path, size=100, seed=42):
    """Equal allocation across available correctness × hop strata, capped by capacity.

    Membership comes exclusively from the frozen research partition file.
    No selection uses A2 or its adoption label.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    with open(partitions_path, newline="") as f:
        entries = list(csv.DictReader(f))
    partitions = {r["question_id"]: r for r in entries}
    if len(partitions) != len(entries):
        raise ValueError("Duplicate question IDs in partition file")
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        if row.get("critic_condition") != "natural":
            continue
        qid = row["question_id"]
        membership = partitions.get(qid)
        if membership is None:
            raise ValueError(f"Question absent from frozen partitions: {qid}")
        if membership["partition"] != "train":
            continue
        if membership["source_split"] != "train":
            raise ValueError("Pilot requires MuSiQue-train development questions")
        if qid in seen:
            raise ValueError(f"Duplicate Natural episode: {qid}")
        seen.add(qid)
        for key in ("question", "solver_attempt_1", "critic_feedback", "ground_truth"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"{qid}: missing {key}")
        groups[stratum(row)].append(row)
    if sum(map(len, groups.values())) < size:
        raise ValueError("Not enough training questions for requested sample")
    rng = random.Random(seed)
    for name in sorted(groups):
        members = groups[name]
        members.sort(key=lambda r: r["question_id"])
        rng.shuffle(members)
    names = sorted(groups)
    rng.shuffle(names)
    counts = Counter()
    selected = []
    while len(selected) < size:
        for name in names:
            if counts[name] < len(groups[name]) and len(selected) < size:
                selected.append(groups[name][counts[name]])
                counts[name] += 1
    rng.shuffle(selected)
    sampling = {name: {"population": len(groups[name]), "selected": counts[name],
                       "inclusion_probability": counts[name] / len(groups[name])}
                for name in sorted(groups)}
    return selected, sampling


def feedback_for(row, arm):
    """Ablate feedback contents only; never change the Solver template."""
    if arm == "original":
        return row["critic_feedback"]
    if arm == "no_feedback":
        return "No additional reviewer feedback is available."
    if arm == "answer_only":
        answer = row.get("critic_advocated_answer")
        return f"The reviewer advocates this answer: {answer}" if answer else None
    if arm == "no_verdict":
        text = row["critic_feedback"]
        for prefix in VERDICTS:
            if text.startswith(prefix):
                return text[len(prefix):].strip() or None
        # Do not guess how to strip unfamiliar feedback formats.
        return None
    raise ValueError(f"Unknown arm: {arm}")


def run_experiment(rows, examples, solver, output_dir, provenance=None):
    """Run all arms from saved A1. Output directory must be new; fail loudly on errors."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    if provenance is not None:
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    records = []
    for row in rows:
        qid = row["question_id"]
        example = examples[qid]
        if example["question"] != row["question"] or example["answer"] != row["ground_truth"]:
            raise ValueError(f"{qid}: source episode differs from pinned dataset")
        for arm in ARMS:
            feedback = feedback_for(row, arm)
            record = {"question_id": qid, "arm": arm, "stratum": stratum(row),
                      "solver_attempt_1": row["solver_attempt_1"],
                      "question": row["question"], "ground_truth": example["answer"],
                      "context": Solver.format_paragraphs(example["paragraphs"]),
                      "critic_feedback": feedback,
                      "critic_advocated_answer": row.get("critic_advocated_answer") if arm != "no_feedback" else None,
                      "critic_noncommittal": row.get("critic_noncommittal", True) if arm != "no_feedback" else True}
            if feedback is None:
                record["status"] = "skipped_unavailable_feedback"
            else:
                prompt = solver.build_revise_prompt(row["question"], example["paragraphs"],
                                                    row["solver_attempt_1"], feedback)
                a2 = solver.revise(row["question"], example["paragraphs"], row["solver_attempt_1"], feedback)
                record.update(status="completed", solver_attempt_2=a2,
                              prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                              generation=getattr(solver, "last_generation", None),
                              historical_a2=row.get("solver_attempt_2"),
                              a1_correct=answer_matches(row["solver_attempt_1"], example["answer"], example.get("answer_aliases", [])),
                              a2_correct=answer_matches(a2, example["answer"], example.get("answer_aliases", [])),
                              retained_a1=answer_matches(a2, row["solver_attempt_1"]))
                record["candidate_label"] = classify_candidate(record)
            append_jsonl(output / "episodes.jsonl", record)
            records.append(record)
    summary = summarize(records)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    export_review(records, output)
    return summary


def summarize(records):
    result = {}
    for arm in ARMS:
        all_rows = [r for r in records if r["arm"] == arm]
        completed = [r for r in all_rows if r["status"] == "completed"]
        eligible = [r for r in completed if r["candidate_label"]["eligible_primary"]]
        counts = Counter(r["candidate_label"]["solver_response"] for r in eligible)
        by_stratum = {}
        for name in sorted({r["stratum"] for r in completed}):
            subset = [r for r in completed if r["stratum"] == name]
            by_stratum[name] = {"completed": len(subset), "a2_correct": sum(r["a2_correct"] for r in subset),
                               "candidate_eligible": sum(r["candidate_label"]["eligible_primary"] for r in subset),
                               "candidate_responses": dict(Counter(r["candidate_label"]["solver_response"] for r in subset))}
        result[arm] = {"completed": len(completed), "skipped": len(all_rows) - len(completed),
                       "candidate_eligible": len(eligible), "candidate_responses": dict(counts),
                       "candidate_adoption_rate": counts["adopted_critic"] / len(eligible) if eligible else None,
                       "retained_a1": sum(r["retained_a1"] for r in completed),
                       "a2_correct": sum(r["a2_correct"] for r in completed),
                       "reached_token_budget": sum(bool((r.get("generation") or {}).get("reached_token_budget")) for r in completed),
                       "historical_a2_matches": sum(answer_matches(r["solver_attempt_2"], r["historical_a2"]) for r in completed if r.get("historical_a2")),
                       "historical_a2_comparable": sum(bool(r.get("historical_a2")) for r in completed),
                       "corrected": sum(not r["a1_correct"] and r["a2_correct"] for r in completed),
                       "harmed": sum(r["a1_correct"] and not r["a2_correct"] for r in completed),
                       "by_stratum": by_stratum}
    baseline = {r["question_id"]: r for r in records if r["arm"] == "original" and r["status"] == "completed"}
    for arm in ARMS[1:]:
        pairs = [(baseline[r["question_id"]], r) for r in records
                 if r["arm"] == arm and r["status"] == "completed" and r["question_id"] in baseline]
        eligible_pairs = [(a, b) for a, b in pairs if a["candidate_label"]["eligible_primary"] and b["candidate_label"]["eligible_primary"]]
        result[arm]["paired_with_original"] = {
            "completed_pairs": len(pairs), "both_candidate_eligible": len(eligible_pairs),
            "adopted_to_retained": sum(a["candidate_label"]["solver_response"] == "adopted_critic" and b["candidate_label"]["solver_response"] == "retained_a1" for a, b in eligible_pairs),
            "retained_to_adopted": sum(a["candidate_label"]["solver_response"] == "retained_a1" and b["candidate_label"]["solver_response"] == "adopted_critic" for a, b in eligible_pairs),
            "accuracy_gains": sum(not a["a2_correct"] and b["a2_correct"] for a, b in pairs),
            "accuracy_losses": sum(a["a2_correct"] and not b["a2_correct"] for a, b in pairs)}
    return {"warning": "Enriched development sample; automatic labels require human QC. No-feedback is a reconsideration control, not rejection.", "arms": result}


def export_review(records, output):
    completed = [r for r in records if r["status"] == "completed"]
    random.Random(42).shuffle(completed)
    columns = ["review_id", "question", "context", "solver_a1", "critic_feedback", "solver_a2",
               "feedback_type", "solver_response", "eligible_primary", "annotator_confidence", "notes"]
    key = []
    with (output / "review.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for i, row in enumerate(completed):
            review_id = f"P{i + 1:04d}"
            writer.writerow({"review_id": review_id, "question": row["question"], "context": row["context"],
                             "solver_a1": row["solver_attempt_1"], "critic_feedback": row["critic_feedback"],
                             "solver_a2": row["solver_attempt_2"]})
            key.append({"review_id": review_id, "question_id": row["question_id"], "arm": row["arm"]})
    (output / "review_private_key.json").write_text(json.dumps(key, indent=2) + "\n")
