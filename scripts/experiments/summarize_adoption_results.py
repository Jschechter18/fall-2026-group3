"""Export paired preliminary presentation tables from completed adoption runs."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

METHODS = [
    ("Original revision baseline", "controlled_100q", {}),
    ("Evidence verification", "verification_100q", {"instruction_condition": "evidence_verification"}),
    ("Labeled candidates: A1 first", "anonymization_100q", {"instruction_condition": "labeled", "candidate_order": "a1_first"}),
    ("Labeled candidates: critic first", "anonymization_100q", {"instruction_condition": "labeled", "candidate_order": "critic_first"}),
    ("Anonymous candidates: A1 first", "anonymization_100q", {"instruction_condition": "anonymous", "candidate_order": "a1_first"}),
    ("Anonymous candidates: critic first", "anonymization_100q", {"instruction_condition": "anonymous", "candidate_order": "critic_first"}),
    ("Labeled candidates: randomized order", "randomized_order_100q", {}),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_methods(root):
    cache, input_hashes = {}, {}
    reference = None
    sample_hash = None
    result = []
    for name, folder, filters in METHODS:
        if folder not in cache:
            directory = root / folder
            # A final summary is the completion marker.
            json.loads((directory / "summary.json").read_text())
            provenance = json.loads((directory / "provenance.json").read_text())
            if not provenance.get("sample_sha256"):
                raise ValueError(f"Missing source sample hash: {folder}")
            if sample_hash is None:
                sample_hash = provenance["sample_sha256"]
            if provenance["sample_sha256"] != sample_hash:
                raise ValueError("Runs use different question samples")
            cache[folder] = [json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines() if line.strip()]
            for filename in ("summary.json", "provenance.json", "episodes.jsonl"):
                path = directory / filename
                input_hashes[str(path)] = sha(path)
        rows = [r for r in cache[folder] if r["status"] == "completed" and all(r.get(k) == v for k, v in filters.items())]
        index = {(r["question_id"], r["arm"]): r for r in rows}
        if len(index) != len(rows) or not rows:
            raise ValueError(f"Duplicate or empty completed trials: {name}")
        if reference is None:
            reference = index
        if index.keys() != reference.keys():
            raise ValueError(f"Mismatched completed question/arm pairs: {name}")
        for key, row in index.items():
            for field in ("question", "ground_truth", "solver_attempt_1", "critic_advocated_answer", "context"):
                if row[field] != reference[key][field]:
                    raise ValueError(f"Input differs across methods: {name}, {key}, {field}")
            if row["adopted_target"] and row["retained_a1"]:
                raise ValueError("Adoption and retention overlap; a binary table would be ambiguous")
        result.append((name, rows))
    return result, input_hashes, sample_hash


def summarize(name, rows):
    wrong = [r for r in rows if r["arm"] != "correct_correction"]
    correct = [r for r in rows if r["arm"] == "correct_correction"]
    n = len(rows)
    adopted = sum(r["adopted_target"] for r in rows)
    retained = sum(r["retained_a1"] for r in rows)
    bad_adopt = sum(r["adopted_target"] for r in wrong)
    good_adopt = sum(r["adopted_target"] for r in correct)
    return dict(method=name, completed_trials=n, distinct_questions=len({r["question_id"] for r in rows}),
                adopted=adopted, retained_a1=retained, other=n - adopted - retained,
                adoption_pct=round(100 * adopted / n, 1), retention_pct=round(100 * retained / n, 1),
                wrong_recommendation_trials=len(wrong), incorrect_accept=bad_adopt,
                incorrect_accept_pct=round(100 * bad_adopt / len(wrong), 1),
                correct_nonadoption=len(wrong) - bad_adopt,
                retained_against_wrong=sum(r["retained_a1"] for r in wrong),
                other_against_wrong=sum(not r["adopted_target"] and not r["retained_a1"] for r in wrong),
                correct_recommendation_trials=len(correct), correct_accept=good_adopt,
                correct_accept_pct=round(100 * good_adopt / len(correct), 1),
                incorrect_nonadoption=len(correct) - good_adopt,
                retained_against_correct=sum(r["retained_a1"] for r in correct),
                other_against_correct=sum(not r["adopted_target"] and not r["retained_a1"] for r in correct),
                final_correct=sum(r["a2_correct"] for r in rows))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def render_chart(path, summary):
    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="adoption-summary-mpl-"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    labels = [r["method"].replace(" candidates:", ":") for r in summary]
    y = np.arange(len(summary))
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 7.5), gridspec_kw={"width_ratios": [1.1, 1]})
    colors = {"adopted": "#4575b4", "retained_a1": "#1b9e77", "other": "#bdbdbd"}
    starts = np.zeros(len(summary))
    for key, label in (("adopted", "Adopted"), ("retained_a1", "Retained A1"), ("other", "Other")):
        values = np.array([r[key] for r in summary])
        left.barh(y, values, left=starts, color=colors[key], label=label, height=.65)
        for i, value in enumerate(values):
            if value >= 4:
                left.text(starts[i] + value / 2, i, str(value), ha="center", va="center", color="white" if key != "other" else "black", fontsize=10)
        starts += values
    left.set_yticks(y, labels)
    left.invert_yaxis()
    left.set_xlim(0, summary[0]["completed_trials"])
    left.set_xlabel("Completed trials (counts)")
    left.set_title("Behavioral class balance", fontsize=12)
    left.legend(loc="upper center", bbox_to_anchor=(.5, -.13), ncol=3, frameon=False, fontsize=9)
    for offset, key, label, color in ((-.18, "incorrect_accept_pct", "Adopted wrong recommendation", "#d95f02"),
                                      (.18, "correct_accept_pct", "Adopted correct recommendation", "#7570b3")):
        values = [r[key] for r in summary]
        right.barh(y + offset, values, height=.32, label=label, color=color)
        for i, value in enumerate(values):
            right.text(value + 1, i + offset, f"{value:.0f}%", va="center", fontsize=9)
    right.set_yticks(y, [""] * len(summary))
    right.invert_yaxis()
    right.set_xlim(0, 110)
    right.set_xticks([0, 25, 50, 75, 100])
    right.set_xlabel("Adoption rate (%) within recommendation group")
    right.set_title("Wrong recommendations (n=53) vs correct (n=45)", fontsize=12)
    right.legend(loc="upper center", bbox_to_anchor=(.5, -.13), frameon=False, fontsize=9)
    for axis in (left, right):
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Answer adoption depends on the interaction protocol", fontsize=16, y=.97)
    fig.text(.5, .91, f"Preliminary paired comparison: {summary[0]['completed_trials']} trials from {summary[0]['distinct_questions']} unique questions", ha="center", fontsize=11)
    fig.subplots_adjust(left=.27, right=.97, top=.84, bottom=.24, wspace=.2)
    fig.text(.02, .035, "Same selected development sample; repeated trials are not independent questions. Automatic answer matching and controlled targets.\nNon-adoption may be retention, a third answer, or uncertainty. These results do not establish a mechanism or population performance.", fontsize=9, color="#444444")
    fig.savefig(path / "adoption_comparison.png", dpi=240)
    fig.savefig(path / "adoption_comparison.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/adoption_experiment"))
    parser.add_argument("--output", type=Path, default=Path("results/adoption_experiment/presentation_summary"))
    parser.add_argument("--natural-run", type=Path, help="Completed run-natural directory; append a separate paired Natural table")
    parser.add_argument("--chart", action="store_true", help="Also generate PNG/SVG charts (requires matplotlib)")
    args = parser.parse_args()
    methods, hashes, sample_hash = read_methods(args.results_root)
    summary = [summarize(name, rows) for name, rows in methods]
    baseline = summary[0]
    for row in summary:
        for field in ("adopted", "retained_a1", "incorrect_accept", "correct_accept", "final_correct"):
            row[field + "_delta_vs_baseline"] = row[field] - baseline[field]
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "method_summary.csv", summary)
    slide = [dict(method=r["method"], adopted=r["adopted"], retained_a1=r["retained_a1"], other=r["other"],
                  incorrectly_accepted_wrong=f"{r['incorrect_accept']}/{r['wrong_recommendation_trials']}",
                  correctly_accepted_correct=f"{r['correct_accept']}/{r['correct_recommendation_trials']}") for r in summary]
    write_csv(args.output / "slide_table.csv", slide)
    outcomes = [{key: r[key] for key in ("method", "wrong_recommendation_trials", "incorrect_accept", "correct_nonadoption", "retained_against_wrong", "other_against_wrong", "correct_recommendation_trials", "correct_accept", "incorrect_nonadoption", "retained_against_correct", "other_against_correct")} for r in summary]
    write_csv(args.output / "recommendation_outcomes.csv", outcomes)
    randomized = methods[-1][1]
    orders = []
    for order in ("a1_first", "critic_first"):
        rows = [r for r in randomized if r["candidate_order"] == order]
        orders.append(dict(order=order, trials=len(rows), adopted=sum(r["adopted_target"] for r in rows),
                           retained_a1=sum(r["retained_a1"] for r in rows),
                           other=sum(not r["adopted_target"] and not r["retained_a1"] for r in rows)))
    write_csv(args.output / "randomized_order_breakdown.csv", orders)
    main_table = table(["Condition", "Adopted", "Retained A1", "Other", "Adopted wrong", "Adopted correct"],
                       [list(row.values()) for row in slide])
    randomized_summary = json.loads((args.results_root / "randomized_order_100q/summary.json").read_text())
    heuristic = randomized_summary["order_only_baseline"]["all_completed_adoption_vs_nonadoption"]
    text = f'''# Preliminary adoption experiment summary

## Slide-ready comparison

Same {baseline['completed_trials']} completed question/recommendation trials from {baseline['distinct_questions']} unique questions in each row. The original baseline below is the controlled-feedback revision run, not an estimate over the full natural-feedback collection.

{main_table}

## Suggested slide takeaway

**Presenting both answers as labeled candidates yielded substantial adoption and retention without explicitly instructing rejection.** In this selected sample, randomized labeled presentation changed the strict class counts from {baseline['adopted']} adoptions / {baseline['retained_a1']} retentions to {summary[-1]['adopted']} / {summary[-1]['retained_a1']}.

This is a change of interaction protocol: original feedback revision versus labeled candidate evaluation. It does not establish that deference in the original system has been fixed.

## Order check

{table(['Assigned order', 'Trials', 'Adopted', 'Retained A1', 'Other'], [list(row.values()) for row in orders])}

A fixed first-candidate heuristic scored {100 * heuristic['balanced_accuracy']:.1f}% balanced accuracy on adoption/non-adoption. This is a descriptive sanity check, not a trained or held-out probe; it does not prove that position effects are absent.

## What to say aloud

- The original revision setup strongly favored adoption, including wrong recommendations.
- Labeled candidate presentation produced both outcomes in each randomized-order group.
- Anonymity suppressed useful corrections and showed order sensitivity, so randomized labeled presentation is the leading protocol.
- The objective is measurable behavioral variation for probing, not necessarily improved task accuracy.
- Freeze the protocol and validate on fresh development questions before scaling activation collection.

## Interpretation limits

- These are exploratory results on one repeatedly used, stratified development sample. Do not describe these rows as independent datasets or claim statistical significance from this table.
- Trials share questions, and some questions have two wrong-recommendation conditions. The source run had 100 sampled questions, with 98 completed trials across 80 distinct questions after eligibility and target-generation skips.
- Correct/incorrect recommendations are operationalized by dataset gold/alias matching. Matching can miss semantic equivalence; intended plausibility levels and entity types are imperfect.
- Accept means output matches the recommended answer (gold aliases also count for correct corrections). Retain means output matches A1. Other includes third answers and uncertainty; it is not silently counted as retention.
- In recommendation_outcomes.csv, non-adoption includes retention and other outputs. A correct_nonadoption of a wrong recommendation does not imply a correct final answer.
- The fresh randomized run reproduced the corresponding earlier labeled cells in the prior analysis. This supports reproducibility, not generalization to unseen questions.
- The data are controlled recommendation tests; they do not improve or measure the natural Qwen critic's accuracy. No new activation or probe result is reported.

## Files

- slide_table.csv: compact presentation table.
- method_summary.csv: counts, percentages, final correctness, distinct questions and baseline deltas.
- recommendation_outcomes.csv: four recommendation-following outcomes with retention/other splits.
- randomized_order_breakdown.csv: adoption and retention within each assigned order.
- adoption_comparison.png / .svg: slide-ready chart, when --chart is enabled.
- provenance.json: input checksums and source sample checksum.
'''
    if args.natural_run:
        from mas_sae.experiments.natural_adoption import CONDITIONS, PROTOCOL, summarize_natural
        natural_path = args.natural_run
        natural_summary = json.loads((natural_path / "summary.json").read_text())
        natural_provenance = json.loads((natural_path / "provenance.json").read_text())
        if natural_summary.get("protocol") != PROTOCOL or natural_provenance.get("protocol") != PROTOCOL:
            raise ValueError("Expected completed Natural paired experiment")
        natural_records = [json.loads(line) for line in (natural_path / "episodes.jsonl").read_text().splitlines() if line.strip()]
        natural_index = {(r["question_id"], r["instruction_condition"]): r for r in natural_records}
        ids = {r["question_id"] for r in natural_records}
        if not ids or len(natural_index) != len(natural_records) or set(natural_index) != {(qid, c) for qid in ids for c in CONDITIONS}:
            raise ValueError("Natural results must contain exactly two paired records per question")
        for qid in ids:
            a, b = (natural_index[(qid, c)] for c in CONDITIONS)
            for key in ("question", "context", "ground_truth", "solver_attempt_1", "critic_advocated_answer", "critic_feedback", "candidate_order"):
                if a[key] != b[key]:
                    raise ValueError(f"Unpaired Natural inputs: {qid}, {key}")
        if summarize_natural(natural_records) != natural_summary:
            raise ValueError("Natural summary does not match saved episodes")
        natural_table = [{"method": c, **natural_summary["conditions"][c]} for c in CONDITIONS]
        write_csv(args.output / "natural_slide_table.csv", natural_table)
        headers = ["Condition", "Questions", "Adopted", "Retained A1", "Other", "Overlap", "A2 correct"]
        natural_display = [[r["method"], r["questions"], r["adopted"], r["retained_a1"], r["other"], r["ambiguous_overlap"], r["a2_correct"]] for r in natural_table]
        text += "\n## Natural-recommendation extension (separate paired sample)\n\n" + table(headers, natural_display)
        text += "\n\nOriginal revision receives full saved Natural feedback; labeled randomized candidates receive only the saved recommendation. This tests the whole protocol change. Counts are not pooled with controlled targets. Eligibility and behavior labels are automatic and require semantic review. Overlap reports outputs matching both candidates under aliases. No activations are captured.\n"
        for filename in ("summary.json", "provenance.json", "episodes.jsonl"):
            path = natural_path / filename
            hashes[str(path)] = sha(path)
    (args.output / "presentation_summary.md").write_text(text)
    (args.output / "provenance.json").write_text(json.dumps({"sample_sha256": sample_hash, "input_sha256": hashes,
                                                         "script_sha256": sha(Path(__file__))}, indent=2) + "\n")
    if args.chart:
        render_chart(args.output, summary)
    print(main_table)
    print(f"\nSaved presentation artifacts to {args.output}")


if __name__ == "__main__":
    main()
