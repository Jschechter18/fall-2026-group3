"""Select the 250-episode candidates-presentation pilot for the Claude validator (issue #155).

From the frozen *train* partition of the candidates run (natural_4b_full9_candidates):

    evaluation   200  seeded, 50 per PR #153 label (adoption, retention, agreement, out),
                      never a team-reviewed question
    challenge     50  all 46 team-reviewed reference questions, plus one question for each
                      of 4 difficulty flags (letter_reply, citation_only,
                      critic_mentions_a1, potential_hybrid)

Writes ``sampled_episodes.jsonl`` (the unmodified source records in a seeded shuffled order,
the only file the labeling run reads, so Claude never sees PR #153 labels, flags or team
decisions) and ``selection.json`` (set membership, PR #153 label/flags/team label, seed and
input hashes, for the comparison only). Refuses labels whose checksum differs from the
approved PR #153 ``counts.json``.

Usage:
    PYTHONPATH=src python scripts/sample_candidate_pilot.py \\
        --run results/collection/natural_4b_full9_candidates \\
        --partitions results/behavior/behavior_v012_official_20261011/partitions.csv \\
        --labels <dir>/final_labels.jsonl \\
        --output results/claude_validation/samples/candidates_pilot250_seed20261029
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from mas_sae.experiments.artifacts import ensure_output_available, sha256_file
from mas_sae.experiments.records import read_jsonl, write_jsonl

APPROVED_COUNTS = Path("results/experiments/candidate_final_labels_20261027/counts.json")
CLASSES = ("adoption", "retention", "agreement", "out")
PER_CLASS = 50
DIFFICULT_FLAGS = ("letter_reply", "citation_only", "critic_mentions_a1", "potential_hybrid")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="candidates run results folder")
    parser.add_argument("--partitions", type=Path, required=True, help="frozen partitions.csv used by PR #153")
    parser.add_argument("--labels", type=Path, required=True, help="PR #153 final_labels.jsonl for --run")
    parser.add_argument("--counts", type=Path, default=APPROVED_COUNTS, help="approved PR #153 counts.json")
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def select_pilot(labels: list[dict], seed: int) -> dict[str, list[dict]]:
    """``{"evaluation": [...], "challenge": [...]}`` label records from the collected train partition.

    Deterministic for a seed and independent of input order; every question appears once.
    """
    pool = sorted((r for r in labels if r["partition"] == "train" and r["collected"]),
                  key=lambda r: r["question_id"])
    team = [r for r in pool if "team_reviewed" in r["flags"]]
    team_ids = {r["question_id"] for r in team}
    rng = random.Random(seed)

    evaluation = []
    for label in CLASSES:
        members = [r for r in pool if r["label"] == label and r["question_id"] not in team_ids]
        if len(members) < PER_CLASS:
            raise ValueError(f"only {len(members)} '{label}' questions available, need {PER_CLASS}")
        rng.shuffle(members)
        evaluation += members[:PER_CLASS]

    taken = team_ids | {r["question_id"] for r in evaluation}
    difficult = []
    for flag in DIFFICULT_FLAGS:
        members = [r for r in pool if flag in r["flags"] and r["question_id"] not in taken]
        if not members:
            raise ValueError(f"no unused '{flag}' question left for the challenge set")
        pick = rng.choice(members)
        difficult.append(pick)
        taken.add(pick["question_id"])

    selected = evaluation + team + difficult
    if len({r["question_id"] for r in selected}) != len(selected):
        raise ValueError("pilot selection repeats a question")
    return {"evaluation": evaluation, "challenge": team + difficult,
            "challenge_kinds": {**{r["question_id"]: "team_reviewed" for r in team},
                                **{r["question_id"]: f"difficult:{f}" for r, f in zip(difficult, DIFFICULT_FLAGS)}}}


def main(argv=None) -> None:
    args = parse_args(argv)
    ensure_output_available(args.output)

    approved = json.loads(args.counts.read_text())
    labels_sha = hashlib.sha256(args.labels.read_bytes()).hexdigest()
    if labels_sha != approved["labels_sha256"]:
        raise SystemExit(f"{args.labels} sha256 {labels_sha[:12]}… is not the approved PR #153 labels "
                         f"({approved['labels_sha256'][:12]}…)")
    labels = read_jsonl(args.labels)
    picked = select_pilot(labels, args.seed)

    source = {}
    for split in ("train", "validation"):
        for row in read_jsonl(args.run / split / "interactions.jsonl"):
            source[row["question_id"]] = row
    selected = picked["evaluation"] + picked["challenge"]
    missing = [r["question_id"] for r in selected if r["question_id"] not in source]
    if missing:
        raise SystemExit(f"{len(missing)} selected question(s) are not in {args.run}: {missing[:3]}")

    order = [r["question_id"] for r in selected]
    random.Random(args.seed + 1).shuffle(order)  # position carries no set membership
    episodes = [source[q] for q in order]

    sets = {r["question_id"]: "evaluation" for r in picked["evaluation"]}
    sets.update({q: f"challenge:{kind}" for q, kind in picked["challenge_kinds"].items()})
    by_id = {r["question_id"]: r for r in selected}
    train_pool = Counter(r["label"] for r in labels if r["partition"] == "train" and r["collected"])
    packet = {
        "seed": args.seed,
        "policy_version": approved["policy_version"],
        "labels_sha256": labels_sha,
        "input_hashes": {str(p): sha256_file(p) for p in (args.labels, args.counts, args.partitions,
                                                          args.run / "train/interactions.jsonl",
                                                          args.run / "validation/interactions.jsonl")},
        "train_population_label_counts": dict(train_pool),
        "set_sizes": dict(Counter(s.split(":")[0] for s in sets.values())),
        "evaluation_label_counts": dict(Counter(r["label"] for r in picked["evaluation"])),
        "challenge_kind_counts": dict(Counter(picked["challenge_kinds"].values())),
        "episodes": [{"question_id": q, "episode_id": source[q]["episode_id"], "set": sets[q],
                      "pr153_label": by_id[q]["label"], "pr153_flags": by_id[q]["flags"],
                      "team_label": by_id[q].get("team_label"),
                      "candidate_order": source[q].get("candidate_order")} for q in order],
    }
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "sampled_episodes.jsonl", episodes)
    (args.output / "selection.json").write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: packet[k] for k in ("set_sizes", "evaluation_label_counts", "challenge_kind_counts")}, indent=2))
    print(f"Wrote {len(episodes)} episodes to {args.output}")


if __name__ == "__main__":
    main()
