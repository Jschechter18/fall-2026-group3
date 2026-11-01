"""Paper figure and table for the synthetic pilot (#174, #179): behavior labels per cohort.

Each cohort is a Claude labeling run (``raw_results.jsonl``) and, optionally, the collection's
``exclusions.jsonl``. A question excluded at collection (e.g. the Critic gave no recommendation) is
counted as Out, as the final labeling rules count collection failures. Inputs are hashed before
they are read and rechecked before anything is written. Writes:

* ``<figures>/synthetic_vs_musique_behavior.{svg,pdf}``: one 100% stacked bar per cohort, segments
  Adoption, Retention, Agreement and Out, with the Adoption + Retention share at the right.
* ``<tables>/synthetic_vs_musique_behavior.csv``: counts, the A+R share with its 95% Wilson interval,
  and the Adoption share of A+R; ``synthetic_vs_musique_inputs.json``: the sha256 of every input.

Refuses a question labeled twice, a question that is both labeled and excluded, and technical
labeling errors (they are not behavior and must be retried first).

Usage:
    python scripts/plot_synthetic_pilot.py \\
        --cohort "Synthetic, random fillers" results/claude_validation/pilot174_synthetic100_opus5_run1/raw_results.jsonl \\
        --cohort "Synthetic, related fillers" results/claude_validation/pilot174_related_fillers99_opus5_run1/raw_results.jsonl \\
                 results/collection/synthetic_sample_100q_related_fillers/train/exclusions.jsonl \\
        --cohort "MuSiQue control" results/claude_validation/pilot174_musique100_opus5_run1/raw_results.jsonl \\
        --figures reports/figures/synthetic_pilot --tables reports/tables/synthetic_pilot
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["svg.hashsalt"] = "synthetic-pilot"  # stable SVG element ids

from mas_sae.evaluation.claude_agreement import wilson  # noqa: E402

LABELS = ("adoption", "retention", "agreement", "out")
NAMES = {"adoption": "Adoption", "retention": "Retention", "agreement": "Agreement", "out": "Out"}
# Adoption follows the Critic (orange) and Retention keeps A1 (blue), the same hues as the candidate-order
# figure; validated categorical pair (CVD dE 24.7). Agreement and Out are not probe targets: recessive grays.
COLORS = {"adoption": "#eb6834", "retention": "#2a78d6", "agreement": "#d3d1cb", "out": "#8f8d87"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e4e3df"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def cohort_counts(labels_path: Path, exclusions_path: Path | None) -> dict:
    """Four-class counts for one cohort; collection exclusions count as Out."""
    seen: dict[str, str] = {}
    for row in read_jsonl(labels_path):
        qid = row["question_id"]
        if qid in seen:
            raise SystemExit(f"{labels_path}: {qid} is labeled twice")
        if row.get("behavior_label") not in LABELS:
            raise SystemExit(f"{labels_path}: {qid} has no behavior label ({row.get('exclusion_reason')}); retry it first")
        seen[qid] = row["behavior_label"]
    excluded = [row["question_id"] for row in read_jsonl(exclusions_path)] if exclusions_path else []
    both = sorted(set(excluded) & set(seen))
    if both or len(excluded) != len(set(excluded)):
        raise SystemExit(f"{exclusions_path}: excluded questions repeat or are also labeled, e.g. {both[:3]}")
    counts = {label: sum(v == label for v in seen.values()) for label in LABELS}
    counts["out"] += len(excluded)
    return counts | {"n": len(seen) + len(excluded), "collection_failures": len(excluded)}


def table_row(name: str, counts: dict) -> dict:
    n, binary = counts["n"], counts["adoption"] + counts["retention"]
    low, high = wilson(binary, n)
    return {"cohort": name, "n": n, **{label: counts[label] for label in LABELS},
            "collection_failures": counts["collection_failures"], "a_plus_r_share": binary / n,
            "a_plus_r_ci_low": low, "a_plus_r_ci_high": high,
            "adoption_share_of_a_plus_r": counts["adoption"] / binary if binary else None}


def plot(rows: list[dict], outputs: list[Path]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 0.62 * len(rows) + 1.25))
    for y, row in enumerate(reversed(rows)):
        left = 0.0
        for label in LABELS:
            share = row[label] / row["n"]
            if share:
                ax.barh(y, share, left=left, height=0.62, color=COLORS[label], edgecolor="white", linewidth=2, zorder=2)
                if share >= 0.045:
                    ink = "white" if label in ("adoption", "retention") else INK
                    ax.text(left + share / 2, y, f"{share:.0%}", ha="center", va="center", fontsize=8.5, color=ink, zorder=3)
            left += share
        ax.text(1.015, y, f"A+R {row['a_plus_r_share']:.0%}", ha="left", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(rows)), [f"{r['cohort']} (n={r['n']})" for r in reversed(rows)])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9, length=0)
    ax.set_xlabel("Share of episodes (Claude Opus 5 labels)", fontsize=9, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[label]) for label in LABELS]
    ax.legend(handles, [NAMES[label] for label in LABELS], frameon=False, fontsize=8.5, ncol=4,
              loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1.2, columnspacing=1.2)
    fig.tight_layout()
    for path in outputs:  # no timestamps, so regenerating from the same inputs gives identical files
        metadata = {"CreationDate": None} if path.suffix == ".pdf" else {"Date": None}
        fig.savefig(path, bbox_inches="tight", facecolor="white", metadata=metadata)
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cohort", nargs="+", action="append", required=True,
                        metavar=("NAME LABELS", "EXCLUSIONS"), help="name, raw_results.jsonl[, exclusions.jsonl]")
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    args = parser.parse_args(argv)
    if any(len(c) not in (2, 3) for c in args.cohort):
        parser.error("--cohort takes a name, a raw_results.jsonl and optionally an exclusions.jsonl")

    rows, hashes = [], {}
    for name, labels, *exclusions in args.cohort:
        paths = [Path(labels), *(Path(e) for e in exclusions)]
        hashes[name] = {str(p): file_sha256(p) for p in paths}
        rows.append(table_row(name, cohort_counts(paths[0], paths[1] if len(paths) > 1 else None)))
    if any(file_sha256(Path(p)) != h for files in hashes.values() for p, h in files.items()):
        raise SystemExit("an input file changed while it was read; nothing written")
    args.figures.mkdir(parents=True, exist_ok=True)
    args.tables.mkdir(parents=True, exist_ok=True)
    with (args.tables / "synthetic_vs_musique_behavior.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.tables / "synthetic_vs_musique_inputs.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    plot(rows, [args.figures / "synthetic_vs_musique_behavior.svg", args.figures / "synthetic_vs_musique_behavior.pdf"])
    for row in rows:
        print(f"{row['cohort']:28s} n={row['n']:3d}  " + "  ".join(f"{NAMES[l][:3]} {row[l]:2d}" for l in LABELS)
              + f"  A+R {row['a_plus_r_share']:.0%} [{row['a_plus_r_ci_low']:.0%}, {row['a_plus_r_ci_high']:.0%}]")


if __name__ == "__main__":
    main()
