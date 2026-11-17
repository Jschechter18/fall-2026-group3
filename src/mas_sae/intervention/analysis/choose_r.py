"""The choose-r report: what each candidate r does to real generations.

Why choose-r: the probes cannot calibrate r. In the matched-edits run
(results/probe_mixed_final_001_layer25/matched_edits) the probe's prediction flips at about
0.0007% of |x|, and even random directions swing it at 1-2.5%, so r must come from generations.

choose-r runs on its own episodes (positions N+1..N+P of the seeded order, never study episodes),
and its results are never part of the study tables. The report is for choosing r by hand; the
"meets suggested rule" column applies the suggested rule (targeted − random flip with a 95% CI above
0, and under 5% out and degenerate answers), which the experimenter may override.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mas_sae.intervention.analysis import stats
from mas_sae.intervention.generation.edits import ARMS
from mas_sae.intervention.experiment.episodes import OPPOSITE
from mas_sae.intervention.analysis.tables import to_markdown

if TYPE_CHECKING:
    from pathlib import Path

REPEAT_RUN = 5            # this many identical tokens in a row makes an answer degenerate
MAX_BAD_PCT = 5.0         # suggested rule: under this % of out and of degenerate answers
EXAMPLES_PER_SIZE = 5


def longest_repeat_run(token_ids: list[int]) -> int:
    """Length of the longest run of one token repeated back to back (0 for no tokens)."""
    longest = run = 0
    for i, t in enumerate(token_ids):
        run = run + 1 if i and t == token_ids[i - 1] else 1
        longest = max(longest, run)
    return longest


def is_degenerate(row: dict) -> bool:
    """Empty, stuck repeating one token, or out of tokens without finishing an answer."""
    return (not row["answer"].strip() or row["reached_token_budget"]
            or longest_repeat_run(row["generated_token_ids"]) >= REPEAT_RUN)


def is_flip(row: dict) -> bool:
    """Whether the row's label is the opposite of its stored label."""
    return row["label"] == OPPOSITE[row["stored_label"]]


def choose_r_table(rows: list[dict], representations: list[str], sizes: Sequence[float], n_draws: int,
                   n_bootstrap: int) -> pd.DataFrame:
    """One row per size × representation × arm: targeted and mean random flip %, their paired Δ with a
    bootstrap 95% CI over episodes, % out and % degenerate (targeted and random rows together), and
    whether the suggested rule is met. ``n_draws`` is not used."""
    df = pd.DataFrame([r for r in rows if r["kind"] == "edit"])
    df["flip"] = df.apply(is_flip, axis=1).astype(float)
    df["degenerate"] = df.apply(is_degenerate, axis=1)
    out = []
    for size in sizes:
        for rep in representations:
            for arm in ARMS:
                cell = df[(df.edit_size == size) & (df.representation == rep) & (df.arm == arm)]
                targeted = cell[cell.source == "targeted"].set_index("episode_id").flip
                random = cell[cell.source == "random"].groupby("episode_id").flip.mean().reindex(targeted.index)
                d = (targeted - random).to_numpy()
                ci = [100 * v for v in stats.bootstrap_ci(d, n_bootstrap, seed=0)]
                out_pct, degenerate_pct = 100 * (cell.label == "out").mean(), 100 * cell.degenerate.mean()
                out.append({"Edit size r": size, "Representation": rep, "Arm": arm, "N": len(targeted),
                            "Targeted flip %": 100 * targeted.mean(), "Random flip %": 100 * random.mean(),
                            "Δ (pp)": 100 * d.mean(), "Δ 95% CI": f"[{ci[0]:+.1f}, {ci[1]:+.1f}]",
                            "% out": out_pct, "% degenerate": degenerate_pct,
                            "Meets suggested rule": bool(ci[0] > 0 and out_pct < MAX_BAD_PCT
                                                         and degenerate_pct < MAX_BAD_PCT)})
    return pd.DataFrame(out)


def examples(rows: list[dict], size: float) -> pd.DataFrame:
    """Up to 5 targeted answers at this size, preferring ones that changed from the baseline."""
    baseline = {r["episode_id"]: r["answer"] for r in rows if r["kind"] == "baseline"}
    targeted = [r for r in rows if r["kind"] == "edit" and r["edit_size"] == size and r["source"] == "targeted"]
    targeted.sort(key=lambda r: r["answer"] == baseline[r["episode_id"]])
    return pd.DataFrame([{"Representation": r["representation"], "Arm": r["arm"], "Stored label": r["stored_label"],
                          "Baseline A2": baseline[r["episode_id"]], "Edited A2": r["answer"], "New label": r["label"]}
                         for r in targeted[:EXAMPLES_PER_SIZE]])


def write_choose_r_report(rows: list[dict], excluded: dict, representations: list[str], sizes: Sequence[float],
                       n_draws: int, n_bootstrap: int, csv_path: Path, md_path: Path,
                       labels_stubbed: bool) -> pd.DataFrame:
    """Write the choose-r table to ``csv_path`` and the markdown report to ``md_path`` (episode and
    exclusion counts, the baseline flip rate, the suggested rule, the table and example answers per
    size); returns the table."""
    table = choose_r_table(rows, representations, sizes, n_draws, n_bootstrap)
    table.to_csv(csv_path, index=False)
    baseline_flips = np.mean([is_flip(r) for r in rows if r["kind"] == "baseline"])
    parts = ["# Choose r: edit-size report\n",
             *(["**LABELS STUBBED: these numbers are meaningless (smoke run).**\n"] if labels_stubbed else []),
             f"{len({r['episode_id'] for r in rows})} choose-r episodes; {len(excluded)} excluded for labeling "
             f"failures. Baseline flip rate {100 * baseline_flips:.1f}% (should be ~0).\n",
             "Choose r and write it into the config as `experiment.edit_size`. Suggested rule: the smallest "
             "size where targeted − random is clearly above 0 for at least one SAE, with out and degenerate "
             f"answers under {MAX_BAD_PCT:g}%. Degenerate: empty, {REPEAT_RUN}+ repeats of one token, or the "
             "token budget reached.\n", "## Flip rates\n", to_markdown(table)]
    for size in sizes:
        parts += [f"## Examples at r = {size}\n", to_markdown(examples(rows, size))]
    md_path.write_text("\n".join(parts))
    return table
