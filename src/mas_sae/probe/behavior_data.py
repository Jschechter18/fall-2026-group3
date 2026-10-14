"""Loaders and alignment for the Natural-run behavior data, under the 6 Oct split contract.

Three sources are joined through the episode table (interactions.jsonl):

  * interactions.jsonl  -> which activation row belongs to which episode
  * labels.csv          -> behavior labels (primary_target, eligible_primary, ...)
  * the NEW manifest    -> `partition` (train / validation / test / intervention), one row per question

The manifest is the only authority for which rows may be used. ``test`` and ``intervention`` are
sealed: ``select_partitions`` refuses to return them unless ``allow_sealed=True`` (reserved for the
single final evaluation after the freeze, not for development).

The dangerous failure is silent misalignment, so the join is strict: every episode must find its
labels and its manifest row (or be listed, with a reason, in ``excluded``), and every episode's
manifest index (row number inside the tensor for that ``source_split``) must be present and agree
with the row index the interactions file records for the same question. Anything else raises.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mas_sae.data.production import PARTITIONS, PURPOSE_PARTITION

# The partition rules live in mas_sae.data.production; sealed = what only `evaluate`/`intervene` may read.
SEALED_PARTITIONS = frozenset({PURPOSE_PARTITION["evaluate"], PURPOSE_PARTITION["intervene"]})


class AlignmentError(RuntimeError):
    """Activations, labels and partitions could not be joined with confidence."""


@dataclass
class Columns:
    partition: str = "partition"
    source_split: str = "source_split"
    target: str = "primary_target"
    eligible: str = "eligible_primary"
    response: str = "solver_response"
    episode_id: str = "episode_id"
    question_id: str = "question_id"


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_tensor(path: Path) -> np.ndarray:
    """Load a saved activation tensor as a float32 numpy array (rows x dim)."""
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    arr = obj.numpy() if hasattr(obj, "numpy") else np.asarray(obj)
    return arr.astype(np.float32, copy=False)


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def build_episode_table(
    results_dir: Path, source_splits: list[str], text_fields: list[str]
) -> pd.DataFrame:
    """One row per episode. ``source_split`` is the folder the episode (and its tensor) lives in."""
    rows = []
    for split in source_splits:
        records = read_jsonl(Path(results_dir) / split / "interactions.jsonl")
        for r in records:
            row = {
                "source_split": split,
                "activation_row": int(r["attempt2_activation_index"]),
                # exported partition files re-index rows and keep the original in source_activation_index
                "verify_index": int(r.get("source_activation_index", r["attempt2_activation_index"])),
                "ep_episode_id": r.get("episode_id"),
                "ep_question_id": r.get("question_id"),
            }
            for f in text_fields:
                row[f"text_{f}"] = "" if r.get(f) is None else str(r[f])
            rows.append(row)
    if not rows:
        raise AlignmentError(f"No interactions found under {results_dir}")
    df = pd.DataFrame(rows)
    for f in text_fields:  # an all-empty text column would silently weaken the text-only baseline
        if not (df[f"text_{f}"].str.len() > 0).any():
            raise AlignmentError(
                f"text field '{f}' is missing/empty in every interaction record. "
                f"Keys available: {sorted(records[0])}")
    return df


def _examples(ids, n: int = 5) -> str:
    ids = sorted(str(i) for i in ids)
    return ", ".join(ids[:n]) + (" ..." if len(ids) > n else "")


def align_episodes(
    episodes: pd.DataFrame, labels: pd.DataFrame, manifest: pd.DataFrame,
    cols: Columns, index_column: str, excluded: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Attach labels and the partition to every episode, and verify the index semantics.

    ``index_column`` is ``full_index`` for the full production run and ``scan_index`` for the
    nine-layer scan.

    ``excluded`` maps question_id -> reason for episodes deliberately left out of the analysis.
    They are dropped and recorded in the diagnostics. Any other episode that does not find its
    labels and its manifest row raises, so the analysis sample can never shrink silently.
    """
    excluded = {str(k): str(v) for k, v in (excluded or {}).items()}
    if cols.partition not in manifest.columns:
        raise AlignmentError(
            f"The manifest has no '{cols.partition}' column (columns: {list(manifest.columns)[:12]}). "
            "This looks like the OLD canonical_split manifest. Layer selection needs the frozen "
            "four-partition split_manifest.csv (train / validation / test / intervention).")
    for needed in (cols.question_id, cols.source_split, index_column):
        if needed not in manifest.columns:
            raise AlignmentError(f"Manifest lacks required column '{needed}'.")
    unknown = set(manifest[cols.partition].astype(str)) - set(PARTITIONS)
    if unknown:
        raise AlignmentError(f"Manifest has unknown partition value(s) {sorted(unknown)}; "
                             f"expected a subset of {list(PARTITIONS)}.")

    diag: dict = {"n_episodes": len(episodes), "n_labels": len(labels), "n_manifest": len(manifest),
                  "index_column": index_column}

    # ---- deliberate exclusions: must name real episodes, and are dropped before joining
    episode_ids = set(episodes["ep_question_id"].astype(str))
    stale = set(excluded) - episode_ids
    if stale:
        raise AlignmentError(f"{len(stale)} excluded question id(s) are not episodes of this run "
                             f"({_examples(stale)}). Remove them from `excluded_episodes`.")
    episodes = episodes[~episodes["ep_question_id"].astype(str).isin(excluded)]
    diag["n_excluded"] = len(excluded)
    diag["excluded_episodes"] = dict(sorted(excluded.items()))

    # ---- labels -> episodes (episode_id if both sides have it, else question_id)
    if cols.episode_id in labels and episodes["ep_episode_id"].notna().all():
        lab_key, ep_key = cols.episode_id, "ep_episode_id"
    elif cols.question_id in labels and episodes["ep_question_id"].notna().all():
        lab_key, ep_key = cols.question_id, "ep_question_id"
    else:
        raise AlignmentError(
            f"labels.csv shares neither '{cols.episode_id}' nor '{cols.question_id}' with "
            f"interactions.jsonl. labels columns: {list(labels.columns)[:12]}")
    lab_cols = [lab_key, cols.target, cols.eligible] + ([cols.response] if cols.response in labels else [])
    try:
        df = episodes.merge(labels[lab_cols], left_on=ep_key, right_on=lab_key,
                            how="left", validate="one_to_one", indicator="_in_labels")
        diag["labels_join"] = lab_key
        m = manifest[[cols.question_id, cols.partition, cols.source_split, index_column]].rename(
            columns={cols.question_id: "m_question_id", cols.source_split: "m_source_split",
                     index_column: "m_index"})
        df = df.merge(m, left_on="ep_question_id", right_on="m_question_id",
                      how="left", validate="many_to_one", indicator="_in_manifest")
    except pd.errors.MergeError as e:
        raise AlignmentError(f"Join was not one-to-one: {e}") from e

    # ---- every remaining episode must have found its labels and its manifest row
    no_labels = df.loc[df["_in_labels"] == "left_only", "ep_question_id"]
    no_manifest = df.loc[df["_in_manifest"] == "left_only", "ep_question_id"]
    if len(no_labels) or len(no_manifest):
        raise AlignmentError(
            f"{len(no_labels)} episode(s) have no row in labels.csv ({_examples(no_labels)}) and "
            f"{len(no_manifest)} have no row in the manifest ({_examples(no_manifest)}). Every episode "
            "must be labelled and partitioned; to leave one out on purpose, list it with a reason "
            "under `excluded_episodes`.")
    df = df.drop(columns=["_in_labels", "_in_manifest"])
    if df.empty:
        raise AlignmentError("Join produced 0 rows.")

    # ---- the manifest index must be the row number inside that source_split's tensor
    bad_split = df[df["m_source_split"].astype(str) != df["source_split"].astype(str)]
    if len(bad_split):
        raise AlignmentError(
            f"{len(bad_split)} questions sit in the '{bad_split['source_split'].iloc[0]}' folder but the "
            f"manifest says source_split='{bad_split['m_source_split'].iloc[0]}'. Do not proceed.")
    missing_index = df["m_index"].isna()
    if missing_index.any():
        raise AlignmentError(
            f"{int(missing_index.sum())} of {len(df)} episodes have no {index_column} in the manifest "
            f"({_examples(df.loc[missing_index, 'ep_question_id'])}), so their activation rows cannot be "
            f"verified. Check index_column (full_index vs scan_index) matches the run you pointed at.")
    bad = df[df["m_index"].astype(int) != df["verify_index"]]
    if len(bad):
        raise AlignmentError(
            f"Manifest {index_column} disagrees with attempt2_activation_index for {len(bad)} of "
            f"{len(df)} questions (e.g. {bad['ep_question_id'].iloc[0]}: manifest "
            f"{int(bad['m_index'].iloc[0])} vs interactions {int(bad['verify_index'].iloc[0])}). "
            "Check index_column (full_index vs scan_index) matches the run you pointed at, and "
            "that the manifest and the activation export come from the same behavior package.")
    diag["index_verified_rows"] = len(df)
    diag["n_joined"] = len(df)
    return df, diag


def target_names(series: pd.Series, positive: str = "adopted_critic", negative: str = "not_adopted") -> pd.Series:
    """Readable class names. In labels.csv `primary_target` is numeric: 1.0 = adopted the Critic,
    0.0 = did not (retained A1 or a third answer), blank when the row is not eligible."""
    if pd.api.types.is_numeric_dtype(series):
        out = series.map({1.0: positive, 0.0: negative})
        if out.isna().any():
            raise AlignmentError(f"unexpected values in the target column: "
                                 f"{sorted(series[out.isna()].dropna().unique().tolist())} (expected 0/1)")
        return out.astype(str)
    return series.astype(str)


def select_partitions(
    df: pd.DataFrame, cols: Columns, partitions: list[str], allow_sealed: bool = False
) -> tuple[pd.DataFrame, dict]:
    """Rows in the requested partitions that are eligible for the primary target.

    ``test`` and ``intervention`` are sealed during development: requesting them raises.
    """
    sealed = SEALED_PARTITIONS & set(partitions)
    if sealed and not allow_sealed:
        raise PermissionError(
            f"Partition(s) {sorted(sealed)} are sealed (test is scored once after the freeze; "
            "intervention is for the causal stage). Do not use them to fit, tune or select anything.")
    in_part = df[df[cols.partition].astype(str).isin(partitions)]
    out = in_part[as_bool(in_part[cols.eligible])].reset_index(drop=True)
    out["target_name"] = target_names(out[cols.target])
    for p in partitions:
        if not (out[cols.partition].astype(str) == p).any():
            raise AlignmentError(f"No eligible_primary rows in partition '{p}'.")
    diag = {
        "partitions": list(partitions),
        "n_rows_in_partitions": len(in_part),
        "n_eligible": len(out),
        "target_counts_by_partition": {
            str(p): g["target_name"].value_counts().to_dict() for p, g in out.groupby(cols.partition)},
    }
    if cols.response in out:
        diag["solver_response_by_partition"] = {
            str(p): g[cols.response].value_counts().to_dict() for p, g in out.groupby(cols.partition)}
    return out, diag


def load_layer_matrix(
    df: pd.DataFrame, layer_dir: Path, activation_file: str = "{split}_attempt2.pt"
) -> np.ndarray:
    """Activation rows for ``df`` (in df order) from one layer folder."""
    out = None
    for split, g in df.groupby("source_split"):
        arr = load_tensor(Path(layer_dir) / activation_file.format(split=split))
        rows = g["activation_row"].to_numpy()
        if rows.max() >= arr.shape[0]:
            raise AlignmentError(
                f"{layer_dir}/{activation_file.format(split=split)} has {arr.shape[0]} rows "
                f"but interactions reference row {rows.max()}.")
        if out is None:
            out = np.empty((len(df), arr.shape[1]), dtype=np.float32)
        positions = np.flatnonzero((df["source_split"] == split).to_numpy())
        out[positions] = arr[rows]
    return out
