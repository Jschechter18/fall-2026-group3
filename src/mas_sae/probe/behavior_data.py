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
from mas_sae.evaluation.paired_ablation import correctness_cell

# The partition rules live in mas_sae.data.production; sealed = what only `evaluate`/`intervene` may read.
SEALED_PARTITIONS = frozenset({PURPOSE_PARTITION["evaluate"], PURPOSE_PARTITION["intervene"]})

# Episode fields kept as ``meta_<field>`` for grouping and breakdowns; None when a run does not record
# one (runs before the distributed collection have no split_group_id, dataset_source or hop_group).
# ``correctness_cell`` is derived from the Solver A1 and Critic correctness flags.
META_FIELDS = ("split_group_id", "dataset_source", "hop_group", "candidate_order")


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
    # behavior label column (e.g. Claude's behavior_label); when set, select_partitions enforces the
    # binary contract: eligible rows carry only the two target labels, each with its own target value
    label: str | None = None


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_tensor(path: Path) -> np.ndarray:
    """Load a saved activation tensor as a float32 numpy array (rows x dim)."""
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    arr = obj.numpy() if hasattr(obj, "numpy") else np.asarray(obj)
    return arr.astype(np.float32, copy=False)


TRUE_VALUES = frozenset({"true", "1", "yes"})
FALSE_VALUES = frozenset({"false", "0", "no"})


def _parse_bool(value) -> bool | None:
    """True/False for a recognised representation; None for nulls and anything unrecognised."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        # CSV parsing turns a 0/1 column into int, or into float (1.0/0.0) when it has blanks.
        return {1: True, 0: False}.get(value) if not pd.isna(value) else None
    if isinstance(value, str):
        text = value.strip().lower()
        return True if text in TRUE_VALUES else False if text in FALSE_VALUES else None
    return None


def as_bool(series: pd.Series) -> pd.Series:
    """Strict boolean parsing: nulls and unrecognised values raise instead of becoming False,
    so a malformed eligibility column can never silently remove experimental rows."""
    if series.dtype == bool:
        return series
    parsed = series.map(_parse_bool)
    bad = series[parsed.isna()]
    if len(bad):
        raise AlignmentError(
            f"{len(bad)} value(s) in '{series.name}' are null or not a recognised boolean "
            f"(e.g. {_examples(bad.astype(str).unique())}). Expected one of "
            f"{sorted(TRUE_VALUES)} or {sorted(FALSE_VALUES)}.")
    return parsed.astype(bool)


def _strict_index(value, field_name: str, question_id) -> int:
    """An activation row index as an int, refusing nulls, strings, bools and non-integral floats
    instead of truncating them."""
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return int(value)
    raise AlignmentError(f"{field_name}={value!r} for question {question_id} is not an integer row index.")


def build_episode_table(
    results_dir: Path, source_splits: list[str], text_fields: list[str]
) -> pd.DataFrame:
    """One row per episode. ``source_split`` is the folder the episode (and its tensor) lives in."""
    rows = []
    for split in source_splits:
        records = read_jsonl(Path(results_dir) / split / "interactions.jsonl")
        for r in records:
            qid = r.get("question_id")
            row = {
                "source_split": split,
                "activation_row": _strict_index(r.get("attempt2_activation_index"),
                                                "attempt2_activation_index", qid),
                # exported partition files re-index rows and keep the original in source_activation_index
                "verify_index": _strict_index(
                    r.get("source_activation_index", r.get("attempt2_activation_index")),
                    "source_activation_index", qid),
                "ep_episode_id": r.get("episode_id"),
                "ep_question_id": r.get("question_id"),
                **{f"meta_{f}": r.get(f) for f in META_FIELDS},
                "meta_correctness_cell": correctness_cell(r) if "solver_attempt_1_correct" in r else None,
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
    if cols.label is not None and cols.label not in labels:
        raise AlignmentError(f"labels.csv has no '{cols.label}' column, so the binary contract cannot be "
                             f"checked. labels columns: {list(labels.columns)[:12]}")
    lab_cols = ([lab_key, cols.target, cols.eligible] + ([cols.response] if cols.response in labels else [])
                + ([cols.label] if cols.label is not None else []))
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
    df["m_index"] = [_strict_index(v, index_column, q) for v, q in zip(df["m_index"], df["ep_question_id"])]
    bad = df[df["m_index"] != df["verify_index"]]
    if len(bad):
        raise AlignmentError(
            f"Manifest {index_column} disagrees with attempt2_activation_index for {len(bad)} of "
            f"{len(df)} questions (e.g. {bad['ep_question_id'].iloc[0]}: manifest "
            f"{int(bad['m_index'].iloc[0])} vs interactions {int(bad['verify_index'].iloc[0])}). "
            "Check index_column (full_index vs scan_index) matches the run you pointed at, and "
            "that the manifest and the activation export come from the same behavior package.")
    check_activation_rows(df)
    diag["index_verified_rows"] = len(df)
    diag["n_joined"] = len(df)
    return df, diag


def check_activation_rows(df: pd.DataFrame) -> None:
    """Both index sources can agree on an invalid value, so validate the rows themselves: every
    index is non-negative, and each question maps to its own row (one-to-one within a source_split)."""
    for column in ("activation_row", "verify_index"):
        negative = df[df[column] < 0]
        if len(negative):
            raise AlignmentError(
                f"{len(negative)} episode(s) have a negative {column} "
                f"({_examples(negative['ep_question_id'])}); NumPy would silently read from the end.")
        shared = df[df.duplicated(["source_split", column], keep=False)]
        if len(shared):
            raise AlignmentError(
                f"{len(shared)} episodes share an activation row ({column}) within a source_split "
                f"({_examples(shared['ep_question_id'])}); each question needs its own row.")
    repeated = df[df.duplicated(["source_split", "ep_question_id"], keep=False)]
    if len(repeated):
        raise AlignmentError(f"Question(s) appear more than once within a source_split "
                             f"({_examples(repeated['ep_question_id'].unique())}).")


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


def check_binary_contract(df: pd.DataFrame, cols: Columns, class_names: tuple[str, str]) -> None:
    """Eligible rows may carry only the two target labels: ``class_names[0]`` with target 1 and
    ``class_names[1]`` with target 0. Agreement, third answers, Out and technical failures must be
    ineligible; turning one into the negative class would change what the probe measures."""
    if cols.label is None:
        return
    positive, negative = class_names
    labels = df[cols.label].astype(str)
    other = df[~labels.isin([positive, negative])]
    if len(other):
        raise AlignmentError(
            f"{len(other)} eligible row(s) have label(s) {sorted(other[cols.label].astype(str).unique())} "
            f"({_examples(other['ep_question_id'])}). Only {positive!r} (target 1) and {negative!r} (target 0) "
            "may enter the binary target; every other outcome must be ineligible.")
    expected = np.where(labels == positive, 1.0, 0.0)
    wrong = df[pd.to_numeric(df[cols.target], errors="coerce").to_numpy() != expected]
    if len(wrong):
        raise AlignmentError(
            f"{len(wrong)} eligible row(s) have a {cols.target} that disagrees with their label "
            f"(e.g. {wrong['ep_question_id'].iloc[0]}: {wrong[cols.label].iloc[0]!r} with "
            f"{cols.target}={wrong[cols.target].iloc[0]!r}).")


def episode_groups(df: pd.DataFrame, column: str | None) -> tuple[np.ndarray, str]:
    """The unit no fold, holdout or bootstrap may split, and its name.

    ``column`` (``split_group_id``) keeps related variants of one question family together. A run
    that records it for no episode falls back to the question id; one that records it for only some
    episodes raises, since a partial grouping would let families leak across folds.
    """
    questions = df["ep_question_id"].astype(str).to_numpy()
    meta = f"meta_{column}"
    if column in (None, "question_id") or meta not in df or df[meta].isna().all():
        return questions, "question_id"
    missing = df[meta].isna()
    if missing.any():
        raise AlignmentError(f"{int(missing.sum())} of {len(df)} episodes have no {column} "
                             f"({_examples(df.loc[missing, 'ep_question_id'])}) while the others do.")
    return df[meta].astype(str).to_numpy(), column


def select_partitions(
    df: pd.DataFrame, cols: Columns, partitions: list[str], allow_sealed: bool = False,
    class_names: tuple[str, str] = ("adopted_critic", "not_adopted"),
) -> tuple[pd.DataFrame, dict]:
    """Rows in the requested partitions that are eligible for the primary target.

    ``test`` and ``intervention`` are sealed during development: requesting them raises.
    ``class_names`` are the readable names of target 1 and target 0; with ``cols.label`` they are also
    the only labels an eligible row may carry (``check_binary_contract``).
    """
    sealed = SEALED_PARTITIONS & set(partitions)
    if sealed and not allow_sealed:
        raise PermissionError(
            f"Partition(s) {sorted(sealed)} are sealed (test only compares completed SAE versions to choose "
            "the one carried forward; intervention is for the causal stage). Do not use them to fit or tune anything.")
    in_part = df[df[cols.partition].astype(str).isin(partitions)]
    out = in_part[as_bool(in_part[cols.eligible])].reset_index(drop=True)
    check_binary_contract(out, cols, class_names)
    out["target_name"] = target_names(out[cols.target], *class_names)
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
        if not np.issubdtype(rows.dtype, np.integer):
            raise AlignmentError(f"activation_row for '{split}' is {rows.dtype}, not integer row indices.")
        if rows.min() < 0 or rows.max() >= arr.shape[0]:
            raise AlignmentError(
                f"{layer_dir}/{activation_file.format(split=split)} has {arr.shape[0]} rows "
                f"but interactions reference rows {rows.min()}..{rows.max()}.")
        if len(np.unique(rows)) != len(rows):
            raise AlignmentError(f"Two episodes reference the same row of "
                                 f"{layer_dir}/{activation_file.format(split=split)}.")
        if out is None:
            out = np.empty((len(df), arr.shape[1]), dtype=np.float32)
        positions = np.flatnonzero((df["source_split"] == split).to_numpy())
        out[positions] = arr[rows]
    return out
