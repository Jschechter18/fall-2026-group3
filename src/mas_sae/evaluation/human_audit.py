"""Human audit of the Claude behavior labels (#237): a blind annotation packet for a seeded,
stratified sample of one labeled partition, and the scoring of the returned files.

Each sampled episode is shown to one annotator exactly as Claude saw it: the question, A1, the
Critic's recommended answer (C1), A2, and which text the Solver saw as Candidate A / Candidate B.
Claude's label and reason, the dataset source and the episode id are never written to the
annotation files; ``selection.json`` maps item ids back to episode ids for scoring, which joins
Claude's labels from the committed labeling run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import yaml

from mas_sae.evaluation.claude_agreement import cohen_kappa, confusion, per_class, wilson
from mas_sae.evaluation.claude_validator import (
    _CANDIDATES_NOTE,
    _PROMPT_TEMPLATE,
    CANDIDATE_SOURCES,
    VALID_LABELS,
    prompt_sha256,
)
from mas_sae.experiments.artifacts import (
    ensure_output_available,
    get_git_commit,
    get_git_diff,
    get_run_command,
    get_untracked_files,
    sha256_file,
    write_json_atomic,
)
from mas_sae.experiments.records import read_jsonl, write_csv

ANNOTATION_COLUMNS = ["item_id", "question", "A1", "C1", "A2", "candidates_shown", "human_label", "notes"]

# Rows written to the annotation files may only carry these fields: nothing that reveals Claude's
# label, the dataset source or the episode id.
_BLIND_FIELDS = frozenset(ANNOTATION_COLUMNS)

# The columns an annotator reads but must not edit.
_SHOWN_COLUMNS = [c for c in ANNOTATION_COLUMNS if c not in ("human_label", "notes")]


def load_audit_config(path: str | Path) -> dict[str, Any]:
    """Read the audit YAML and check its quotas and annotator sizes add up to the same total."""
    config = yaml.safe_load(Path(path).read_text()) or {}
    for key in ("claude_run", "interactions", "output", "partition", "seed", "quotas", "annotators"):
        if key not in config:
            raise ValueError(f"{path}: missing '{key}'.")
    unknown = set(config["quotas"]) - set(VALID_LABELS)
    if unknown:
        raise ValueError(f"{path}: quotas name labels that are not Claude labels: {sorted(unknown)}.")
    total = sum(n for by_source in config["quotas"].values() for n in by_source.values())
    if sum(config["annotators"].values()) != total:
        raise ValueError(f"{path}: annotator sizes sum to {sum(config['annotators'].values())}, "
                         f"but the quotas sum to {total}.")
    return config


def index_unique(rows: list[dict[str, Any]], key: str, source: str) -> dict[str, dict[str, Any]]:
    """``{row[key]: row}``; refuses a repeated key instead of keeping the last row."""
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row[key] in out:
            raise ValueError(f"{source} repeats {key} {row[key]!r}.")
        out[row[key]] = row
    return out


def audit_pool(labels: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter]:
    """Claude's own labels, which are the ones audited, and a count of the rows left out by reason.

    Technical failures (no label) and rule-assigned labels are left out, each checked on its own row.
    """
    def audited(row: dict[str, Any]) -> bool:
        return row["label_source"] == "claude" and row["claude_label"] in VALID_LABELS

    pool = [r for r in labels.values() if audited(r)]
    left_out = Counter(f"{r['label_source']}:{r['exclusion_reason']}" for r in labels.values() if not audited(r))
    return pool, left_out


def stratified_sample(rows: list[dict[str, Any]], stratum: Callable[[dict[str, Any]], tuple[str, str]],
                      quotas: dict[str, dict[str, int]], seed: int) -> list[dict[str, Any]]:
    """Exactly ``quotas[label][source]`` rows from each stratum, drawn with ``seed``.

    Each stratum's pool is sorted by episode id before drawing, so the sample depends only on the
    rows and the seed, not on file order. Refuses a stratum with fewer rows than its quota.
    """
    pools: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        pools.setdefault(stratum(row), []).append(row)
    rng = random.Random(seed)
    sample = []
    for label in sorted(quotas):
        for source in sorted(quotas[label]):
            want = quotas[label][source]
            pool = sorted(pools.get((label, source), []), key=lambda r: r["episode_id"])
            if len(pool) < want:
                raise ValueError(f"Stratum ({label}, {source}) has {len(pool)} episodes; the quota is {want}.")
            sample.extend(rng.sample(pool, want))
    return sample


def assign_annotators(sample: list[dict[str, Any]], sizes: dict[str, int]) -> list[str]:
    """One annotator per sampled row, dealt in turn over the strata-ordered sample so each person
    gets a similar mix of labels and sources; an annotator whose size is reached is skipped."""
    if sum(sizes.values()) != len(sample):
        raise ValueError(f"Annotator sizes sum to {sum(sizes.values())}, but the sample has {len(sample)} rows.")
    names = list(sizes)
    counts = Counter()
    assigned = []
    turn = 0
    for _ in sample:
        while counts[names[turn % len(names)]] >= sizes[names[turn % len(names)]]:
            turn += 1
        name = names[turn % len(names)]
        counts[name] += 1
        assigned.append(name)
        turn += 1
    return assigned


def candidates_shown_text(candidate_order: str) -> str:
    """'Candidate A = A1, Candidate B = C1' (or the reverse) for the order the Solver saw."""
    sources = CANDIDATE_SOURCES[candidate_order]
    return f"Candidate A = {sources['A']}, Candidate B = {sources['B']}"


def blind_row(item_id: str, episode: dict[str, Any]) -> dict[str, str]:
    """The annotation row for one episode: what Claude saw, with empty label and notes columns."""
    return {"item_id": item_id, "question": episode["question"], "A1": episode["solver_attempt_1"],
            "C1": episode["critic_advocated_answer"], "A2": episode["solver_attempt_2"],
            "candidates_shown": candidates_shown_text(episode["candidate_order"]),
            "human_label": "", "notes": ""}


def guide_text(sizes: dict[str, int]) -> str:
    """Annotator instructions, with the labeling rules taken from Claude's prompt verbatim."""
    rules = _PROMPT_TEMPLATE.split("Resolve your decision before responding.")[0].rstrip()
    people = ", ".join(f"{name}.csv ({n} items)" for name, n in sizes.items())
    return (
        "# Human audit of the Claude labels (#237)\n\n"
        f"Files: {people}. Label only your own file.\n\n"
        "For each row, read the question, A1 (the Solver's first answer), C1 (the Critic's recommended "
        "answer) and A2 (the Solver's final answer). `candidates_shown` says which text the Solver saw "
        "as Candidate A and Candidate B, so a reply such as \"Candidate B\" means that text.\n\n"
        "Write one of `adoption`, `retention`, `agreement`, `out` in `human_label`. Use `notes` for "
        "anything unclear. Don't look up Claude's labels or the answer key until your file is done.\n\n"
        "## Rules (the same rules Claude was given)\n\n"
        f"{rules}\n\n{_CANDIDATES_NOTE.strip().replace('candidates_shown_to_solver', '`candidates_shown`')}\n"
    )


def build_packet(config: dict[str, Any], root: Path = Path(".")) -> dict[str, Any]:
    """Sample, check and write the blind packet; returns the selection record.

    Refuses: an existing output folder; an episodes file whose sha256 differs from the labeling run's
    ``episodes_sha256``; a labeling prompt that differs from this code's (so the guide's rules are the
    ones Claude used); repeated episode ids; sampled rows outside ``partition``; and any input file
    that changes while the packet is built.
    """
    run_dir = root / config["claude_run"]
    raw_path, settings_path = run_dir / "raw_results.jsonl", run_dir / "run_settings.json"
    interactions_path = root / config["interactions"]
    output = root / config["output"]
    if output.exists():
        raise FileExistsError(f"{output} exists; the packet is never rebuilt over earlier annotations.")

    hashes = {str(p.relative_to(root)): sha256_file(p) for p in (raw_path, settings_path)}  # before reading
    hashes[config["interactions"]] = sha256_file(interactions_path)
    settings = json.loads(settings_path.read_text())
    if hashes[config["interactions"]] != settings["episodes_sha256"]:
        raise ValueError(f"{interactions_path} is not the episodes file Claude labeled (sha256 differs).")
    if prompt_sha256(True, settings["a2_presentation"]) != settings["prompt_sha256"]:
        raise ValueError("The labeling prompt in code differs from the one the run recorded.")

    labels = index_unique(read_jsonl(raw_path), "episode_id", str(raw_path))
    episodes = index_unique(read_jsonl(interactions_path), "episode_id", str(interactions_path))
    if set(labels) != set(episodes):
        raise ValueError("The labeling run and the episodes file cover different episodes.")
    pool, left_out = audit_pool(labels)

    sample = stratified_sample(pool, lambda r: (r["claude_label"], r["dataset_source"]),
                               config["quotas"], config["seed"])
    annotators = assign_annotators(sample, config["annotators"])
    order = list(range(len(sample)))
    random.Random(config["seed"] + 1).shuffle(order)  # item ids carry no stratum or annotator order

    items = []
    for position, i in enumerate(order, 1):
        row = sample[i]
        episode = episodes[row["episode_id"]]
        if episode["partition"] != config["partition"] or row["partition"] != config["partition"]:
            raise ValueError(f"{row['episode_id']} is not in the {config['partition']} partition.")
        items.append({"item_id": f"H{position:03d}", "episode_id": row["episode_id"],
                      "question_id": row["question_id"], "dataset_source": row["dataset_source"],
                      "partition": row["partition"], "annotator": annotators[i]})
    check_selection(items, labels, config)

    for path, digest in [(raw_path, hashes[str(raw_path.relative_to(root))]),
                         (settings_path, hashes[str(settings_path.relative_to(root))]),
                         (interactions_path, hashes[config["interactions"]])]:
        if sha256_file(path) != digest:
            raise ValueError(f"{path} changed while the packet was built.")

    output.mkdir(parents=True)
    (output / "annotations").mkdir()
    files = {}
    for name in config["annotators"]:
        rows = [blind_row(it["item_id"], episodes[it["episode_id"]]) for it in items if it["annotator"] == name]
        rows.sort(key=lambda r: r["item_id"])
        if any(set(r) - _BLIND_FIELDS for r in rows):
            raise ValueError("An annotation row carries a field outside the blind columns.")
        path = output / "annotations" / f"{name}.csv"
        ensure_output_available(path)
        write_csv(path, rows, lineterminator="\n")  # LF, so the recorded hash matches the committed file
        files[str(path.relative_to(output))] = sha256_file(path)
    guide = output / "GUIDE.md"
    guide.write_text(guide_text(config["annotators"]), encoding="utf-8")
    files["GUIDE.md"] = sha256_file(guide)

    selection = {
        "issue": 237, "partition": config["partition"], "seed": config["seed"], "quotas": config["quotas"],
        "annotators": config["annotators"], "git_commit": get_git_commit(),
        "git_dirty": bool(get_git_diff() or get_untracked_files("src", "scripts", "configs")),
        "command": get_run_command(),
        "labeling_run": {"model_id": settings["model_id"], "prompt_version": settings["prompt_version"],
                         "prompt_sha256": settings["prompt_sha256"]},
        "input_sha256": hashes, "pool": len(pool), "left_out_of_pool": dict(left_out),
        "items": items, "output_sha256": files,
    }
    write_json_atomic(output / "selection.json", selection, overwrite=False)
    return selection


def check_selection(items: list[dict[str, Any]], labels: dict[str, dict[str, Any]], config: dict[str, Any]) -> None:
    """Refuse a sample that breaks the plan: repeated episodes or questions, unknown episodes,
    quotas not met exactly, a missing label or source, or annotator sizes not met."""
    episode_ids = [it["episode_id"] for it in items]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("The sample repeats an episode.")
    if len({it["question_id"] for it in items}) != len(items):
        raise ValueError("The sample has two episodes of one question.")
    if any(e not in labels for e in episode_ids):
        raise ValueError("The sample names an episode the labeling run does not have.")
    got = Counter((labels[e]["claude_label"], labels[e]["dataset_source"]) for e in episode_ids)
    want = Counter({(label, source): n for label, by_source in config["quotas"].items()
                    for source, n in by_source.items()})
    if got != want:
        raise ValueError(f"Strata counts {dict(got)} differ from the quotas {dict(want)}.")
    if {label for label, _ in got} != set(VALID_LABELS) or len({source for _, source in got}) < 2:
        raise ValueError("The sample must cover all four labels and both sources.")
    if Counter(it["annotator"] for it in items) != Counter(config["annotators"]):
        raise ValueError("Annotator sizes are not met.")


# ---------------------------------------------------------------- scoring the returned files

def lf_sha256(path: Path) -> str:
    """SHA-256 of a file with CRLF endings read as LF: the hash of the copy Git stores, so a file
    re-saved by Excel records the same hash as its committed copy."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _cell(text: str | None) -> str:
    return (text or "").replace("\r\n", "\n")


def read_annotations(path: Path, items: list[dict[str, Any]],
                     episodes: dict[str, dict[str, Any]]) -> dict[str, dict[str, str]]:
    """``{item_id: {"human_label", "notes"}}`` for one returned annotation file.

    Accepts a file re-saved by Excel or Sheets (byte-order mark, CRLF endings, label case or
    surrounding spaces). Refuses other columns; a repeated, missing or extra item; any edit to the
    text shown (question, A1, C1, A2, candidates); and a blank or unknown label.
    """
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ANNOTATION_COLUMNS:
            raise ValueError(f"{path.name}: columns {reader.fieldnames} are not {ANNOTATION_COLUMNS}.")
        rows = index_unique(list(reader), "item_id", path.name)
    expected = {it["item_id"]: it for it in items}
    if set(rows) != set(expected):
        raise ValueError(f"{path.name}: items differ from the packet (missing {sorted(set(expected) - set(rows))}, "
                         f"extra {sorted(set(rows) - set(expected))}).")
    answers, bad = {}, []
    for item_id, row in rows.items():
        shown = blind_row(item_id, episodes[expected[item_id]["episode_id"]])
        if any(_cell(row[c]) != _cell(shown[c]) for c in _SHOWN_COLUMNS):
            raise ValueError(f"{path.name}: {item_id} no longer shows the episode's text.")
        label = (row["human_label"] or "").strip().lower()
        if label not in VALID_LABELS:
            bad.append(f"{item_id}={row['human_label']!r}")
        answers[item_id] = {"human_label": label, "notes": (row["notes"] or "").strip()}
    if bad:
        raise ValueError(f"{path.name}: blank or unknown labels: {', '.join(bad)}.")
    return answers


def agreement_scores(scored: list[dict[str, Any]]) -> dict[str, Any]:
    """Human–Claude agreement over the scored items, with the human label as the reference.

    Precision for a label is the share of Claude's uses of it that a person confirmed; recall is the
    share of the person's uses that Claude matched.
    """
    pairs = [(r["human_label"], r["claude_label"]) for r in scored]
    matrix = confusion(pairs)  # rows: the human label; columns: Claude's
    agree = sum(human == claude for human, claude in pairs)
    return {
        "n": len(pairs), "agree": agree, "agreement": agree / len(pairs) if pairs else None,
        "ci95": wilson(agree, len(pairs)), "cohen_kappa": cohen_kappa(matrix), "per_class": per_class(matrix),
        "confusion_human_rows_claude_columns": {h: {c: matrix[h][c] for c in VALID_LABELS} for h in VALID_LABELS},
    }


def disagreement_row(row: dict[str, Any], episode: dict[str, Any]) -> dict[str, str]:
    """One disagreement with the text shown, both labels and Claude's reason."""
    shown = blind_row(row["item_id"], episode)
    return {"item_id": row["item_id"], "annotator": row["annotator"], "dataset_source": row["dataset_source"],
            "episode_id": row["episode_id"], **{c: shown[c] for c in _SHOWN_COLUMNS[1:]},
            "claude_label": row["claude_label"], "claude_reason": row["claude_reason"] or "",
            "human_label": row["human_label"], "human_notes": row["notes"]}


def _pct(rate: float | None) -> str:
    return "n/a" if rate is None else f"{100 * rate:.1f}%"


def report_text(record: dict[str, Any], disagreements: list[dict[str, str]]) -> str:
    """The markdown report for one scoring run."""
    s, run = record["scores"], record["labeling_run"]
    low, high = s["ci95"]
    lines = [
        "# Human audit of the Claude labels (#237): results", "",
        f"{s['n']} {record['partition']} episodes, balanced across Claude's four labels and the two sources, each "
        f"labeled blind by one person with the rules Claude was given. Claude labels: `{run['model_id']}`, "
        f"prompt `{run['prompt_version']}`.", "",
        "| | Value |", "|---|---|",
        f"| Human–Claude agreement | {s['agree']}/{s['n']} = {_pct(s['agreement'])} (95% CI {_pct(low)}–{_pct(high)}) |",
        f"| Cohen's kappa | {s['cohen_kappa']:.3f} |",
        "", "## Confusion matrix (rows: human label, columns: Claude label)", "",
        "| Human \\ Claude | " + " | ".join(VALID_LABELS) + " | Total |",
        "|---|" + "---|" * (len(VALID_LABELS) + 1),
    ]
    for human, row in s["confusion_human_rows_claude_columns"].items():
        lines.append(f"| {human} | " + " | ".join(str(row[c]) for c in VALID_LABELS) + f" | {sum(row.values())} |")
    lines += ["", "## Precision, recall and F1 by class (human label as reference)", "",
              "| Label | Precision | Recall | F1 | Claude used it | Human used it |", "|---|---|---|---|---|---|"]
    for label, m in s["per_class"].items():
        f1 = "n/a" if m["f1"] is None else f"{m['f1']:.3f}"
        lines.append(f"| {label} | {_pct(m['precision'])} | {_pct(m['recall'])} | {f1} | {m['predicted']} | {m['support']} |")
    lines += ["", f"## Disagreements ({len(disagreements)})", ""]
    if disagreements:
        lines += ["| Item | Source | Claude | Human | Annotator's note |", "|---|---|---|---|---|"]
        lines += [f"| {d['item_id']} | {d['dataset_source']} | {d['claude_label']} | {d['human_label']} | "
                  f"{d['human_notes'].replace('|', '/')} |" for d in disagreements]
        lines += ["", "Full text and Claude's reasons: `disagreements.csv`."]
    else:
        lines.append("None.")
    lines += [
        "", "## Limits", "",
        "- The sample is balanced by Claude's label and source, so these figures describe the audit sample, "
        "not Claude's accuracy across the whole dataset.",
        "- Each example was labeled by one person, so this measures human–Claude agreement, not agreement "
        "between annotators.",
        "- Left out before sampling: "
        + (", ".join(f"{n} × `{reason}`" for reason, n in record["left_out_of_pool"].items()) or "nothing") + ".",
    ]
    return "\n".join(lines) + "\n"


def score_audit(config: dict[str, Any], root: Path = Path("."), output: Path | None = None) -> dict[str, Any]:
    """Check the returned annotation files against the packet, score them against Claude's labels and
    write ``scores.json``, ``disagreements.csv`` and ``report.md`` to ``output`` (default
    ``<packet>/scoring``); returns the scores record.

    Refuses: an existing output folder; a selection that does not match the config; a labeling run or
    episodes file that changed since the packet was built; a returned file that fails
    ``read_annotations``; and any input that changes while scoring.
    """
    packet = root / config["output"]
    output = packet / "scoring" if output is None else output
    if output.exists():
        raise FileExistsError(f"{output} exists; a scoring run is never overwritten.")
    selection_path = packet / "selection.json"
    run_dir = root / config["claude_run"]
    raw_path, settings_path = run_dir / "raw_results.jsonl", run_dir / "run_settings.json"
    interactions_path = root / config["interactions"]
    annotation_paths = {name: packet / "annotations" / f"{name}.csv" for name in config["annotators"]}
    checks = [(p, sha256_file) for p in (selection_path, raw_path, settings_path, interactions_path)]
    checks += [(p, lf_sha256) for p in annotation_paths.values()]
    hashes = {str(p.relative_to(root)): digest(p) for p, digest in checks}  # before reading

    selection = json.loads(selection_path.read_text())
    if [selection[k] for k in ("partition", "seed", "quotas", "annotators")] != \
            [config[k] for k in ("partition", "seed", "quotas", "annotators")]:
        raise ValueError(f"{selection_path} was built from a different config.")
    for name, digest in selection["input_sha256"].items():
        if hashes.get(name) != digest:
            raise ValueError(f"{name} changed since the packet was built.")

    labels = index_unique(read_jsonl(raw_path), "episode_id", str(raw_path))
    episodes = index_unique(read_jsonl(interactions_path), "episode_id", str(interactions_path))
    pool, left_out = audit_pool(labels)
    if len(pool) != selection["pool"]:
        raise ValueError(f"The audited pool has {len(pool)} labels; the packet recorded {selection['pool']}.")
    items = selection["items"]
    index_unique(items, "item_id", str(selection_path))
    check_selection(items, labels, config)

    scored = []
    for name, path in annotation_paths.items():
        mine = [it for it in items if it["annotator"] == name]
        answers = read_annotations(path, mine, episodes)
        for it in mine:
            label = labels[it["episode_id"]]
            scored.append({**it, "claude_label": label["claude_label"], "claude_reason": label["claude_reason"],
                           **answers[it["item_id"]]})
    scored.sort(key=lambda r: r["item_id"])
    scores = agreement_scores(scored)
    disagreements = [disagreement_row(r, episodes[r["episode_id"]]) for r in scored
                     if r["claude_label"] != r["human_label"]]

    for path, digest in checks:
        if digest(path) != hashes[str(path.relative_to(root))]:
            raise ValueError(f"{path} changed while scoring.")

    record = {
        "issue": 237, "partition": config["partition"], "git_commit": get_git_commit(),
        "git_dirty": bool(get_git_diff() or get_untracked_files("src", "scripts", "configs")),
        "command": get_run_command(), "labeling_run": selection["labeling_run"],
        "input_sha256": hashes, "annotation_hash_note": "annotation files are hashed with CRLF read as LF",
        "pool": len(pool), "left_out_of_pool": dict(left_out), "scores": scores,
        "items": [{k: r[k] for k in ("item_id", "episode_id", "question_id", "dataset_source", "annotator",
                                     "claude_label", "human_label")} for r in scored],
    }
    output.mkdir(parents=True)
    files = {}
    if disagreements:
        write_csv(output / "disagreements.csv", disagreements, lineterminator="\n")
        files["disagreements.csv"] = sha256_file(output / "disagreements.csv")
    report = output / "report.md"
    report.write_text(report_text(record, disagreements), encoding="utf-8")
    files["report.md"] = sha256_file(report)
    record["output_sha256"] = files
    write_json_atomic(output / "scores.json", record, overwrite=False)
    return record
