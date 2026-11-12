"""Human audit of the Claude behavior labels (#237): a blind annotation packet for a seeded,
stratified sample of one labeled partition.

Each sampled episode is shown to one annotator exactly as Claude saw it: the question, A1, the
Critic's recommended answer (C1), A2, and which text the Solver saw as Candidate A / Candidate B.
Claude's label and reason, the dataset source and the episode id are never written to the
annotation files; ``selection.json`` maps item ids back to episode ids for scoring, which joins
Claude's labels from the committed labeling run.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import yaml

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
    # Only Claude's own labels are audited: technical failures (no label) and rule-assigned labels
    # are left out, each checked on its own row.
    pool = [r for r in labels.values() if r["label_source"] == "claude" and r["claude_label"] in VALID_LABELS]
    left_out = Counter(f"{r['label_source']}:{r['exclusion_reason']}" for r in labels.values()
                       if not (r["label_source"] == "claude" and r["claude_label"] in VALID_LABELS))

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
