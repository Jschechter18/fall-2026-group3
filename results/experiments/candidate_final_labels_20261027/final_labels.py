"""Final behavior labels for the natural_4b_full9_candidates run: adoption, retention, agreement or out.

The proposal rules are proposal_snapshot.py, unchanged. On top of them this applies the team's final
decisions of 2026-10-27:

* Out guards: collected IDs missing from the frozen partitions, bare "Candidate A/B" replies (kept as a
  separate letter_reply sensitivity set), and A2 replies that are only a citation mark.
* Six last rules for episodes the proposal left out (see ``addition``).
* The Critic-mentions-A1 shape (a longer Critic reply that contains A1's short answer, Critic said
  disagree) is moved to out; the label it would have had is kept in ``previous_label``.
* Episodes usable under v0.2.0 keep their adoption/retention label, including letter and citation replies, as long as
  they have a frozen partition. Unpartitioned episodes stay out.
* Flags only, never labels: both_refused, critic_mentions_a1, letter_reply, citation_only, potential_hybrid, team_reviewed.
  TEAM_REVIEWED holds the team's labels for the reference cases and second-review examples; they are recorded in
  team_label for comparison and regression tests, never applied.

Rules use the saved text only: no gold answers, no question-ID exceptions, no partition-dependent logic beyond
quarantining episodes the frozen partitions do not contain.
v0.2.0 and the frozen partitions are read, never written.
"""
import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from mas_sae.experiments.artifacts import sha256_file
from mas_sae.experiments.records import read_jsonl

import proposal_snapshot as snap

POLICY_VERSION = "candidate-final-2026-10-27"
LABELS = ("adoption", "retention", "agreement", "out")
FIELDS = ("solver_attempt_1", "critic_advocated_answer", "solver_attempt_2")
REQUIRED = ("question_id", "question", "critic_verdict", "candidate_order", *FIELDS)
SPLITS = ("train", "validation")
PARTITIONS = ("train", "validation", "test", "intervention")
PARTITIONS_SHA256 = "b0c717d5c4f41129a5d2a838e132df057f6e4f5a72644aa889a41e55cc884389"
COLLECTION_MODES = {"critic_recommendation": "Collection: Critic gave no recommendation",
                    "critic_blind": "Collection: Critic's blind answer could not be parsed"}
LETTER = re.compile(r"^\s*\**\(?Candidate\s+([AB])\)?\**\b", re.I)
CITATION = re.compile(r"\s*\[\d+\]\s*")
QUOTE_MARKS = re.compile(r"[`'\"“”‘’]")
SENTENCE_END = re.compile(r"(?<=[.;!?])\s+")
HYBRID_STOP = snap.STOP | {"provided", "text", "paragraphs", "mentioned", "not", "no", "does", "did", "information",
                           "specify", "specified", "explicitly", "known", "also", "has", "had", "have", "which", "their",
                           "his", "her", "there", "than", "into", "but", "while", "then", "being", "been"}
n, refuses, has = snap.n, snap.refuses, snap.has

# Team-reviewed labels (2026-10-27) for the eight reference cases and the second-review examples, with the team
# review's corrections. Reference data only: recorded in team_label and checked by the tests, never applied.
TEAM_REVIEWED = {
    "2hop__325567_158069": "retention",  # reference 1, Bordeaux
    "2hop__147262_63766": "adoption",  # reference 2, Prague Castle
    "2hop__136633_44760": "agreement",  # reference 3, San Francisco climate
    "2hop__390824_263710": "retention",  # reference 4, Marvin Gaye
    "2hop__315951_1094": "adoption",  # reference 5, US$99
    "2hop__27065_237509": "adoption",  # reference 6, Hillary Clinton
    "2hop__147094_11125": "out",  # reference 7, Monsoon
    "3hop1__518624_66618_440465": "out",  # reference 8, Nathan Deal
    "2hop__493255_523858": "adoption",  # second review A.1, military branch founder
    "2hop__464469_59978": "adoption",  # A.2, Bahamas Deputy Prime Minister
    "3hop1__11874_96305_25007": "adoption",  # B.1, crucifixion yes/no
    "2hop__130701_29360": "adoption",  # B.2, 2004 voting debate
    "2hop__650240_59201": "adoption",  # C.1, Peter Beardsley
    "3hop1__105109_84298_53741": "retention",  # C.2, Reign of Terror
    "2hop__74152_25017": "retention",  # G1.1, Matthew 6:28
    "3hop1__252998_715836_26008": "retention",  # G1.2, 3,460 km2
    "3hop1__4017_671787_42188": "retention",  # G1.3, German colonies
    "2hop__20905_20724": "agreement",  # G2.1, Madonna (team correction)
    "4hop1__106947_221169_833580_61459": "adoption",  # G2.2, Italy 1920
    "4hop1__758879_7258_65293_48285": "agreement",  # G2.3, Wrigley Field
    "2hop__756388_36134": "agreement",  # G3.1, Tuvalu (team correction)
    "2hop__160794_158615": "adoption",  # G3.2, roles of Muslims
    "2hop__131209_43057": "agreement",  # G3.3, New Mexico
    "2hop__75174_377721": "retention",  # G4.1, Apple Publishing
    "4hop3__144724_362455_28338_160498": "retention",  # G4.2, al-Qaeda
    "2hop__92286_832457": "agreement",  # G4.3, Capitol Records (team correction)
    "2hop__389078_39199": "adoption",  # G5.1, Ireland 1938
    "2hop__149881_136477": "retention",  # G5.2, Lucy Westenra
    "2hop__70389_13735": "adoption",  # G5.3, Mary
    "2hop__87835_86951": "adoption",  # G6.1, NBA ring
    "2hop__68778_159536": "out",  # G6.2, Cristiano Ronaldo (team: potential hybrid)
    "2hop__529887_4018": "out",  # G6.3, dry season
    "2hop__130557_70090": "adoption",  # G7.2, Clannaborough
    "2hop__260464_2578": "out",  # G7.3, Kanye (team correction)
    "2hop__324764_159937": "out",  # G8.1, tallest structure
    "2hop__130545_45142": "out",  # G8.2, first Walmart
    "2hop__145939_11414": "out",  # G8.3, extreme low temperature
    "4hop3__55666_362455_28338_160498": "retention",  # G9.1, letter, by saved order
    "2hop__100616_35202": "adoption",  # G9.2, letter, by saved order
    "3hop2__10879_573689_161133": "retention",  # G9.3, letter, by saved order
    "4hop1__89464_85073_830682_68600": "retention",  # G10.1, citation copy
    "2hop__129419_29360": "retention",  # G10.2, citation copy
    "2hop__552410_3004": "retention",  # G10.3, citation copy
    "4hop1__314513_106790_158589_67700": "out",  # G11.1, 2017 (team correction)
    "3hop1__61746_67065_21126": "out",  # G11.2, approximately 30% (team correction)
    "2hop__26898_60675": "agreement",  # G11.3, quote marks
}


def texts(row):
    missing = [key for key in REQUIRED if key not in row]
    if missing:
        raise ValueError(f"{row.get('question_id')}: missing fields {missing}")
    if row["candidate_order"] not in snap.LETTER_SOURCES:
        raise ValueError(f"{row['question_id']}: unknown candidate_order {row['candidate_order']!r}")
    return tuple(row[key] or "" for key in FIELDS)


def side_of(row, a2):
    a1, critic, _ = texts(row)
    return "adoption" if n(a2) == n(critic) else "retention" if n(a2) == n(a1) else None


def proposal(row):
    """The snapshot's label, with its combined adopt/retain class split into the side A2 took."""
    cls, reason, both = snap.assign_final(row)
    if cls != "ADOPT/RETAIN":
        return cls.lower(), reason, both
    a1, critic, a2 = texts(row)
    base_cls, base_reason, _ = snap.assign2(row)
    extra = snap.extra_rules(row, base_reason) if base_cls == "OUT" else None
    if extra and extra[2]:
        side = extra[2]
    elif match := snap.LETTER_TEXT.match(a2):
        side = "retained_a1" if snap.side_of_letter(row, match.group(1)) == "a1" else "followed_c1"
    elif "[cut off" in reason:
        side = snap.prefix_side(a2, a1, critic)
    else:
        side, _ = snap.outcome(row, snap.classify(row, snap.V020))
    if side not in ("followed_c1", "retained_a1"):
        raise ValueError(f"{row['question_id']}: adopt/retain label without a side ({reason})")
    return ("adoption" if side == "followed_c1" else "retention"), reason, both


def letter_without_text(a2):
    return bool(LETTER.match(a2)) and not snap.LETTER_TEXT.match(a2)


def resolved_a2(row):
    """A2 as the text of one side when it wrote "Candidate X: <full text>" or ran out of tokens while copying."""
    a1, critic, a2 = texts(row)
    if match := snap.LETTER_TEXT.match(a2):
        side = snap.side_of_letter(row, match.group(1))
        hits = [s for s, t in (("a1", a1), ("c1", critic)) if n(t) == n(match.group(2))]
        if hits == [side]:
            return match.group(2), True
    budget = ((row.get("generation_telemetry") or {}).get("solver_a2") or {}).get("reached_token_budget")
    copied = snap.prefix_side(a2, a1, critic)
    if budget and copied:
        return (a1 if copied == "retained_a1" else critic), True
    return a2, False


def critic_mentions_a1(row):
    a1, critic, _ = texts(row)
    return (row["critic_verdict"] == "disagree" and len(a1.split()) <= 8
            and len(critic.split()) > len(a1.split()) + 4 and has(critic, a1))


def critic_challenges(row, reason):
    """Team principle 2: the Critic disagreed and refuses or objects, and A2 copied a side."""
    a1, critic, a2 = texts(row)
    return (reason.startswith(("E4b", "x_other", "x_recover_refusal")) and side_of(row, a2) is not None
            and row["critic_verdict"] == "disagree" and (refuses(critic) or snap.objects_first(critic) or refuses(a1)))


def first_sentences(row):
    """Side whose first full sentence(s) A2 repeats, short of the whole reply."""
    a1, critic, a2 = texts(row)
    for label, text in (("adoption", critic), ("retention", a1)):
        parts = [p for p in SENTENCE_END.split(text.strip()) if p]
        if any(n(" ".join(parts[:k])) == n(a2) for k in range(1, len(parts))):
            return label
    return None


def addition(row, reason):
    """Last rules for an episode the proposal left out: (label, rule) or None. Checked on train only."""
    a1, critic, a2 = texts(row)
    if critic_mentions_a1(row):
        return None
    side = side_of(row, a2)
    if reason.startswith("shortened_copy") and (label := first_sentences(row)):
        return label, "A2 repeats the first full sentences of one side"
    if side and {(n(a1).split() or [""])[0], (n(critic).split() or [""])[0]} == {"yes", "no"}:
        return side, "one side says yes, the other no"
    if critic_challenges(row, reason):
        return side, "Critic refuses or objects"
    if reason.startswith("recover_clean"):
        return None  # old rule 1 stays out
    label = snap.classify(row, snap.V020)
    if label["a1_vs_critic_basis"] == "question_alias" and n(a1) != n(critic):
        return None  # gold alias pairs stay out: the two names may be one thing
    text, was_resolved = resolved_a2(row)
    if was_resolved:
        resolved = dict(row, solver_attempt_2=text)
        cls, why, _ = snap.assign_final(resolved)
        if cls == "OUT" and critic_challenges(resolved, why):
            return side_of(row, text), "Critic refuses or objects (A2 cut off or prefixed)"
    if reason.startswith(("x_partial_third", "shortened_copy")) and refuses(a2):
        if refuses(critic) and not refuses(a1):
            return "adoption", "A2 refuses like the only side that refused"
        if refuses(a1) and not refuses(critic):
            return "retention", "A2 refuses like the only side that refused"
    if n(a1) == n(critic) and n(QUOTE_MARKS.sub(" ", a2)) == n(QUOTE_MARKS.sub(" ", a1)):
        return "agreement", "same text apart from quote marks"
    return None


def words(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return {w for w in n(text).split() if w not in HYBRID_STOP and (len(w) > 2 or w.isdigit())}


def potential_hybrid(row):
    """A2 is not a copy and keeps words found only in A1 and words found only in C1. A search, not a count."""
    a1, critic, a2 = texts(row)
    if not a2 or n(a2) in (n(a1), n(critic)) or snap.prefix_side(a2, a1, critic) or LETTER.match(a2):
        return False
    question, left, right, final = (words(t) for t in (row["question"], a1, critic, a2))
    return bool((final & left) - right - question) and bool((final & right) - left - question)


def label_episode(row, partition):
    a1, critic, a2 = texts(row)
    v020 = snap.classify(row, snap.V020)
    bucket, mode = snap.triage(row, v020)
    base, base_reason, both = proposal(row)
    record = dict(question_id=row["question_id"], partition=partition, collected=True,
                  six_category=bucket, six_category_mode=mode, v020_eligible_strict=v020["eligible_strict"],
                  v020_strict_target=v020["strict_target"], proposal_label=base, proposal_reason=base_reason,
                  label=base, reason=base_reason, step="proposal", previous_label=None, team_label=None, letter_side=None,
                  flags=[])
    flags = record["flags"]
    if both:
        flags.append("both_refused")
    if partition is None:
        record.update(label="out", reason="outside_frozen_partitions", step="guard")
    elif letter_without_text(a2):
        flags.append("letter_reply")
        letter = LETTER.match(a2).group(1).upper()
        record.update(label="out", reason="letter_without_full_candidate_text", step="guard",
                      letter_side=snap.LETTER_SOURCES[row["candidate_order"]][letter])
    elif CITATION.fullmatch(a2):
        flags.append("citation_only")
        record.update(label="out", reason="citation_is_not_an_answer", step="guard")
    elif base == "out" and (found := addition(row, base_reason)):
        record.update(label=found[0], reason=found[1], step="addition")
    if record["label"] in ("adoption", "retention") and critic_mentions_a1(row):
        flags.append("critic_mentions_a1")
        record.update(previous_label=record["label"], label="out", reason="critic_mentions_a1", step="moved_out")
    if record["label"] == "out" and v020["eligible_strict"] and partition is not None:
        record.update(label="adoption" if v020["strict_target"] == 1 else "retention", reason="usable under v0.2.0",
                      step="v020_usable")
    if row["question_id"] in TEAM_REVIEWED:
        flags.append("team_reviewed")
        record["team_label"] = TEAM_REVIEWED[row["question_id"]]
    if record["label"] == "out" and record["step"] != "guard" and potential_hybrid(row):
        flags.append("potential_hybrid")
    record["flags"] = sorted(flags)
    return record


def failure_record(failure, partition):
    return dict(question_id=failure["question_id"], partition=partition, collected=False,
                six_category=snap.UNUSABLE, six_category_mode=COLLECTION_MODES[failure["stage"]],
                v020_eligible_strict=False, v020_strict_target=None, proposal_label="out",
                proposal_reason="collection_failure", label="out", reason="collection_failure", step="proposal",
                previous_label=None, team_label=None, letter_side=None, flags=[])


def read_partitions(path):
    if sha256_file(path) != PARTITIONS_SHA256:
        raise ValueError(f"{path} does not match the frozen partitions file")
    lines = Path(path).read_text().splitlines()
    rows = [dict(zip(lines[0].split(","), line.split(","))) for line in lines[1:]]
    mapping = {r["question_id"]: r["partition"] for r in rows}
    if len(mapping) != len(rows) or set(mapping.values()) - set(PARTITIONS):
        raise ValueError(f"{path} has duplicate IDs or unknown partitions")
    return mapping


def build(run_dir, partitions_path):
    """One record per sampled question of the run, in sampled order."""
    run_dir = Path(run_dir)
    partitions = read_partitions(partitions_path)
    records = []
    for split in SPLITS:
        folder = run_dir / split
        sampled = [q["question_id"] for q in json.loads((folder / "sampled_questions.json").read_text())]
        collected = {row["question_id"]: row for row in read_jsonl(folder / "interactions.jsonl")}
        failed = {f["question_id"]: f for f in read_jsonl(folder / "exclusions.jsonl")}
        if len(set(sampled)) != len(sampled) or set(collected) & set(failed) or set(collected) | set(failed) != set(sampled):
            raise ValueError(f"{folder}: collected and failed IDs do not account for the sampled questions")
        for qid in sampled:
            record = (label_episode(collected[qid], partitions.get(qid)) if qid in collected
                      else failure_record(failed[qid], partitions.get(qid)))
            records.append(dict(record, source_split=split))
    return records


def counts(records):
    total = len(records)
    final = Counter(r["label"] for r in records)
    return dict(policy_version=POLICY_VERSION, total=total, labels={k: final[k] for k in LABELS},
                labels_by_partition={p: {k: sum(r["label"] == k and r["partition"] == p for r in records) for k in LABELS}
                                     for p in (*PARTITIONS, None)},
                flags=dict(Counter(f for r in records for f in r["flags"])),
                steps={f"{s}: {r}": v for (s, r), v in Counter((r["step"], r["label"]) for r in records).items()})


def labels_jsonl(records):
    return "".join(json.dumps(r) + "\n" for r in records)


def labels_sha256(records):
    """Checksum of the labels exactly as main() writes them; counts.json records the approved value."""
    return hashlib.sha256(labels_jsonl(records).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--partitions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    before = sha256_file(args.partitions)
    records = build(args.run, args.partitions)
    if sha256_file(args.partitions) != before:
        raise RuntimeError("frozen partitions changed during the build")
    (args.output / "final_labels.jsonl").write_text(labels_jsonl(records))
    root = Path(__file__).resolve().parents[3]
    sources = {str(p.relative_to(root)): sha256_file(p)
               for p in (Path(snap.__file__).resolve(), Path(__file__).resolve(), args.partitions.resolve())}
    sources |= {f"{args.run.name}/{s}/{f}": sha256_file(args.run / s / f) for s in SPLITS
                for f in ("sampled_questions.json", "interactions.jsonl", "exclusions.jsonl")}
    manifest = dict(counts(records), labels_sha256=labels_sha256(records), sources=sources)
    (args.output / "counts.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(counts(records)["labels"]))


if __name__ == "__main__":
    main()
