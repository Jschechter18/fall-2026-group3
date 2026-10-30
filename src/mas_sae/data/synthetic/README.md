# Synthetic MuSiQue starter

This package owns the answerable candidate contract, structural validation and
normalized exact question deduplication. Generation, factual checking, semantic
novelty and reasoning-quality evaluation remain separate responsibilities.
No model API calls are made here.

## Generator contract

Produce a JSON object matching `SyntheticCandidate` in `contracts.py`, including
`question_decomposition`. `generation.py` (below) does this; the older
`examples/claude/claude_for_data_gen.py` omits this field.

- IDs: one of `2hop`, `3hop1`, `3hop2`, `4hop1`, `4hop2`, `4hop3`, followed by
  `__` and a unique suffix (for example, a UUID). The generator owns ID uniqueness;
  the question index does not enforce record-ID uniqueness.
- Paragraphs: normally 20, with sequential zero-based indices equal to list
  positions. The local original records contain 16–20 paragraphs, so the default
  permits up to 20; use `expected_paragraphs=20` to require exactly 20.
- Decomposition: one step per hop, unique integer step IDs, and supporting paragraph
  indices. `#1` means the answer of the first step, not the step's `id`. References
  must point backwards. Every intermediate step must feed a later step, and every
  supporting paragraph must be used. References must match the subtype dependency
  shape defined in `hop_types.py`, in generation and standalone validation alike.
- Final step answer must equal the record answer. `answer_aliases` is a list;
  `answerable` must be true. Unanswerable examples are outside this starter scope.
- Store model/prompt versions, raw responses, sources and other provenance
  separately, keyed by record ID. Extra fields are tolerated, not validated.

`src/tests/fixtures/synthetic_musique/valid.json` is an annotated-by-field example
with **fictional facts for testing only**, not suitable for dataset inclusion.

## Integrating a generator

```python
from dataclasses import asdict
from mas_sae.data.synthetic import DuplicateIndex, validate_candidate

with DuplicateIndex("data/MuSiQue/synthetic/question_index.sqlite") as index:
    # Do this before generation, including all original splits available to you.
    index.import_jsonl("data/MuSiQue/clean/train.json", source="original:train")
    index.import_jsonl("data/MuSiQue/clean/validation.json", source="original:validation")

    result = validate_candidate(candidate)
    if not result.valid:
        rejection = [asdict(issue) for issue in result.errors]
    else:
        match = index.register_question(candidate["id"], candidate["question"], source="synthetic:run-1")
        if match is not None:
            rejection = asdict(match)
        else:
            # Persist candidate and provenance here.
            pass
```

Registration returns `None` for a new question or a `DuplicateMatch` for an existing
one. It checks and inserts atomically; do not rely on `find_duplicate()` alone as
an acceptance gate. Use the same SQLite file and separate connections for workers
on one machine. Distributed execution needs a shared database design.

Normalization applies Unicode NFKC, case folding and whitespace collapse.
Punctuation is retained. Rewordings and punctuation-only variations can therefore
pass: this is normalized exact matching, not semantic novelty detection. Importing
originals is idempotent; the first stored match supplies the reported ID/source.
A malformed import rolls back that file's additions.

## Generating a batch

`generation.py` asks Claude (via `mas_sae.models.claude.ClaudeClient` on Bedrock)
for one record per call: question, answer, decomposition, one supporting paragraph
per hop and three hard distractors. Each prompt fixes the step dependency shape of
the hop type (for example `4hop3`: 1 → 2, 3 independent, 2 + 3 → 4). The remaining
paragraphs up to 20 come from original MuSiQue records (`--original`), skipping
titles already used and paragraphs that mention the answer. Code, not the model,
shuffles paragraphs and assigns `idx`, `is_supporting`, step IDs and the record ID.
Each response is pre-checked with `validate_candidate`; a failed parse, truncation
or pre-check triggers one repair retry containing the rejected raw response and
specific failure details. Records that still fail are not written. Prompt version
`v3` lets the model choose its subject and treats the generated documents as ground
truth; real-world factual accuracy is not required. Internal consistency and
necessary reasoning hops remain requirements. Each subtype uses a matching compact
example from `prompt_examples.py`. Structural checks cannot establish semantic
quality or diversity; manually review a generated batch before dataset inclusion.

```bash
PYTHONPATH=src python scripts/generate_synthetic_musique.py --n 6 --seed 0
```

This makes paid API calls (one per record, two on retry) using the credentials in
`.env`. Hop types cycle round-robin (`--hop-types` to restrict). Outputs go to
`data/MuSiQue/synthetic/runs/<run-id>/generation/`: `candidates.jsonl`,
`provenance.jsonl` (prompt version, raw responses, tokens and errors, keyed
by record ID) and `summary.json`. Invalid raw `support` values (including floats,
booleans, and out-of-range integers) are rejected before paragraph-index remapping.
API and unexpected runtime errors stop the command with a nonzero exit status;
completed attempts and known token usage are retained. The summary distinguishes
`api_calls` (client invocations, including failed calls) from `completed_responses`
and includes `status` and `fatal_error`. Token totals include only returned usage;
SDK-internal retries are not separately counted. Exhausting candidate-validation
retries is an ordinary record failure and does not abort the batch.
Then check the batch as below, passing
`--output-dir` as the sibling `validation/` directory.

Filler paragraphs are real text while supporting paragraphs may be invented, so a
filler can occasionally conflict with an invented fact; check this in manual review.

## Checking a saved batch

From the repository root with the project environment activated:

```bash
PYTHONPATH=src python scripts/validate_synthetic_musique.py candidates.jsonl \
  --original data/MuSiQue/clean/train.json data/MuSiQue/clean/validation.json \
  --index data/MuSiQue/synthetic/question_index.sqlite
```

By default, each invocation creates
`data/MuSiQue/synthetic/runs/<UTC-timestamp>_<uuid>/validation/`, relative to the
working directory. The printed summary includes the resolved output path.
Use `--output-dir PATH` to override it; that directory must be new.
Keep the same `--index` across runs to detect previously accepted questions.
Outputs are `accepted.jsonl`, `rejected.jsonl`
(with input line, candidate/raw text, errors and duplicate matches), and
`summary.json`. Counts by rejection code count records with that code, so their
sum may exceed rejected records. Exit status is zero for a completed batch even
when candidates are rejected; setup/read/write failures fail the command.

The command **registers** accepted questions in the persistent index. Reusing the
index rejects them in future batches. The index registration and output file write
are not a single transaction: if interrupted between them, a registered candidate
may be absent from the accepted file. Preserve the incoming JSONL for recovery.
Before unattended production use, integrate acceptance with durable candidate
storage in one transaction. This starter is a batch checking and integration tool.

## What passing means

Passing establishes structural validity and normalized exact question uniqueness
against the files you imported and prior registrations, plus the decomposition
reference topology implied by the hop subtype. It does not establish factual
correctness, necessary reasoning hops, semantic uniqueness, or distributional
similarity. These need additional evaluation before dataset inclusion.

The validator was checked against the local train/validation files. Two training
records contain the literal song title `#9 Dream`. The reference parser interprets
`#9` as a non-backward step reference and flags those records; distinguishing
literal titles from reference notation remains a separate parser limitation.

Run focused tests:

```bash
python -m pytest src/tests/unit/test_synthetic_musique.py -q
```
