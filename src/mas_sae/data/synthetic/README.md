# Synthetic MuSiQue starter

This package owns the answerable candidate contract, structural validation and
normalized exact question deduplication. Generation, factual checking, semantic
novelty and reasoning-quality evaluation remain separate responsibilities.
No model API calls are made here.

## Generator contract

Produce a JSON object matching `SyntheticCandidate` in `contracts.py`, including
`question_decomposition`. The existing Claude example script omits this field and
will need to be adapted by the generation owner.

- IDs: one of `2hop`, `3hop1`, `3hop2`, `4hop1`, `4hop2`, `4hop3`, followed by
  `__` and a unique suffix (for example, a UUID). The generator owns ID uniqueness;
  the question index does not enforce record-ID uniqueness.
- Paragraphs: normally 20, with sequential zero-based indices equal to list
  positions. The local original records contain 16–20 paragraphs, so the default
  permits up to 20; use `expected_paragraphs=20` to require exactly 20.
- Decomposition: one step per hop, unique integer step IDs, and supporting paragraph
  indices. `#1` means the answer of the first step, not the step's `id`. References
  must point backwards. Every intermediate step must feed a later step, and every
  supporting paragraph must be used. Branches are supported.
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
against the files you imported and prior registrations. It does not establish
factual correctness, necessary reasoning hops, semantic uniqueness, distributional
similarity, or the exact dependency topology implied by hop subtypes. These need
additional evaluation before including examples in a research dataset.

The validator was checked against the local train/validation files. Two training
records contain non-backward decomposition references and are intentionally
flagged rather than treated as valid examples to imitate.

Run focused tests:

```bash
python -m pytest src/tests/unit/test_synthetic_musique.py -q
```
