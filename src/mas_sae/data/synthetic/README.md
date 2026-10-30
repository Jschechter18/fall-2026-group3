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

This makes paid API calls using `.env` credentials. Opus 5 is the default;
`--model-id` overrides it. `--n` retains its meaning: a fixed number of generation
requests, each with up to two model invocations including a repair. Default
concurrency is 1. `--concurrency 16` allows up to 16 requests to run simultaneously
on one machine. Hop types are scheduled round-robin (`--hop-types` to restrict).

Both the CLI and review notebook use `batch.run_batch`. Each worker reuses its own
client and receives an independent RNG derived from the run seed and request ID.
Only the coordinator touches the duplicate index, counters, and output files.
Results are written as they complete. Request IDs join records to validation and
provenance, irrespective of file order. Seeds are recorded, but model outputs and
which concurrent duplicate wins are not guaranteed reproducible.

The full-run settings are saved in `configs/synthetic/opus5_v3_full.yaml`:

```bash
PYTHONPATH=src python scripts/generate_synthetic_musique.py \
  --config configs/synthetic/opus5_v3_full.yaml
```

This targets 80,004 accepted questions at concurrency 256. It makes paid calls when
executed; creating or reading the config does not launch a run. All config paths
are relative to the working directory; launch from the repository root. YAML keys
use the CLI names with underscores (for example, `target_per_hop`). Unknown keys
and invalid types are rejected. Explicit CLI flags override config values:

```bash
PYTHONPATH=src python scripts/generate_synthetic_musique.py \
  --config configs/synthetic/opus5_v3_full.yaml \
  --target-per-hop 5 --max-requests-per-hop 10 --concurrency 16 \
  --output-dir data/MuSiQue/synthetic/runs/config-smoke-001/generation
```

`--n` switches a quota config to fixed-request mode, clearing its quota and cap.
`--target-per-hop` switches a fixed-request config to quota mode; supply the request
cap as well. The output directory must be new. Saved config support does not add
resume; complete resume support before relying on recovery of a full-size run.
Resolved generation and transport settings are recorded in the run summary.

For five accepted questions per subtype, with at most ten requests per subtype:

```bash
PYTHONPATH=src python scripts/generate_synthetic_musique.py \
  --target-per-hop 5 --max-requests-per-hop 10 --concurrency 16 --seed 42 \
  --dedup-original data/MuSiQue/clean/train.json data/MuSiQue/clean/validation.json
```

Quota mode counts accepted plus in-flight requests before scheduling, avoiding
quota overshoot. Failed/duplicate requests free slots for replacements, within
the cap. `--n` and `--target-per-hop` are mutually exclusive. A completed fixed
request batch exits zero even with record failures. An unmet accepted quota exits
1 with `status=quota_not_met`. Fatal errors propagate and fail either mode.

Every new run writes under `data/MuSiQue/synthetic/runs/<run-id>/generation/`:

- `candidates.jsonl`: structurally pre-checked candidates before deduplication.
- `accepted.jsonl`: candidates passing validation and normalized-exact deduplication.
- `validation.jsonl`: per-request acceptance, rejection, cancellation, or fatal error.
- `provenance.jsonl`: request IDs/seeds, timestamps, raw responses, repairs, usage,
  requested/returned model IDs, and errors. Exhausted requests may have no candidate.
- `question_index.sqlite`: originals explicitly selected for deduplication plus accepted questions.
- `summary.json`: status, counts, quotas, configuration, elapsed time, accepted/minute,
  request-latency p50/p95 (including repairs), terminal API error codes, and SDK retry totals.

`--dedup-original` indexes every supplied original split. In quota mode it defaults
to `--original`; fixed-request mode defaults to within-batch deduplication only.
This index does not automatically cover earlier synthetic runs. Use the standalone
validation command with a shared index for cross-run exact deduplication before
release. Acceptance is not manual approval. Index registration and file writing
are not transactional; forced termination or storage errors may leave partial artifacts.

On an API/programming error or KeyboardInterrupt, scheduling stops, unstarted work
is cancelled, and running requests are allowed to finish and are saved before
clients close and the failure propagates. A running request may finish its bounded
repair attempts. Shutdown can therefore take time. Repeated interruption, process
termination, or disk failure cannot guarantee draining. Existing output directories
are refused; automatic resume is not implemented.

Transport options are `--connect-timeout` (10 seconds), `--read-timeout` (300 seconds),
and `--sdk-max-attempts` (3 total attempts per invocation, standard SDK retry mode).
These are socket timeouts and retry limits, not a hard run deadline. Transport
retries differ from model-output repairs. `api_calls` counts client invocations;
`sdk_retry_attempts` counts retries reported on successful SDK responses. Failed
invocations may have unreported usage/retries. `api_errors_by_code` counts terminal
errors, not throttles recovered internally. No rate limiter or quota increase is
implied by concurrency. See [Botocore configuration](https://docs.aws.amazon.com/botocore/latest/reference/config.html).

To benchmark, run separate fixed-size batches at concurrency 1, 4, 8, and 16 with
new output directories (for example, `--n 48 --concurrency 4 --seed 42`). These are
paid calls; the software does not launch a benchmark automatically. Compare
`accepted_per_minute`, elapsed time, p50/p95 latency, SDK retries and terminal API
errors. More concurrency is useful only while actual throughput improves.

`notebooks/synthetic_data_generation_sample.ipynb` calls the same runner with
`CONCURRENCY`, subtype quotas, and both original splits. Generation is disabled by
default. Existing saved batches/review worksheets are retained; loading and review
cells make no calls. For a prior run, set `REVIEW_RUN_DIR` to its saved path.

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
