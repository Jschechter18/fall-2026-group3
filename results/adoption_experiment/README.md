# Adoption protocol experiment

The current experiment is the eight-row controlled comparison shown in
[presentation_summary/adoption_comparison.svg](presentation_summary/adoption_comparison.svg).
All rows use the same 98 completed question/recommendation trials from 80 questions.

| Directory | Purpose |
| --- | --- |
| sample_100q | Frozen original sample and source metadata |
| controlled_100q | Controlled recommendations and original revision results |
| verification_100q | Evidence-verification comparison |
| anonymization_100q | Labeled and anonymous responses in both fixed orders |
| randomized_order_100q | Frozen question-level orders and labeled randomized run |
| presentation_summary | Current chart, tables and provenance |

The anonymous randomized row selects existing anonymous responses using the exact
same frozen order assignments as the labeled randomized row. No new inference or
resampling is performed. Saved alias-aware scoring is retained in this chart.

Regenerate from the repository root:

```bash
PYTHONPATH=src python scripts/experiments/summarize_adoption_results.py \
  --results-root results/adoption_experiment \
  --output results/adoption_experiment/presentation_summary \
  --replace-output --chart
```

`archive/` contains earlier ablations, Natural runs and supporting artifacts, plus
the superseded presentation export. It is ignored by Git. Historical artifact
contents and provenance were preserved; old paths into moved folders now resolve
under archive/. Shared active inputs remain at their original paths.

## Natural feedback on the same 80 questions

`adoption_results/plan.json` freezes the exact 80 plotted questions,
their original A1 answers (through the source sample hash), and the original
question-level candidate orders. No new questions are sampled or screened out.
51 questions have usable saved natural feedback; the remaining 29 require fresh
natural feedback from the original Qwen critic. The regeneration budget is 1,024
tokens rather than the original 256, to allow previously incomplete responses
to finish. No correct or incorrect targets are supplied to this critic.

Run from the repository root (automatically selects CUDA, MPS, then CPU):

```bash
PYTHONPATH=src python scripts/experiments/compare_natural_protocols.py
```

To select Apple Silicon explicitly, add `--device mps`. MPS and CPU default to
float32; CUDA retains the original model precision. `--dtype` overrides this
explicitly. Runtime device and precision are saved in the plan, alongside the
original solver configuration. Cross-device/precision results may differ even
with deterministic decoding. Both models are loaded sequentially.

After a completed run, redraw only the SVG and PNG without model inference:

```bash
PYTHONPATH=src python scripts/experiments/compare_natural_protocols.py --plot-only
```

Use `--prepare-only` to validate sources and save the plan without inference.
Feedback and six unique solver prompt responses per question are checkpointed;
rerunning the same command resumes them. An unstarted plan can be updated for
another device; once inference is checkpointed, use a separate `--output` for
different settings. Randomized rows select responses using
the original frozen orders (generation is deterministic). If regenerated feedback
has no usable recommendation, the runner retries with a one-sentence explanation,
then with an empty explanation, to avoid truncated JSON. Each failed attempt is
saved. After three unsuccessful attempts it stops instead of dropping the question
or inventing a target. Feedback-only checkpoints can resume after code updates
with unchanged experimental settings; the previous plan is saved to
`plan_history.jsonl`.

On completion, the separate output directory contains `adoption_comparison.svg`
and `.png` with the same eight rows and two-panel structure as the fixed plot,
plus `method_summary.csv`, `feedback.jsonl`, and `episodes.jsonl`. Each row has
80 trials, one natural recommendation per question, instead of the fixed plot's
98 controlled recommendation trials. Original and evidence-verification revision
receive full natural feedback; candidate conditions receive the advocated answer,
as in the original protocols. The right panel groups recommendations by actual
gold/alias correctness; its sample sizes are calculated from the new feedback.
The left panel separates outputs matching both A1 and recommendation, since
natural agreement makes adoption versus retention ambiguous. Adoption rates in
the right panel include those matches, consistent with the alias-aware scoring.
This is a replay of a selected development cohort, with saved A1 and 51 saved
critic responses; it does not measure a fresh end-to-end sample.
