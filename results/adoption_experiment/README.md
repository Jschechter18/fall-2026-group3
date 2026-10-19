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
