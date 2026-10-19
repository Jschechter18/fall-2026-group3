# Preliminary adoption experiment summary

## Slide-ready comparison

Same 98 completed question/recommendation trials from 80 unique questions in each row. The original baseline below is the controlled-feedback revision run, not an estimate over the full natural-feedback collection.

| Condition | Adopted | Retained A1 | Other | Adopted wrong | Adopted correct |
| --- | --- | --- | --- | --- | --- |
| Original revision baseline | 89 | 4 | 5 | 51/53 | 38/45 |
| Evidence verification | 86 | 8 | 4 | 48/53 | 38/45 |
| Labeled candidates: A1 first | 35 | 61 | 2 | 16/53 | 19/45 |
| Labeled candidates: critic first | 36 | 59 | 3 | 18/53 | 18/45 |
| Anonymous candidates: A1 first | 10 | 73 | 15 | 3/53 | 7/45 |
| Anonymous candidates: critic first | 26 | 50 | 22 | 14/53 | 12/45 |
| Labeled candidates: randomized order | 32 | 65 | 1 | 16/53 | 16/45 |
| Anonymous candidates: randomized order | 18 | 61 | 19 | 9/53 | 9/45 |

## Suggested slide takeaway

**Presenting both answers as labeled candidates yielded substantial adoption and retention without explicitly instructing rejection.** In this selected sample, randomized labeled presentation changed the strict class counts from 89 adoptions / 4 retentions to 32 / 65.

This is a change of interaction protocol: original feedback revision versus labeled candidate evaluation. It does not establish that deference in the original system has been fixed.

## Order check

| Assigned order | Trials | Adopted | Retained A1 | Other |
| --- | --- | --- | --- | --- |
| a1_first | 47 | 17 | 29 | 1 |
| critic_first | 51 | 15 | 36 | 0 |

A fixed first-candidate heuristic scored 46.2% balanced accuracy on adoption/non-adoption. This is a descriptive sanity check, not a trained or held-out probe; it does not prove that position effects are absent.

## What to say aloud

- The original revision setup strongly favored adoption, including wrong recommendations.
- Labeled candidate presentation produced both outcomes in each randomized-order group.
- Anonymity suppressed useful corrections and showed order sensitivity, so randomized labeled presentation is the leading protocol.
- The objective is measurable behavioral variation for probing, not necessarily improved task accuracy.
- Freeze the protocol and validate on fresh development questions before scaling activation collection.

## Interpretation limits

- These are exploratory results on one repeatedly used, stratified development sample. Do not describe these rows as independent datasets or claim statistical significance from this table.
- Trials share questions, and some questions have two wrong-recommendation conditions. The source run had 100 sampled questions, with 98 completed trials across 80 distinct questions after eligibility and target-generation skips.
- Correct/incorrect recommendations are operationalized by dataset gold/alias matching. Matching can miss semantic equivalence; intended plausibility levels and entity types are imperfect.
- Accept means output matches the recommended answer (gold aliases also count for correct corrections). Retain means output matches A1. Other includes third answers and uncertainty; it is not silently counted as retention.
- In recommendation_outcomes.csv, non-adoption includes retention and other outputs. A correct_nonadoption of a wrong recommendation does not imply a correct final answer.
- The fresh randomized run reproduced the corresponding earlier labeled cells in the prior analysis. This supports reproducibility, not generalization to unseen questions.
- The data are controlled recommendation tests; they do not improve or measure the natural Qwen critic's accuracy. No new activation or probe result is reported.

## Files

- The anonymous randomized row selects saved anonymous responses using the exact same question-level order assignments as the labeled randomized row. No new inference or resampling is performed.

- slide_table.csv: compact presentation table.
- method_summary.csv: counts, percentages, final correctness, distinct questions and baseline deltas.
- recommendation_outcomes.csv: four recommendation-following outcomes with retention/other splits.
- randomized_order_breakdown.csv: adoption and retention within each assigned order.
- adoption_comparison.png / .svg: slide-ready chart, when --chart is enabled.
- provenance.json: input checksums and source sample checksum.
