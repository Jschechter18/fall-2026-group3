# Negative-class feedback-quality audit (issue #115)

Date: 2026-10-19. Scope: the Natural-condition development partitions of `natural_4b_full` (15,936 train and 2,413 validation episodes). The labels audited are `behavior_v0.1.2_qc_candidate`. Test and intervention rows were not read. No labels were changed.

## Summary

The primary target compares adopted (positive) with retained_a1 + third_answer (negative). The audit asked whether its negatives are real cases of the Solver weighing coherent Critic feedback and not adopting it. Mostly they are not. Answer matching alone does not identify the behavioural contrast the SAE experiment needs.

A usable example has to pass three separate checks:

| Question | Construct | Evidence used here |
|---|---|---|
| Did A2 match the Critic's answer? | Observed answer matching | Label rules (text relation); human `solver_response` |
| Was the Critic's feedback coherent and usable? | Feedback quality | Frozen coherence rules; human `feedback_type` and contradiction notes (indirect) |
| Was there a real A1-vs-Critic disagreement that A2 resolved? | Primary eligibility | Human `eligible_primary`. The rule `candidate_status` is only a screen. |

What #115 established:

1. **It detects contradictory or unusable Critic feedback.**
   - The frozen rules sort the delivered feedback into consistent, inconsistent, uncertain or unusable. They use A1 and the Critic output only, never A2, gold or correctness.
   - They agree with human contradiction notes. Of 125 QC rows with a noted contradiction, the rules called 100 inconsistent. Of 434 without one, they called 4 inconsistent.
2. **It identifies mislabelled negatives.**
   - **Kept A1:** humans judged 2 of the 111 train negatives eligible.
   - **Third answer:** in a blind sample of 30, humans labelled 13 as `adopted_critic` and none as `third_answer`. Four were eligible, and all four are eligible adoptions: positives sitting in the negative class.
3. **It separates observed answer matching from eligible adoption or rejection.**
   - **Matched is not eligible.** Of the 13 matched rows above, 9 were not eligible.
   - **The positive class is diluted too.** Humans judged 12 of 30 sampled `adopted_critic` rows eligible. In the largest stratum it was 3 of 10.

**Feasibility.** The v0.1.2 labels have supported preliminary behavioural probing, and they can continue to. But substantial eligibility and answer-matching errors limit how confidently those results, and any causal findings built on them, can be read as adoption versus rejection:

- few negatives survive human review;
- some negatives are mislabelled adoptions;
- the positives are diluted by ineligible matches.

Relabelling the negatives alone would not fix this. Both classes need the same eligibility filtering before an adoption-versus-rejection interpretation can be made with confidence.

## What #115 does not do

- **It does not fix the class imbalance.** The Solver adopts far more often than it rejects: 3,830 vs 347 eligible in train. #115 improves how the classes are measured, not their sizes. A class-balance experiment is separate work.
- **It does not do semantic answer matching or handle partial revisions.** The issue lists both under Solver re-evaluation. #115 identifies the resulting errors but does not correct them:
  - "St. Paul" vs "Saint Paul" (F0007);
  - sentence answers that restate an entity;
  - A2s that partly revise A1.

  This work is deferred to #108. #115 is completed with this documented scope limitation, not as finished functionality. The team should acknowledge the scope decision before the issue is closed. Until #108 lands, third-answer and adoption detection rely on text matching.
- **It does not relabel anything.** Official v0.1.2 artifacts are unchanged. The issue excluded relabelling.

## Procedure

- **Script:** `scripts/audit_feedback_quality.py`.
- **Logic:**
  - `mas_sae/evaluation/feedback_coherence.py` (coherence rules)
  - `mas_sae/evaluation/feedback_audit.py` (audit rows, candidate status, human reconciliation)
- **Shared loaders:** `mas_sae/data/production.py` gained `interaction_paths` and `load_labeled_rows(..., partitions=)`. A run therefore only opens the interaction file for its partition.
- **Critic parser:** `mas_sae/agents/critic.py` exposes `first_json_object_span`, so the audit can see text generated after the parsed JSON. Production parsing behaviour is unchanged.
- **Development and freeze:**
  - The rules were developed on train only.
  - They were frozen on 2026-10-19, before validation was read.
  - The validation run is pinned to the frozen hash (`--expect-procedure-sha256`) and was read once.

| Item | sha256 |
|---|---|
| Audit procedure (frozen) | `28b75b5c425e1cfaca71a15c202316313c30e84b92dc131d12b1a515c39ec92f` |
| Coherence rules | `cd74c6c9a1bb9bc4cef4c338880ef93abd0e246246a22a3ce8319a4d3b0c179b` |
| Train interactions | `000700768a1aca9ea3fdd6acd372d7506d9d74fdb0501125ffba3a0bc9f8e78e` |
| Validation interactions | `5054d85068f5f3893922296b4aa14b704241937d54af4cd83f625713b8fe5ae9` |
| v0.1.2 labels.csv (unchanged) | `8d9d1a49a1ff562ec44ea531aea544b9844244cf785514d91409b0365fe2bf6b` |
| v0.1.2 split_manifest.csv (unchanged) | `aa576d11773091080b444cebcdbe633314cd7826f18f0f8baf182d8d6a81ec5b` |

The input hashes are checked before and after each run.

## Results

### Coherence of eligible primary-target rows (v0.1.2 eligibility)

| Partition | Class | n | Consistent | Uncertain | Inconsistent |
|---|---|---:|---:|---:|---:|
| Train (rules developed) | Adopted critic | 3,830 | 3,520 | 278 | 32 |
| | Kept A1 | 111 | 19 | 31 | 61 |
| | Third answer | 236 | 175 | 55 | 6 |
| Validation (frozen) | Adopted critic | 578 | 515 | 58 | 5 |
| | Kept A1 | 13 | 3 | 3 | 7 |
| | Third answer | 45 | 34 | 8 | 3 |

### Automatic eligibility screen (not human-confirmed)

| Partition | Class | rule_pass | uncertain | invalid |
|---|---|---:|---:|---:|
| Train | Adopted critic | 3,122 | 676 | 32 |
| | Kept A1 | 3 | 47 | 61 |
| | Third answer | 63 | 167 | 6 |
| Validation | Adopted critic | 458 | 115 | 5 |
| | Kept A1 | 1 | 5 | 7 |
| | Third answer | 8 | 34 | 3 |

`rule_pass` means the automatic checks found no problem. It is not a confirmed eligible example: humans judged only 8 of 22 `rule_pass` rows in the blind packet eligible. Validation shows the same pattern as train.

Across all train episodes, coherence was 12,734 consistent, 1,769 uncertain, 498 inconsistent and 935 unusable (noncommittal). Validation was 1,881 / 289 / 88 / 155. Role confusion is the most common reason for an uncertain call: 1,625 train rows.

### Human evidence (train only)

**A. The 559-row QC.** It informed rule development, so it is in-sample.

- **All 111 eligible kept-A1 negatives were reviewed:**
  - eligible: 2 yes, 103 no, 6 uncertain
  - `solver_response`: 84 retained_a1, 25 unchanged_same_position, 2 unresolved
- **29 of 236 third-answer negatives were reviewed:** eligible 4 yes, 23 no, 2 uncertain.
- **Human eligibility vs the rule screen**, over QC rows that have a candidate status:

  | Human \ rules | rule_pass | uncertain | invalid |
  |---|---:|---:|---:|
  | eligible = yes | 28 | 9 | 0 |
  | eligible = no | 14 | 64 | 56 |
  | eligible = uncertain | 2 | 2 | 5 |

**B. The blind 104-row packet.**

- Packet: `issue115_feedback_audit_20261013/train/review_packet`, 11 strata, seed 42.
- Annotator A, completed 2026-10-19.
- No overlap with the 559 QC rows, and rule outputs were hidden from the annotator.

| Sampled group (rows) | Human eligible yes / no / uncertain | Human `solver_response` |
|---|---|---|
| Adopted critic (30) | 12 / 17 / 1 | 26 adopted_critic, 4 unchanged_same_position |
| Third answer (30) | 4 / 26 / 0 | 13 adopted_critic, 10 produced_answer, 5 unchanged_same_position, 1 retained_a1, 1 unresolved |
| Excluded, partial match (20) | 1 / 19 / 0 | – |
| Excluded, wording conflict (24) | 0 / 22 / 2 | – |

The adopted-critic consistent stratum holds 3,396 of 3,790 rows in the sampling population. Human eligibility there was 3 of 10. Weighting the three adopted strata to their population sizes gives about 31% eligible. That rests on 10 rows per stratum, so treat it as a warning, not an estimate.

**Third-answer group (30 rows):**

| | Rows |
|---|---:|
| Human `adopted_critic` | 13 |
| Human `eligible_primary = yes` | 4 |
| Both | 4 |
| Adopted but not eligible | 9 |

The packet had no reason field and the notes are blank, so why those 9 were excluded is not recorded. As a proxy, human `feedback_type` was premise_rejection for 4 (for example F0054), unresolved for 1, and answer for 4. The coherence rules rated those 9 as 5 consistent and 4 uncertain, so they do not explain the exclusions.

Coherence against human `feedback_type`: of 13 packet rows the rules called inconsistent, the annotator labelled 9 as unresolved feedback.

Agreement with the v0.1.2 labels across the 104 rows: `solver_response` 40 of 104, `feedback_type` 61 of 104.

## Tests and reproducibility

- Full unit suite (`PYTHONPATH=src`, capstone env): 720 passed, 3 skipped. `test_interpretability.py` was excluded because `shap` is not installed in the environment.
- #115 test files (`test_feedback_audit.py`, `test_feedback_coherence.py`, `test_production_data.py`): 60 passed.
- A train rerun in a scratch folder reproduced `audit.csv` byte for byte (`2156c14d…`).

## Limitations

- **One annotator.** There is no second annotator and no agreement estimate.
- **No direct coherence question.** Coherence was checked only indirectly, through `feedback_type` and contradiction notes.
- **In-sample evidence.** The 559-row QC informed the rules. The validation partition has no human labels, so validation results are automatic only.
- **Small samples.** 30 rows per packet group; 13 kept-A1 and 45 third-answer negatives in validation.
- **Reformatted cells.** The annotator's spreadsheet reformatted four displayed cells: F0008 and F0059 (date), F0024 ("1/3" shown as "01-Mar") and F0033 (percentage). Scoring used the original packet text, but F0024 may have been judged on the reformatted value.
- **Text matching.** Matching relies on it until #108.
- **Correctness.** Correctness fields, including the LLM validator, are recorded but not used. The validator is not itself validated.

## Outputs

| Path (under `~/capstone-artifacts/`) | sha256 |
|---|---|
| `issue115_feedback_audit_20261014/train/audit.csv` | `2156c14d7554d0e44cc8e8b61371742a84a38c1b0c10ce07d671fdc68d779bd0` |
| `issue115_feedback_audit_20261019/validation/audit.csv` | `6969bbf5a2dc44ca057df27cc98d059ace8aaf32c64d05ec1cd2bf33a7c50c7b` |
| `issue115_feedback_audit_20261019/validation/counts.json` | `12831af32b9801bba157d5ab469db5d24ac89bcdcacce31a048f7a637b50d487` |
| `issue115_feedback_audit_20261013/train/review_packet/annotator_a.csv` (blank packet, unchanged) | `41860f63e6289a49…` |
| `issue115_feedback_audit_20261013/train/review_packet/private_key.json` (unchanged) | `09960e7c58a9676a…` |
| `issue115_feedback_audit_20261013/train/review_packet/annotator_a_completed_20261019.csv` (completed, read-only copy) | `40de165afe3f313c756020b90487555dab902c63fef324ada9aee19828069941` |

To reproduce validation (it is read once, so this is a check, not a re-tune):

```
PYTHONPATH=src python scripts/audit_feedback_quality.py \
    --results ~/capstone-artifacts/natural_4b_full/results \
    --labels ~/capstone-artifacts/behavior_v012_official_20261011_release \
    --partition validation \
    --human-qc ~/capstone-artifacts/qc_v011_review/human_qc_559_unified.csv \
    --human-notes ~/capstone-artifacts/issue55_human_evidence/annotator_a_completed_original.csv \
    --expect-procedure-sha256 28b75b5c425e1cfaca71a15c202316313c30e84b92dc131d12b1a515c39ec92f \
    --output <new folder>
```
