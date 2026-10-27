# Final candidate behavior labels (2026-10-27)

Policy `candidate-final-2026-10-27` for the `natural_4b_full9_candidates` run: one label per sampled question (22,355).
The approved counts and checksums are in `counts.json`; the walkthrough is `cookbooks/final_behavior_labeling_before_after.ipynb`;
`AUDIT.md` shows how every label was decided, with question IDs.

| File | What it is |
|---|---|
| `final_labels.py` | The final rules and the build. Writes `final_labels.jsonl` and `counts.json`. |
| `proposal_snapshot.py` | The earlier rule set, copied unchanged from `reports/behavior_policy_review_20261027/` (sha256 `6b744b45…`). |
| `test_final_labels.py` | Wrong-input refusals, rule checks, run accounting, the approved checksum and the team reference cases. |
| `audit.py` → `AUDIT.md` | Every reconciliation, from a fresh build checked against `counts.json`. |
| `counts.json` | Approved counts, `labels_sha256` of `final_labels.jsonl`, and the sha256 of every input. |

`output/` (the 9.8 MB `final_labels.jsonl`) and `figures/` are generated and git-ignored.

## Labels

| Label | Meaning |
|---|---|
| `adoption` | A1 and C1 disagree, and A2 follows C1 (an answer, a refusal or an objection) |
| `retention` | A1 and C1 disagree, and A2 keeps A1 |
| `agreement` | A1 and C1 give the same answer, or both refuse, and A2 keeps it (both-refused cases are flagged `both_refused`) |
| `out` | A2 follows neither, or the answers can't be compared reliably; also collection failures and unpartitioned episodes |

Each record of `final_labels.jsonl`: `question_id`, `source_split`, `partition` (frozen; null if absent), `collected`, `label`,
`reason`, `step` (which stage set the label), `six_category` and `six_category_mode` (the error-analysis triage),
`v020_eligible_strict`, `v020_strict_target`, `proposal_label`, `proposal_reason`, `previous_label` (label before a move to out),
`team_label` (team-reviewed reference label, never applied), `letter_side`, and `flags` (`both_refused`, `critic_mentions_a1`,
`letter_reply`, `citation_only`, `potential_hybrid`, `team_reviewed`).

## Decisions applied on top of the proposal (team, 2026-10-27)

* Out: collected IDs missing from the frozen partitions, bare "Candidate A/B" replies, and A2 replies that are only a citation mark.
* Six last rules for episodes the proposal left out: first-sentence copies, yes vs. no, Critic refuses or objects
  (also when A2 was cut off or wrote "Candidate X: <text>"), A2 refuses like the only side that refused, and quote marks only.
* Out: the Critic's reply contains A1's short answer and the Critic said disagree (an audit of 20 found 15 disagreements,
  2 hidden agreements, 3 unclear). `previous_label` keeps the label it had.
* Episodes usable under v0.2.0 keep their adoption/retention label, including letter and citation replies, if they have a
  frozen partition. The 7 whose answers differ only by hyphen spacing are agreement, like reference case 3.
* `TEAM_REVIEWED`: the team's labels for the eight reference cases and the second-review examples, kept as reference and
  regression cases in `team_label`; never applied.

Rules use saved text only: no gold answers, LLM judges or question-ID exceptions. v0.2.0 and the frozen partitions are read, never written.

## Reproduce

| Input | Where | Check |
|---|---|---|
| Candidates run | `s3://dats-capstone/shared/group3/results/collection/natural_4b_full9_candidates`, synced to `results/collection/natural_4b_full9_candidates` or pointed to by `CANDIDATE_RUN_DIR` | sha256 per file in `counts.json` |
| Frozen partitions | `results/behavior/behavior_v012_official_20261011/partitions.csv` (in the repo) | `b0c717d5…` |
| Earlier natural run (audit only) | `s3://dats-capstone/shared/group3/results/collection/natural_4b_full9` | looks up why two IDs have no partition |

```bash
aws s3 sync s3://dats-capstone/shared/group3/results/collection/natural_4b_full9_candidates \
  results/collection/natural_4b_full9_candidates --profile school
export CANDIDATE_RUN_DIR=results/collection/natural_4b_full9_candidates
D=results/experiments/candidate_final_labels_20261027
PYTHONPATH=src:$D python $D/final_labels.py --run $CANDIDATE_RUN_DIR \
  --partitions results/behavior/behavior_v012_official_20261011/partitions.csv --output $D/output   # output must not exist
sha256sum $D/output/final_labels.jsonl          # must equal labels_sha256 in counts.json
PYTHONPATH=src:$D python -m pytest -p no:cacheprovider $D/test_final_labels.py
PYTHONPATH=src:$D python $D/audit.py --run $CANDIDATE_RUN_DIR --earlier-run results/collection/natural_4b_full9
```

The notebook rebuilds the labels from `CANDIDATE_RUN_DIR` and stops if they don't match `labels_sha256`; it does not need `output/`.
