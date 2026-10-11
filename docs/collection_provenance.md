# Collection provenance: tracing a run to its exact configuration

A collection run is identified by `output.run_name` in its config. Everything needed to reproduce it, or to refuse a resume that would silently change it, is written by `src/mas_sae/experiments/provenance.py` and `collection_artifacts.py`. This page lists the files, what each field means, and what is and is not recorded.

## Files per run

| Path | Written by | Content |
|---|---|---|
| `results/collection/<run>/<split>/resolved_config.yaml` | `collection_artifacts.py` | The config after resolution, plus a `provenance:` block (below). |
| `results/collection/<run>/<split>/summary.json` | `collection_artifacts.py` | Counts: questions, episodes, exclusions, SAE rows per layer, behaviour and legacy label counts, tensor shapes per layer, hop groups, `experiment_split` counts. |
| `results/collection/<run>/<split>/collected_question_ids.json` | `collection_artifacts.py` | Question ids in tensor row order. |
| `results/collection/<run>/<split>/exclusions.jsonl` | `collection_artifacts.py` | Questions that produced no episode (fields in `episode_record.md`). |
| `results/collection/<run>/<split>/interactions.jsonl` | `collection_artifacts.py` | The episode records (S3 for the production run). |
| `data/activations/<run>/_progress/<split>/provenance.json` | `collection_progress.py` | The run identity used by `--resume`. For the production run this file is committed as `results/collection/natural_4b_full/<split>_provenance.json`. |
| `data/activations/<run>/_progress/<split>/chunk_XXXXXXXX/` | `collection_progress.py` | Per-chunk `metadata.json`, `records.jsonl`, `exclusions.jsonl`, `activations.pt` until the run is merged. |
| `results/collection/<export run>/export_summary.json` | `scripts/export_partition_activations.py` | Commit, manifest and labels hashes, frozen-partition path and hash, layers, counts, file hashes of a partition export. |
| `results/sae/musique/runs/layer_NN/<run id>/manifest.json` | `scripts/train_sae.py` | SAE training provenance, including sha256 of the activation files used. |

## The identity record (`provenance.json`)

Built by `build_collection_progress_provenance`. A resume compares every key in `IDENTITY_KEYS` (`collection_progress.py`) and refuses on any difference; `created_at_utc` and `hash_basis` are informational.

| Key | Meaning |
|---|---|
| `config_sha256` | sha256 of the canonical JSON of the collection config without its `output` section. |
| `manifest_sha256` | sha256 of the canonical JSON of the ordered question list `{question_id, experiment_split, hop_group}`. Different sampling, order or count changes it. |
| `dataset_revision` | Hugging Face Hub revision of `dgslibisey/MuSiQue` that was loaded. |
| `model_revisions.{solver,critic,validator}` | `requested_revision` from the config and `resolved_revision` from the loaded checkpoint; they must agree. |
| `git_commit` | HEAD of the repository when the run started. |
| `git_diff_sha256` | sha256 of `git diff --binary HEAD` (tracked files). The empty-string hash `e3b0c4…` means no tracked change. |
| `untracked_code_sha256` | sha256 of names and bytes of untracked, non-ignored files under `src/` and `scripts/`. Same empty-hash convention. |
| `package_versions` | `torch`, `transformers`, `datasets`. |
| `gpus` | Device names from `torch.cuda.get_device_name`. |
| `hash_basis` | Plain-language description of how each hash above was computed. |

## The `provenance:` block of `resolved_config.yaml`

Adds, per role, the full `role_metadata`: `model_id`, requested and resolved revision, `architecture`, `loader`, `dtype`, `device`, `device_map`, `requested_placement`, `quantization`, `generation`, `chat_template_kwargs`. Also `environment` (`git_dirty`, `cuda_version`, `gpus`), `protocol_version`, `capabilities` (`blind_then_compare`, `controlled_as_own_conclusion`, `type_checked_target`: the behaviours the protocol version selected), `prompt_versions` (the named prompt constants used by each role and condition), and the token budgets.

Version labels are recorded here as configuration; the capabilities list is what actually ran. A later protocol is defined by choosing capabilities, not by branching on the version name.

## The production run, `natural_4b_full`

| Item | Value |
|---|---|
| Config | `configs/collection/v2/natural_4b_full_{train,validation}.yaml`; resolved copies committed per split |
| Code | commit `4e00dd22`, tracked diff empty, untracked `src/`/`scripts/` empty |
| Dataset | `dgslibisey/MuSiQue` at `c8f4f8c9465fb69d31a8eae894c3fd509c4ca321` |
| Solver and Validator | `google/gemma-3-4b-it` at `093f9f388b31de276ce2de164bdc2081324b9767`, bf16, `cuda:0` |
| Critic | `Qwen/Qwen3-4B-Instruct-2507` at `cdbee75f17c01a7cc42f958dc650907174af0554`, bf16, `cuda:0` |
| Decoding | greedy; budgets 64 / 256 / 4 tokens |
| Layers | 8, 17, 25, 33 (Solver decoder blocks) |
| Questions asked | 19,938 train, 2,417 validation |
| Episodes | 19,919 train, 2,413 validation (19 and 4 excluded at the blind step) |
| Packages | torch 2.13.0, transformers 5.17.0, datasets 5.0.1, CUDA 13.0 |
| GPU | one NVIDIA A10G |
| Started | 2026-09-28 09:34 UTC |

The committed files: `results/collection/natural_4b_full/{train,validation}/{resolved_config.yaml,summary.json,collected_question_ids.json,exclusions.jsonl}` and `results/collection/natural_4b_full/{train,validation}_provenance.json`. The resolved config shows `git_dirty: true` while both code hashes are empty: the working tree had changes outside tracked files and outside `src/` and `scripts/` (for example untracked results), which the identity hashes ignore on purpose.

## What is not recorded

- GPU driver version, GPU memory, CPU, host name, operating system and Python version. The machine that produced the production run and the layer scan was an AWS g5 instance with one A10G (23 GB), driver 595.91, Ubuntu 26.04, Python 3.11, recorded here from the instance rather than by the code.
- cuDNN and other library versions beyond the three listed.
- The layer-scan run (`natural_4b_layer_scan`) predates the identity file and has only `artifact_verification.json`; its configs and resolved configs are committed.
- Older export summaries (`natural_4b_partitioned`, `natural_4b_layer_scan_partitioned`) predate the `frozen_partitions*` keys the export script now writes.

Adding driver and host facts to `environment_metadata` would close the first gap; it is a one-function change and a separate ticket.

## Appendix: every field of `interactions.jsonl`

The plain-English dictionary is `episode_record.md`. This table is the complete list, for anyone writing code against the records. `P` is `src/mas_sae/experiments/pipeline.py`; text comparisons use `normalize_answer` in `src/mas_sae/evaluation/scoring.py` (casefold, strip punctuation, drop articles, exact match).

| Field | Type | Set in | Definition |
|---|---|---|---|
| `episode_id` | str | P | `<question_id>__<critic_condition>` |
| `question_id`, `question`, `ground_truth`, `answer_aliases` | str, str, str, list | P from the MuSiQue example | aliases `[]` when absent |
| `seed` | int | `collection.py` | `collection.seed + question_index` |
| `source_split` | str | `collection.py` | MuSiQue split |
| `experiment_split` | str | `collection.py` | role from `dataset.experiment_split` at collection time; absent when the config omits the block |
| `solver_attempt_1` | str | P | `Solver.solve`, once per question |
| `solver_attempt_1_correct` | bool | P | `answer_matches(A1, gold, aliases)` |
| `critic_condition` | str | P | `natural`, `controlled_correct`, `controlled_incorrect` |
| `critic_blind_answer` | str or null | P | `Critic.answer_blind`; null outside natural; present when `capabilities.blind_then_compare` |
| `critic_blind_raw_output` | str | P | natural only |
| `critic_verdict` | str or null | `critic.py` | natural: parsed `agree`/`disagree`, else null; controlled: `agree` iff A1 matches the target |
| `critic_advocated_answer` | str or null | `critic.py` | natural: parsed and stripped, null when empty; controlled: the target |
| `critic_feedback` | str | `CriticFeedback.to_solver_text` | verdict sentence + advocated answer + explanation, as shown to the Solver |
| `critic_raw_output` | str | P | raw compare/critique generation |
| `critic_noncommittal` | bool | `critic.py` | `verdict is None or advocated_answer is None`; always false for controlled |
| `controlled_target_source`, `controlled_target_type_check` | str/any or null | P | present when `capabilities.type_checked_target`; null for natural; `gold`/null for controlled-correct |
| `solver_attempt_2` | str | P | `Solver.revise` |
| `solver_attempt_2_correct` | bool | `scoring.py` | `answer_matches(A2, gold, aliases)` |
| `critic_feedback_correct` | bool or null | `scoring.py` | `answer_matches(advocated, gold, aliases)`; null without an advocated answer |
| `solver_accepted_feedback` | bool or null | `scoring.py` | legacy V1: A2 matches gold if the Critic was right, else matches the advocated string; null without an advocated answer |
| `solver_changed_answer` | bool | `scoring.py` | `not answer_matches(A2, A1)` |
| `validator_final_correct`, `validator_raw_output` | bool or null, str | `validator.py` | first output token YES/NO, else null |
| `behavior_schema_version`, `behavior_matching_basis` | str | `behavior.py` | `lexical_v2`, `normalized_text_not_semantic` |
| `critic_textual_relation` | str | `behavior.py` | advocated vs A1: `exact`, `containment`, `different`, `unresolved` |
| `critic_position_relation` | str | `behavior.py` | `refusal_or_nonanswer`, `same_position`, `different_nonrefusal_candidate`, `ambiguous_or_unresolved` |
| `solver_behavior` | str | `behavior.py` | `retained_solver_a1`, `adopted_critic`, `third_answer_revision`, `no_conflict`, `ambiguous` |
| `direct_critic_adoption` | bool or null | `behavior.py` | `solver_behavior == adopted_critic` for the three resolved outcomes, else null |
| `solver_copied_nonanswer` | bool or null | `behavior.py` | for refusal-type feedback: A2 equals the refusal string; else null |
| `generation_telemetry` | dict | P, `agents/base.py` | `solver_a1`, `solver_a2`, `critic_blind`, `validator` entries and `critic_review` list; each `{generated_tokens, max_new_tokens, reached_token_budget, finish_reason}`; `finish_reason` always null |
| `attempt1_activation_index`, `attempt2_activation_index` | int | `collection.py`, reassigned by `collection_progress.merge_chunks` | rows in `<split>_attempt1.pt` / `<split>_attempt2.pt` |
| `sae_attempt1_index`, `sae_attempt2_index` | int | `collection_artifacts._add_sae_indices` | `attempt1_index`; `len(A1 rows) + attempt2_index` |
| `partition`, `source_run`, `source_activation_index` | str, str, int | `production.export_partition_activations` | partition exports only; indices reset to the exported row position |

`exclusions.jsonl` rows: `question_id`, `question_index`, `seed`, `stage` (`critic_blind` or `controlled_target`), `error_type`, `reason`, `a1_exists` (true), `critic_exists`, `a2_exists` (false). The production run excluded 19 of 19,938 train and 4 of 2,417 validation questions, all at the blind step.
