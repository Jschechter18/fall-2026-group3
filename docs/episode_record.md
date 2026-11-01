# What is in one episode record

One line of `interactions.jsonl` is one episode: a question, the Solver's first
answer, the Critic's review, the Solver's second answer, and the recorded
metadata. Examples below use the existing Gemma/Qwen roles.

For [distributed mixed-dataset runs](distributed_collection.md), the shared
manifest retains the exact input payloads; the separate intervention bundle
also contains `contexts.jsonl` for causal replay. Historical MuSiQue runs recover
paragraphs by original question ID and dataset revision.

This page explains the fields a reader needs to follow the experiment. The complete technical list of every field, with types and the code that sets each one, is the appendix of `collection_provenance.md`; the activation files and row indices are explained in `activations.md`.

## The exchange

| Field | What it means | Why it matters |
|---|---|---|
| `question_id` | `dataset_source:original_id` in mixed runs; historical MuSiQue IDs encode hop count in their prefix | The key that joins labels, partitions and activations |
| `episode_id` | `question_id` plus the feedback condition, and the A2 presentation in runs that record one | One row per episode; the condition is always `natural` in production |
| `question` | The question Gemma was asked | What the whole exchange is about |
| `ground_truth`, `answer_aliases` | the source record's correct answer and accepted alternative spellings | Lets us say whether each answer was right |
| `solver_attempt_1` | Gemma's first answer (A1), before any feedback | The starting position |
| `critic_blind_answer` | Qwen's own answer, given before it saw A1 | Shows the Critic formed a position of its own |
| `critic_verdict` | `agree` or `disagree` with A1 | Whether the Critic pushed back |
| `critic_advocated_answer` | The answer Qwen argued for | What "adopting the Critic" means for this episode |
| `critic_feedback` | The exact text Gemma read before answering again, in the feedback presentation | The treatment in the feedback presentation; the candidates presentation shows only `critic_advocated_answer` |
| `critic_noncommittal` | `true` when Qwen's output gave no usable position | Such episodes cannot measure uptake and are excluded |
| `critic_raw_output`, `critic_blind_raw_output` | Qwen's unedited outputs | Kept so the review can be re-read or re-parsed |
| `solver_attempt_2` | Gemma's second answer (A2), after reading the feedback or the two candidates | The outcome: adopt, keep, or change |
| `critic_condition` | The feedback condition | Always `natural` in the production run; the controlled conditions are off |

## How A2 was prompted (runs that record it)

Runs whose config sets `a2_presentations` record how the second answer was asked for. Runs collected before that, including `natural_4b_full` and `natural_4b_full9`, used the feedback presentation and do not have these fields.

| Field | Meaning |
|---|---|
| `a2_presentation` | `feedback`: Gemma read the Critic's feedback text. `candidates`: Gemma saw A1 and the Critic's recommendation as Candidate A and Candidate B, labelled "your previous answer" and "the reviewer's recommendation" |
| `candidate_order` | `a1_first` or `critic_first`: which answer was Candidate A. Seeded per question, balanced within each hop group, and fixed before any generation; null in the feedback presentation |
| `candidate_first_token_ids` | the first token id of each candidate (`a1`, `critic`) |
| `decision_token_distinct` | whether the two candidates start with different tokens; when they do not, the activation before the first output token cannot tell them apart |
| `solver_a2_first_token_id` | the first token Gemma generated for A2 |

## Correctness (string match against the gold answer and aliases)

Correctness is kept apart from behaviour on purpose: a wrong Critic can still be adopted.

| Field | Meaning |
|---|---|
| `solver_attempt_1_correct` | A1 matches the gold answer |
| `critic_feedback_correct` | Qwen's advocated answer matches the gold answer |
| `solver_attempt_2_correct` | A2 matches the gold answer |
| `validator_final_correct`, `validator_raw_output` | a separate Gemma call that only checks whether A2 means the same as the gold answer; its yes/no and raw text |

## Behaviour: what Gemma did

For the feedback runs, the labels to use are the official Behavior v0.1.2 labels (policy `v0.1.2-qc`, `results/behavior/behavior_v012_official_20261011/labels.csv`; see `behavior_v01.md`). They are joined to the record by `question_id`.

| Field (in `labels.csv`) | Meaning |
|---|---|
| `rule_version` | which version of the rules produced this label |
| `feedback_type` | what Qwen gave: `answer`, `refusal`, `premise_rejection` (rejected the question itself), `unresolved` |
| `solver_response` | what Gemma did: `adopted_critic`, `retained_a1`, `third_answer`, `unchanged_same_position`, `non_answer`, `produced_answer`, `unresolved` |
| `conflict_status` | A1 versus Qwen's answer: `lexical_conflict_candidate` (different), `same_position`, `containment` (one contains the other, e.g. "Paris" and "Paris, France"), `unresolved` |
| `eligible_primary` | can be used to train or score the probe: Qwen gave a real, different answer and Gemma's reply is classifiable |
| `primary_target` | the probe's target when eligible: 1 = adopted the Critic, 0 = kept A1 or gave a third answer |
| `eligible_strict`, `strict_target` | same, but only adopt versus keep (third answers excluded) |
| `exclusion_reasons` | why an episode is not eligible, e.g. `no_conflict`, `critic_refusal`, `a2_nonanswer`, `a1_critic_containment` |
| `human_validated_behavior_v1` | still empty; the human labels from the QC live in the private evidence folders named in `behavior_v01.md` |

Older behaviour fields on the record itself (`solver_behavior`, `critic_position_relation`, `direct_critic_adoption`, `solver_copied_nonanswer`, `solver_accepted_feedback`, `solver_changed_answer`) come from the earlier `lexical_v2` rule set and are kept for comparison only.

## Where the activations are

| Field | Meaning |
|---|---|
| `partition` | research role in prepared exports; intervention is reserved before collection, while ordinary roles are assigned after merging |
| `source_split` | actual tensor/interaction folder; also the original MuSiQue split in legacy collection files |
| `original_source_split` | original dataset split, retained separately in mixed runs |
| `experiment_split` | the group the collection run assigned at the time; history only, the partition is the authority |
| `attempt1_activation_index`, `attempt2_activation_index` | row of this episode in `<split>_attempt1.pt` and `<split>_attempt2.pt` (always equal in production) |
| `sae_attempt1_index`, `sae_attempt2_index` | rows in the combined `<split>.pt` (A1 rows first, then A2 rows) |
| `source_run`, `source_activation_index` | legacy exports: the original run and row before regrouping |
| `collection_activation_index` | new exports: original merged row for traceability; attempt indices address the final tensors |

## Bookkeeping

| Field | Meaning |
|---|---|
| `seed` | the question's collection seed; frozen in the distributed manifest independently of chunk assignment |
| `generation_telemetry` | one entry per model call (`solver_a1`, `critic_blind`, `critic_review`, `solver_a2`, `validator`): tokens written, the limit, and whether the limit was hit, which means the output may be cut off |
| `controlled_target_source`, `controlled_target_type_check` | only used by the controlled feedback conditions; `null` here |
