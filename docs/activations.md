# Activations: what is captured, how it is stored, how rows are indexed

## What one activation is

For every Solver generation the collector installs a forward hook on each requested decoder block (`src/mas_sae/activations/capture.py`, sites from `activations/sites.py`: `model.language_model.layers.N` for Gemma 3). The hook keeps only the **first** call of the block, which is the prompt prefill pass, and stores the hidden state at the **last prompt token**, as float32 on CPU. Because the chat template is applied with `add_generation_prompt=True`, that token is the end of the assistant turn header: the model's state just before it produces its first output token. Nothing from the generated tokens is stored.

So, per question and layer:

- **Attempt 1 activation**: the state before the Solver writes A1, with the question and paragraphs in context.
- **Attempt 2 activation**: the state before the Solver writes A2, with the question, paragraphs, A1 and the Critic's feedback in context (in the candidates presentation, A1 and the Critic's recommendation shown as two labelled candidates, without the feedback text). This is the position where "will the Solver take the feedback" is decided but not yet expressed.

Only Solver activations are captured. The hidden size for Gemma-3-4b-it is 2560.

## Files

The paths below use the legacy collection roots. Distributed runs use the same
three tensor filenames inside `prepared/activations/<run>/layer_NN/`; see the
[workflow guide](distributed_collection.md) for the complete bundle layout.

```
data/activations/<run>/layer_NN/<source_split>_attempt1.pt   # one row per collected question
data/activations/<run>/layer_NN/<source_split>_attempt2.pt   # one row per episode
data/activations/<run>/layer_NN/<source_split>.pt            # attempt1 rows, then attempt2 rows
```

The third file is the SAE-facing tensor; `scripts/sae/train_sae.py` reads it as unlabeled rows (train / validation / test). Every file is written through `ActivationStore` and read back to verify the round trip before the run is marked finished (`collection_artifacts.py`).

While a legacy run is in progress, chunks live under `data/activations/<run>/_progress/<source_split>/chunk_XXXXXXXX/` with their own `records.jsonl` and `activations.pt`; `merge_chunks` concatenates them in order and assigns the global indices below.
Manifest workers checkpoint under `<run-directory>/chunks/<work-id>/chunk_XXXXXXXX/`;
the distributed merge uses the same index-reconstruction logic.

## Indices in `interactions.jsonl`

| Field | Row in | Formula |
|---|---|---|
| `attempt1_activation_index` | `<split>_attempt1.pt` | position of the question in collection order |
| `attempt2_activation_index` | `<split>_attempt2.pt` | position of the episode in collection order |
| `sae_attempt1_index` | `<split>.pt` | `attempt1_activation_index` |
| `sae_attempt2_index` | `<split>.pt` | `len(attempt1 rows) + attempt2_activation_index` |

With one condition and one A2 presentation per question (the production policy) the two collection-order indices are equal, and `production._activation_index` refuses a record where they differ. A run with two presentations, such as the candidates pilot, writes two A2 rows for each A1 row, so its indices differ. `collected_question_ids.json` lists question ids in the same order, so row `i` of every per-layer file is the same question.

Alignment is checked twice: by `_validate_alignment` when chunks are merged, and by `load_production_activations` when rows are read, which compares the manifest index with the record's index and the file's row count.

## Partitions and exports

For distributed collection, intervention membership is reserved before inference,
but its baseline episodes and activations are collected with the rest of the
master pool. Train/validation/test membership is assigned after merging.
Intervention baselines are exported under a separate run name.

In new final exports, `source_split` is the actual storage folder and
`original_source_split` retains dataset origin. `full_index` and attempt indices
refer to the final split's tensor rows; `collection_activation_index` records
the original merged row. See the [workflow guide](distributed_collection.md).

### Historical MuSiQue exports

In the historical runs, a MuSiQue source split mixes research roles. The behavior
package manifest maps each question to its research partition and its original
full-run/layer-scan indices. Historical exports retain `source_activation_index`
so their existing readers can verify the original address.

`scripts/activations/export_partition_activations.py` gathers tensors by partition. An
intervention export contains baseline episodes and tensors under its own run
name; causal replay later regenerates A2 live.

The following diagram records the historical MuSiQue allocation. Use the current
workflow above for new mixed-dataset runs.

![Historical MuSiQue partition workflow](diagrams/data_split_workflow.svg)

Editable source: `docs/diagrams/data_split_workflow.drawio`. The overall experimental pipeline diagram is `proposal/revised_capstone_diagram.drawio` (`.drawio.svg` beside it), embedded in `proposal/proposal.md`.

## Who reads what

| Consumer | Reads | Uses the indices? |
|---|---|---|
| SAE training (`scripts/sae/train_sae.py`, `sae/dataloader.py`) | `layer_NN/{train,validation,test}.pt` of an exported run | No: unlabeled rows, A1 and A2 together |
| Probes (`production.load_labeled_rows` + `load_production_activations`) | `labels.csv`, the manifest, `layer_NN/<split>_attempt{1,2}.pt` | Yes: manifest index must equal the record index, with purpose gating (`fit`, `tune`, `evaluate`, `intervene`) |
| Layer-selection probe (PR #51) | exported partition runs through the same loader | Yes |
| Causal stage | separate intervention episodes, matching labels, and saved contexts for mixed datasets; re-runs the Solver live | Indices not needed: the hook runs at generation time |
| Alignment checks (`experiments/alignment.py`, `scripts/activations/validate_pilot_activations.py`) | records and tensors | Yes |

Use `attempt=2` rows for feedback-uptake questions and `attempt=1` rows for the "before feedback" control; never mix a row from `<split>.pt` with an index meant for the per-attempt files.
