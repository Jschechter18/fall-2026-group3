# Capstone Project

Master's Data Science Capstone project.

Setup: [Getting started](#getting-started) and [cookbooks/setup.md](cookbooks/setup.md).

## Getting started

```bash
git clone git@github.com:Jschechter18/fall-2026-group3.git
cd fall-2026-group3
conda env create -f environment.yml    # Python 3.11, environment "capstone"
conda activate capstone
python -m pip install -e .
```

Full instructions are in [cookbooks/setup.md](cookbooks/setup.md):

- **Claude on Amazon Bedrock:** used by stages 2 and 6.
- **Hugging Face model access:** the Gemma licence and pinned model revisions.
- **Amazon S3:** where the collection artifacts, activations and SAE checkpoints are stored.
- **Jupyter kernel:** for the analysis notebooks.

When `environment.yml` changes, run `conda env update -f environment.yml --prune`. When you add a
dependency, add it to `environment.yml` so that everyone uses the same environment.

## Experimental pipeline

**Question:** when a Critic gives the Solver (Gemma-3-4B-it) a different answer, are SAE features in
the Solver's layer-25 residual stream causal for whether it _adopts_ the Critic's answer or
_retains_ its own?

The final results come from one chain of runs, listed below. Every later stage reads the exact
paths written by the stage before it. Run everything from the repository root with
`conda activate capstone` and `export PYTHONPATH=src`.

| #   | Stage                                               | Config                                                                  | Final output                                                                                   |
| --- | --------------------------------------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| 1   | [Collection](#1-collection)                         | `configs/collection/distributed_final.yaml`                             | `artifacts/collection/mixed_final_001/prepared/` (S3)                                          |
| 2   | [Claude behavior labels](#2-claude-behavior-labels) | none (CLI flags)                                                        | `results/claude_validation/mixed_final_001/`, `results/behavior/mixed_final_001_claude_probe/` |
| 3   | [Layer selection](#3-layer-selection)               | `configs/layer_selection/layer_selection_mixed_final_001.yaml`          | `results/layer_selection_mixed_final_001_quick/20261102T201434082931Z/` → layer 25             |
| 4   | [SAEs](#4-saes)                                     | `configs/sae/layer_25_mixed_final_001_baselines.yaml` + CLI             | 4 SAE folders, see below                                                                       |
| 5   | [Feature selection](#5-feature-selection)           | `configs/feature_selection/feature_selection_intervention_layer25.yaml` | `results/feature_selection_intervention_layer25/20261105T025730344698Z_pool/feature_sets.json` |
| 6   | [Intervention](#6-intervention)                     | `configs/intervention/intervention.yaml`                                | `results/intervention_layer25/20261106T000808809026Z-02871d677287/merge/`                      |
| 7   | [Analysis](#7-analysis)                             | none                                                                    | `notebooks/intervention/`, `results/feature_interpretation_layer25/`                           |

Partitions are fixed in stage 1 and respected everywhere after it. Probes, SAEs and feature
selection fit on `train` and measure on `validation`. `test` is never read. The `intervention`
partition (10% of questions, reserved before collection) is used only in stage 6.

### 1. Collection

Mixed MuSiQue + synthetic questions. Each question goes Solver A1 → Natural Critic
(Qwen3-4B-Instruct-2507, blind first) → Solver A2, with the candidates presentation. Solver
activations are saved at layers 4, 8, 17, 25, 29–33. The collection spread the work across several
EC2 machines.

```bash
python scripts/collection/distributed_collection.py prepare --config <run_dir>/collection.yaml --num-chunks <N> --output <run_dir>/manifest.json
python scripts/collection/collect_activations.py --config <run_dir>/collection.yaml --manifest <run_dir>/manifest.json --chunk <K ...>   # each worker
python scripts/collection/distributed_collection.py merge --manifest <run_dir>/manifest.json
python scripts/collection/distributed_collection.py split --manifest <run_dir>/manifest.json --config <run_dir>/collection.yaml
```

- **Inputs:** `data/MuSiQue/clean/{train,validation}.jsonl` and `data/synthetic/final/accepted.jsonl`
  (version `synthetic-final-001`). `scripts/data/get_musique_dataset.py` writes the MuSiQue files as
  `train.json` and `validation.json`; they already hold JSON Lines, so copy them to the `.jsonl` names
  the config reads.
- **Full walkthrough:** [cookbooks/runbooks/mixed_collection_pipeline.md](cookbooks/runbooks/mixed_collection_pipeline.md).
  It is written for the earlier `mixed_opus5_full_001` run name; the steps are the same. See also
  [cookbooks/distributed_collection.md](cookbooks/distributed_collection.md).
- **Output:** `artifacts/collection/mixed_final_001/prepared/`, which is gitignored. You don't need
  to rerun collection; download this folder from S3 (see [cookbooks/setup.md](cookbooks/setup.md#s3-directory-layout)).
  - `activations/mixed_final_001/layer_NN/{train,validation,test}_attempt{1,2}.pt`
  - `activations/mixed_final_001_intervention/layer_NN/`
  - `results/mixed_final_001/{train,validation,test}/interactions.jsonl`, plus `partitions.csv` and `split_manifest.csv`
  - `results/mixed_final_001_intervention/intervention/interactions.jsonl` and `contexts.jsonl`

### 2. Claude behavior labels

Claude Sonnet 4.5 on Bedrock (prompt `claude_binary_v3_candidates`, temperature 0) labels each A2 as
adoption, retention, agreement or out. Only adoption vs. retention is used as the target. The
exact settings are in each run's `run_settings.json`.

```bash
python scripts/labels/run_claude_validation.py \
  --episodes artifacts/collection/mixed_final_001/prepared/results/mixed_final_001/<partition>/interactions.jsonl \
  --partition <partition> --output results/claude_validation/mixed_final_001/<partition>

python scripts/labels/export_final_labels_for_probes.py \
  --claude-runs results/claude_validation/mixed_final_001/train results/claude_validation/mixed_final_001/validation \
  --full-results artifacts/collection/mixed_final_001/prepared/results/mixed_final_001 \
  --layout partitioned --output results/behavior/mixed_final_001_claude_probe
```

- **Raw labels:** `results/claude_validation/mixed_final_001/{train,validation,test,intervention}/`.
  The `intervention/` folder supplies the stored labels that stage 6 tries to flip. How to run it:
  [cookbooks/claude_intervention_labels.md](cookbooks/claude_intervention_labels.md).
- **Label package for probes:** `results/behavior/mixed_final_001_claude_probe/`
  (`labels.csv`, `split_manifest.csv`, `excluded_episodes.json`, `counts.json`). This is the only
  label input for stages 3–5.

### 3. Layer selection

Fits a tuned linear probe (adoption vs. retention) per layer on `train` and ranks the layers on
`validation`.

```bash
python scripts/layer_selection/layer_selection.py --config configs/layer_selection/layer_selection_mixed_final_001.yaml --inspect   # check inputs first
python scripts/layer_selection/layer_selection.py --config configs/layer_selection/layer_selection_mixed_final_001.yaml
```

- **Result:** layer 25 is best, with validation balanced accuracy 0.843 (CI 0.830–0.856). Layer 17
  is next at 0.828.
- **Location:** the committed run is `results/layer_selection_mixed_final_001_quick/20261102T201434082931Z/`
  (`layer_comparison.csv`). It used the config's probe search but set `n_permutations: 0` and wrote
  to a `_quick` folder, through a copy of the config that is not in the repository. So the
  label-shuffle control (a p-value and selectivity for each layer) was not run. The ranking,
  balanced accuracy and bootstrap CIs don't depend on that control. Running the committed config
  as is takes about 10 CPU-hours for the 100 shuffles and writes to
  `results/layer_selection_mixed_final_001`, which has no committed run.
- `results/layer_selection_mixed_final_001_attempt1_quick/` is the same quick check on attempt-1
  activations (before the Critic's feedback). It is a side check, not the selection.

### 4. SAEs

The study compares four layer-25 SAEs, plus raw activations as a control:

| Name in configs                | Kind                                         | Folder                                                                                |
| ------------------------------ | -------------------------------------------- | ------------------------------------------------------------------------------------- |
| L1 SAE                         | trained here                                 | `results/sae/musique/runs/layer_25/20261102T182346930753Z-150e78c341c6`               |
| Pretrained SAE                 | Gemma Scope 2, `kind: gemma_scope`           | `results/sae/pretrained/gemma-scope-2-4b-it/resid_post_all/layer_25_width_16k_l0_big` |
| Fine-tuned decoder             | decoder-only fine-tune of the Pretrained SAE | `results/sae/decoder_finetuning/20261104T164310458065Z/best`                          |
| Fully fine-tuned (exploratory) | full fine-tune of the Pretrained SAE         | `results/sae/full_finetuning/20261104T151230848807Z/best`                             |

**L1 SAE.** Run a grid search, then finish the chosen trial (trial 2 of study
`results/sae/musique/tuning/layer_25/20261102T130735203392Z-150e78c341c6/`). Details are in
[results/sae/README.md](results/sae/README.md).

```bash
python scripts/sae/train_sae.py --hp-tune --config configs/sae/layer_25_mixed_final_001_baselines.yaml
python scripts/sae/train_sae.py --finish-run results/sae/musique/tuning/layer_25/<study_id>/trials/trial_0002 --epochs 200
```

**Pretrained SAE.** Download the folder (`config.json` + `params.safetensors`) from
[google/gemma-scope-2-4b-it](https://huggingface.co/google/gemma-scope-2-4b-it):

```bash
hf download google/gemma-scope-2-4b-it --include "resid_post_all/layer_25_width_16k_l0_big/*" \
  --local-dir results/sae/pretrained/gemma-scope-2-4b-it
```

**Fine-tuned SAEs.** Both start from the original Pretrained SAE; add `--train full` for the full
fine-tune. Details are in [scripts/sae/PRETRAINED_FINETUNING.md](scripts/sae/PRETRAINED_FINETUNING.md).

```bash
python scripts/sae/finetune_pretrained_sae.py [--train full] \
  --sae-dir results/sae/pretrained/gemma-scope-2-4b-it/resid_post_all/layer_25_width_16k_l0_big \
  --activations-dir artifacts/collection/mixed_final_001/prepared/activations/mixed_final_001/layer_25 \
  --layer 25 --probe-config configs/feature_selection/feature_selection_mixed_final_001_layer25.yaml
```

`decoder_finetuning/` and `full_finetuning/` hold several runs each. Only the two listed above are
used, and the configs in stages 5 and 6 name them.

### 5. Feature selection

For each representation, this ranks features by how well they separate adoption from retention
(fit on `train`, measured on `validation`). It keeps the top 10 adoption and top 10 retention
features, plus 100 matched random draws per label as controls.

```bash
python scripts/feature_selection/feature_selection.py --config configs/feature_selection/feature_selection_intervention_layer25.yaml
python scripts/intervention/build_feature_set_pool.py \
  results/feature_selection_intervention_layer25/20261105T025730344698Z \
  results/feature_selection_intervention_layer25/20261105T025730344698Z_pool
```

- **Input to stage 6:** `…_pool/feature_sets.json`. The second command copies the run's
  `feature_sets.json` and adds the full pool of random draws. The original run folder is unchanged.
- **Raw-activation center:** the Raw activations control is treated as a pseudo-SAE whose latent
  _i_ is neuron _i_ minus its mean over the `train` rows, so zeroing a latent is a mean ablation.
  That mean vector is `results/probe_mixed_final_001_layer25/centers/raw_layer25.npy`, the config's
  `path` for `kind: activations`. Feature selection writes it if it is missing and refuses to run if
  a saved file doesn't match. It sits in that folder only because the earlier feature-selection run
  (`configs/feature_selection/feature_selection_mixed_final_001_layer25.yaml`) wrote it first. `feature_sets.json`
  records its path and sha256, and stage 6 checks both and reads the file for the raw basis width.
  So moving it means rerunning feature selection; the contents would be the same.

### 6. Intervention

At layer 25, at the last prompt token of the A2 prefill, each episode is steered along the
targeted features and along matched random features. The outcome is whether Claude's label of the
regenerated A2 flips. The design, edit definitions and pre-registered analysis plan are in the
docstring of [scripts/intervention/run_intervention.py](scripts/intervention/run_intervention.py)
and in the config's `analysis` section.

```bash
python scripts/intervention/run_intervention.py choose-r --config configs/intervention/intervention.yaml   # pick edit size r
python scripts/intervention/run_intervention.py prepare  --config configs/intervention/intervention.yaml   # freeze run, baselines, shards
python scripts/intervention/run_intervention.py run      --config configs/intervention/intervention.yaml --shard K   # K = 0..3, one per GPU
python scripts/intervention/run_intervention.py merge    --config configs/intervention/intervention.yaml   # label, tables, figures
```

Each run gets its own folder, `results/intervention_layer25/<UTC time>-<git commit>/`. The config's
`run_id` selects which folder the later stages use.

| Run                                   | What it is                                                                                                                                                                                                                                                |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `20261106T000808809026Z-02871d677287` | **The study.** choose-r at r ∈ {0.02, 0.05, 0.10, 0.20}, then the full 1,000-episode study at r = 0.2. `merge/` holds the results: `summary.json`, `table_1.*`, `table_2_exploratory.*`, `figures/`. `frozen_run.json` records every input path and hash. |
| `20261106T030649156970Z-7064c87d114f` | Exploratory choose-r at r ∈ {0.3, 0.5}. No row met the rule, so r stayed 0.2.                                                                                                                                                                             |

A smoke run of the same pipeline: [configs/intervention/intervention_smoke.yaml](configs/intervention/intervention_smoke.yaml).

### 7. Analysis

- **Choosing r:** `notebooks/intervention/choose_edit_size_pilot_analysis.ipynb`
- **Main results:** `notebooks/intervention/full_study_results.ipynb`
- **What the features mean:** `scripts/intervention/interpret_features.py` →
  `results/feature_interpretation_layer25/`, shown in `notebooks/intervention/feature_interpretation.ipynb`
- **Exploratory, LLM-edited SAE:** `notebooks/intervention/llm_edited_sae_analysis.ipynb` (see below)

### Exploratory: LLM-edited SAE (not part of the pre-registered study)

**Question:** can an LLM improve the Pretrained SAE by hand, with no training, so its 20 selected
features steer better? Claude Opus 5 edited only those features' decoder rows (each row keeps its
length and a cosine ≥ 0.95 with the original) in a locked Docker sandbox, on train data. Two
controls make the same kind of edit without an LLM: a **mechanical** edit (rows rotated toward the
train mean-difference direction) and a **random** edit (the LLM edit's angles, random directions).

**Result (stage 6, r = 0.2):**
- **The edits make the features steer 3–5× harder:**
  - Pretrained SAE: Δ from +1.2 to +2.5 pp.
  - Mechanical edit: +6.2 to +6.4.
  - LLM edit: +8.5 to +9.5.
  - Random edit: +0.6 to +1.5.
- **The LLM edit beats the mechanical one** by +2.1 to +3.1 pp, but mostly by pushing harder: it also gives
  more off-target answers, mostly "The evidence is insufficient."
- **Where to read more:**
  - The notebook: the summary, the limits, and where this fits in the paper.
  - [cookbooks/LLM_EDITED_SAE.md](cookbooks/LLM_EDITED_SAE.md): the full design and decisions.

| Step                    | Command (`scripts/sae/`, unless shown)                                                          | Output                                                                                        |
| ----------------------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Train contexts          | `llm_edit_train_contexts.py`                                                                    | `artifacts/llm_edit/train_contexts.jsonl`                                                     |
| Sandbox inputs          | `llm_edit_sandbox_inputs.py`                                                                    | `artifacts/llm_edit/sandbox_inputs/`                                                          |
| Mechanical control      | `mechanical_edit_sae.py --steer-test`                                                           | `results/sae/mechanical_edit/20261111T211535990035Z-6a61cf6c6b31/`                            |
| Claude's edit (Bedrock) | `sg docker -c "PYTHONPATH=src python scripts/sae/llm_edit_sae.py"`                              | `results/sae/llm_edited/20261111T213251406629Z-c53c08f04ab9/`                                 |
| Random control          | `random_edit_sae.py --steer-test`                                                               | `results/sae/random_edit/20261111T234022791132Z-c53c08f04ab9/`                                |
| Audit                   | `audit_llm_edit.py`                                                                             | `results/sae/llm_edit_audit/20261111T235919947163Z-ea73ace51fec/`                             |
| Held-out `steer_test`   | `heldout_steer_test.py`                                                                         | `results/sae/llm_edit_heldout/20261112T011537298517Z-d90b8c2b79fd/`                           |
| Stage 5                 | `scripts/feature_selection/feature_selection.py --config configs/feature_selection/feature_selection_llm_edited_layer25.yaml`, then `llm_edit_feature_sets.py` | `results/feature_selection_llm_edited_layer25/20261112T002233925900Z_study_pool/`             |
| Stage 6                 | `run_intervention.py` (choose-r, prepare, run, merge) with `configs/intervention/intervention_llm_edited.yaml` | `results/intervention_llm_edited_layer25/20261112T005617641082Z-f48aae4eb4fa/`                |

- **Run order:** the mechanical control was built before Claude's edit, so it was fixed in advance.
  The random control copies the angles of Claude's edit, so it comes after it.
- **Environment:** run with `conda activate capstone` and `export PYTHONPATH=src`.
  - Claude's edit run needs Docker and Bedrock, and it costs money ($6.45).
  - Stage 6 labels with Bedrock (Sonnet 4.5, like the study).
  - Every other step uses the GPU or the CPU only.
- **Random pool:** stage 5 on another machine picks the same features but a slightly different
  random-control pool. `llm_edit_feature_sets.py` therefore gives the edited SAEs the study's
  Pretrained SAE pool, so their random-control generations reproduce the study's token for token.
- **Where the files are:**
  - Each edited SAE's record is in git: `edits.py`, `rows/`, `run.json`, `sae/config.json`, and the
    LLM run's `transcript.jsonl`.
  - Its `sae/params.safetensors` is gitignored and mirrored to `s3://dats-capstone/shared/group3/results/sae/`.
  - `artifacts/llm_edit/` is not on S3. To recreate it, rerun the first two steps. They make no Bedrock
    calls, and `sandbox_inputs/inputs.json` records the original sha256s to compare against.

### Earlier work, not part of the final results

The repository also holds earlier pipelines and runs. Don't cite them as final results:

- **Earlier collections:** `configs/archive/collection/v1`, `v2` (e.g. `natural_4b_full`) and
  `configs/collection/v3`.
- **Mini experiments:** side studies with their own question and answer, in `scripts/experiments/`
  and `configs/experiments/`: `adoption/` (feedback, instruction and anonymity, #138),
  `synthetic_pilot/` (#174, #179), `sae_improvement/` (resampling, refinement, weight L2),
  `critic_selection/` and `edit_size/`. Their results stay in their original `results/` folders.
- **Archived code:** pipelines the study replaced, in `scripts/archive/` (`lexical_labels_v01/`,
  `early_smoke_tests/`, `single_machine_collection/`) and `configs/archive/`.
- **The earlier causal pipeline:** `causal_replay`, `src/mas_sae/causal/`, `configs/causal_*.yaml`,
  `results/causal_*`. See [cookbooks/causal_replay/distributed_causal_replay.md](cookbooks/causal_replay/distributed_causal_replay.md).
  The stage-6 intervention replaced it. Exception: `mas_sae.causal.offline` is still used by
  feature selection (stage 5).
- **Earlier label versions:** `results/behavior/` contains older label packages besides `mixed_final_001_claude_probe`.

## Testing

```bash
pytest
```

`src/tests/unit/llm_edit/test_sandbox_docker.py` skips itself unless Docker is usable and
`artifacts/llm_edit/sandbox_inputs/` exists.

## Documentation

| Topic                                                             | Where                                                                                                                                                                                                                          |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Environment, Bedrock, Hugging Face, S3, Jupyter                   | [cookbooks/setup.md](cookbooks/setup.md)                                                                                                                                                                                       |
| Collection tooling, condition policy, earlier single-machine runs | [cookbooks/collection/collection.md](cookbooks/collection/collection.md)                                                                                                                                                                 |
| Distributed collection, end to end (collection → Claude labels)   | [cookbooks/runbooks/mixed_collection_pipeline.md](cookbooks/runbooks/mixed_collection_pipeline.md), [cookbooks/distributed_collection.md](cookbooks/distributed_collection.md)                                                 |
| Episode records, provenance, activation layout                    | [cookbooks/collection/episode_record.md](cookbooks/collection/episode_record.md), [cookbooks/collection/collection_provenance.md](cookbooks/collection/collection_provenance.md), [cookbooks/collection/activations.md](cookbooks/collection/activations.md) |
| Earlier rule-based labels (v0.1) and the human QC guide           | [cookbooks/labels/behavior_v01.md](cookbooks/labels/behavior_v01.md), [cookbooks/labels/qc_guide.md](cookbooks/labels/qc_guide.md)                                                                                             |
| SAE training, tuning and `sae_versions.csv`                       | [results/sae/README.md](results/sae/README.md)                                                                                                                                                                                 |
| Pretrained SAE fine-tuning                                        | [scripts/sae/PRETRAINED_FINETUNING.md](scripts/sae/PRETRAINED_FINETUNING.md)                                                                                                                                                   |
| Intervention design and analysis plan                             | docstring of [scripts/intervention/run_intervention.py](scripts/intervention/run_intervention.py)                                                                                                                              |
| Exploratory LLM-edited SAE: design, decisions, phases             | [cookbooks/LLM_EDITED_SAE.md](cookbooks/LLM_EDITED_SAE.md)                                                                                                                                                       |

## Project Structure

```text
.
├── artifacts/                    # Distributed collection outputs (gitignored; synced from S3)
├── configs/                      # Run configuration files, by pipeline stage
│   ├── data/                     # Synthetic data generation
│   ├── collection/               # Activation collection (final: distributed_final.yaml)
│   ├── layer_selection/          # Layer selection
│   ├── feature_selection/        # Feature selection (final: *_intervention_layer25.yaml)
│   ├── sae/                      # SAE training (final L1 SAE study)
│   ├── intervention/             # Intervention experiment
│   ├── experiments/              # Mini experiments (synthetic_pilot/, sae_improvement/)
│   ├── validation/               # Larger-model Natural Critic check
│   └── archive/                  # Configs of replaced pipelines and earlier runs
├── cookbooks/                    # Setup, how-to guides and reference notes, by stage: collection/, labels/,
│                                 #   sae/, causal_replay/; runbooks/ holds the collection runbook
├── data/                         # Downloaded project datasets (gitignored)
├── demo/                         # Demonstrations
│   └── fig/                      # Demo figures
├── docs/                         # Project documents
│   ├── diagrams/                 # Editable (.drawio) + SVG figures
│   └── research_paper/           # Research paper source and materials
│       ├── Latex/                # LaTeX paper source (figures come from reports/figures/)
│       ├── Word/                 # Word version of the paper
│       └── references/           # Reference PDFs for the literature review
├── documents/                    # Literature PDFs
├── examples/                     # Small Claude/Bedrock examples
├── notebooks/                    # All Jupyter notebooks, by stage: data/, collection/, labels/, sae/, causal_replay/, intervention/ (final results)
├── presentation/                 # Presentation materials
├── proposal/                     # Project proposal and its diagram
├── reports/                      # Project and progress reports
│   ├── figures/                  # Figures written by the scripts (the paper includes some)
│   ├── tables/                   # Tables written by the scripts
│   ├── Markdown_Report/          # Markdown-format report notes
│   ├── Progress_Report/          # Weekly progress reports
│   └── Word_Report/              # Word-format reports (e.g. literature review)
├── results/                      # Experimental outputs and results
├── scripts/                      # Entry-point scripts, by pipeline stage: data/, collection/, labels/,
│                                 #   layer_selection/, feature_selection/, sae/, intervention/;
│                                 #   experiments/ (mini experiments) and archive/ (replaced pipelines)
├── src/
│   ├── component/                # Reusable, non-pipeline components
│   ├── docs/                     # Project documentation
│   ├── mas_sae/                  # Reusable Python package
│   │   ├── activations/          # Activation capture and layer sites
│   │   ├── agents/               # Solver and critic agent logic
│   │   ├── causal/               # Earlier causal replay pipeline (offline.py still used by feature selection)
│   │   ├── data/                 # Dataset loading and processing
│   │   ├── evaluation/           # Evaluation functions, metrics and Claude labeling
│   │   ├── experiments/          # Collection pipeline, configs, and run artifacts
│   │   ├── intervention/         # The intervention experiment (stage 6)
│   │   ├── llm_edit/             # Exploratory LLM-edited SAE: sandbox, edit rules, steer_test
│   │   ├── models/               # Language-model loading and interaction
│   │   ├── probe/                # Layer selection, feature selection and probes
│   │   ├── sae/                  # Sparse-autoencoder functionality
│   │   └── utils/                # Timestamped run-folder helper
│   ├── shellscripts/             # Shell utility scripts
│   └── tests/                    # Unit and integration tests
├── environment.yml               # Conda environment and dependencies
├── pyproject.toml                # Python package configuration
├── pytest.ini                    # Pytest configuration
└── README.md                     # Experimental pipeline and quick start
```

The project structure may change as development progresses.
