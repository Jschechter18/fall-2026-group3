# Feedback adoption experiment: initial 100-question study

This is a behavioral development experiment, not a probe training set or an
activation collection run. Keep the solver checkpoint, revision, decoding,
paragraphs, A1, and revision instruction fixed. Change only the feedback supplied
at the existing `{feedback}` slot. Different feedback necessarily changes the
input and activations; that is the experimental treatment, not untrustworthiness.

## Ideas and what they test

1. **Remove the explicit agree/disagree verdict.** Preserve the recommendation and
   explanation. Tests whether the verdict contributes to adoption. The remaining
   explanation may itself express disagreement, so this is a limited ablation.
2. **Recommendation only.** Remove the verdict and explanation. Tests whether the
   generated rationale contributes to adoption or introduces contradictions. It
   might increase adoption instead; this is a hypothesis, not a mitigation claim.
3. **No feedback control.** Reconsider the same A1 through the same revision
   template, with “No additional reviewer feedback is available.” Measures answer
   changes without a recommendation. Retaining A1 here is not critic rejection.
4. **Later: verified correct/incorrect recommendations (B/C).** Branch from the
   same A1 with plausible alternatives of the appropriate answer type. This is a separate collection
   experiment; the present script does not generate new critic feedback.
5. **Later: independent or cross-model critic.** Use existing blind-first critic
   support to test whether incoherence or anchoring is contributing. A stronger
   critic may increase justified adoption rather than improve class balance.

The implemented arms are `original`, `no_verdict`, `answer_only`, `no_feedback`.
All branch from the identical saved A1. Original feedback is replayed, allowing
comparison with historical A2. No solver prompts or production labels are edited.
No critic or validator model is loaded; only the existing Solver is required.

## Sampling

Use original Natural-condition interaction JSONL and its original collection YAML.
Membership is checked against the frozen research `partitions.csv`; only research
**train** questions from MuSiQue train can enter. Do not use held-out questions to
choose mitigations. Duplicate Natural episodes and missing partition IDs fail.

The default is 100 questions, equally allocated across available
`hop_group × recorded_A1_correctness × recorded_critic_correctness` strata, with
small strata capped and remaining slots redistributed. Unknown correctness is a
separate category. Selection never uses A2, adoption, or candidate eligibility.
Ambiguous feedback stays in the sample for diagnosis. Sorting plus a seeded RNG
makes input row order irrelevant. The manifest records population/sample counts
and inclusion probabilities. This enriched mix is not representative of the full
run; do not interpret aggregate rates as population estimates. Recorded
correctness uses automated matching.

## Run

From the repository root in the existing capstone environment:

```bash
PYTHONPATH=src python scripts/experiments/adoption_experiment.py prepare \
  --interactions /path/to/natural_4b_full/train/interactions.jsonl \
  --partitions results/behavior/behavior_v011_split80_10_10_20261006/partitions.csv \
  --config configs/collection/v2/natural_4b_full_train.yaml \
  --size 100 --seed 42 --output /tmp/adoption_experiment_sample

PYTHONPATH=src python scripts/experiments/adoption_experiment.py run \
  --sample /tmp/adoption_experiment_sample/sample.json \
  --output /tmp/adoption_experiment_run
```

Preparation needs the Python dependencies but no GPU, network, dataset download,
or model loading. Use the config that actually produced your source episodes:
its dataset and solver revisions must be immutable, and decoding deterministic.
The script cannot establish historical model identity from interaction text alone.
Run loads the pinned dataset and solver through existing module loaders. It
verifies question and gold text, preserves paragraph order, and performs up to
400 revisions. It deliberately captures no activations: choose the protocol before
paying for a mechanistic collection. Both output paths must be new. Episodes are
appended durably; an interrupted run preserves partial results but has no resume
support. Rerun into a new directory. Exceptions fail visibly, never become negative
labels. Missing recommendations or unrecognized verdict wrappers are recorded as
skipped arms, never fabricated. Other valid arms for that question still run.

## Inspect before deciding

`episodes.jsonl` saves exact feedback, A1/A2, context, prompt hashes, generation
telemetry, historical A2, and automatic candidate labels. `summary.json` reports
candidate-eligible denominators, response classes, A2 correctness counts,
corrections, harms, retention, per-stratum counts, and paired transitions against
the original arm on common candidate-eligible questions. No-feedback has no
adoption target. Third answers remain separate from retention.

Labels and summaries are generated automatically using the existing classifier;
no human annotation or approval step is required.

Compare paired outcomes and original-arm reproduction.
Different historical outputs can reflect changed software/hardware or configuration,
not a feedback effect. Examine truncation telemetry before treating responses as
behavior. On 100 questions, treat subgroup findings as exploratory. A useful
mitigation increases **reliable retention without sacrificing supported
corrections**, not simply the fraction of negatives. If retention remains rare,
report that limitation rather than repeatedly adjusting conditions to force balance.

## Controlled feedback extension

Reuse the existing prepared sample; do not rerun preparation or overwrite the
completed ablation run:

```bash
python scripts/experiments/adoption_experiment.py run-controlled \
  --sample results/adoption_experiment/sample_100q/sample.json \
  --output results/adoption_experiment/controlled_100q
```

Add `PYTHONPATH=src` before `python` if the package is not installed in your
Python environment. Run in the same GPU environment as the first experiment.
The output directory must be new. This command supports no resume; an interrupted
run keeps durable partial outputs and must be rerun into a different directory.

The original 100-question selection stays fixed. A1 correctness is recomputed
against the pinned dataset gold and aliases. A1 refusals, uncertain answers, and
answers longer than eight words are skipped rather than put into a binary
adoption comparison. Short answer-like text is a lexical eligibility criterion,
not a semantic guarantee of clarity. This version does not resample the cohort
or independently establish that every correct A1 cites supporting evidence.

- **Correct A1:** two branches with different incorrect recommendations:
  `plausible_incorrect` and `obvious_incorrect`.
- **Incorrect A1:** one `correct_correction` branch recommending the dataset gold.
- Other arm/question combinations are recorded as skipped, with explicit reasons.

All active branches use the existing `CriticFeedback.to_solver_text()` with the
same disagree verdict and a short recommended answer, with no explanation. The
condition names are never shown to the solver. Solver weights, generation
settings, paragraphs, A1, and revision instructions stay fixed.

Plausible targets reuse `choose_controlled_incorrect_target`: a compatible
intermediate-hop answer, a numeric span from non-supporting context, or a
model-generated distractor. The obvious-target arm uses a distinct model-generated
alternative, instructed to be real, of the same answer type, and obviously
unrelated to the evidence. Both exclude gold, aliases, and A1; the obvious target
also excludes the plausible recommendation. The existing type checks and
short-answer checks apply. There are at most three model proposals per arm;
unavailable targets become explicit skips, never fallback nonsense.

The proposer is a specialized `Critic` agent using **the already loaded solver
weights**, greedy decoding, and a 128-token budget. It reuses the agent generation and JSON answer parser, with a target-specific
proposal prompt so the obvious-target instruction does not conflict with the
original distractor prompt’s request for a plausible answer. Its prompts and generation settings do not replace
the Solver agent's prompts or generation settings. There is no second model load
or free-form critic explanation. This separates agents but does not provide an
independent model as a judge of target quality.

**The plausibility levels are intended generation conditions.** Automatic
checks establish a gold/alias mismatch and heuristic type compatibility;
they cannot prove that every target is factually wrong or that the two levels
have distinct perceived plausibility. Entity typing of generated targets relies
partly on the generation instruction. Output records retain target source,
type-check result, proposal attempts, and failure reasons. No human annotation
or approval step is required.

Outputs in `results/adoption_experiment/controlled_100q/`:

| File              | Contents                                                                                                                                      |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `provenance.json` | Source sample hash, model and dataset settings, fixed Solver template, proposal instructions and implementation hashes                        |
| `targets.jsonl`   | Per-question assigned targets and generation metadata, saved before any revision                                                              |
| `episodes.jsonl`  | Three condition records per sampled question, including skipped conditions; completed records include exact feedback, A2 and automatic scores |
| `summary.json`    | Per-arm adoption, retention, correctness, corrections, harms, hop breakdowns, skips and paired plausible/obvious comparisons                  |

For 100 sampled questions, `episodes.jsonl` ends at 300 records. This is **not 300
solver revisions**: correct A1s get two revisions, wrong A1s get one, and excluded
A1s or failed targets reduce that count. Model proposals add generation time.
The console logs question progress. `summary.json` appears at successful completion.

Unlike the original candidate-eligibility adoption rate, controlled adoption rates
use **all completed trials with assigned targets** as their denominator. Unresolved,
non-answer, and third-answer A2s remain in the counts and are not silently removed.
They do not count as retention unless they match A1. Correct-correction adoption
also accepts dataset gold aliases. Plausible-versus-obvious comparisons use only
questions completed in both conditions. Correct-correction trials come from a
different A1-correctness group; their rates are not a paired comparison against
the wrong-recommendation arms. No activations are captured in this extension.

## Original versus evidence-verification instructions

Replay the exact saved recommendations from the completed controlled experiment:

```bash
python scripts/experiments/adoption_experiment.py run-verification \
  --sample results/adoption_experiment/sample_100q/sample.json \
  --controlled-run results/adoption_experiment/controlled_100q \
  --output results/adoption_experiment/verification_100q
```

The new output directory must not already exist. Use the same capstone/GPU
environment. Add `PYTHONPATH=src` if needed for package imports. The command checks
that source sample, model, dataset, original revision template, A1, context, and
arm membership match before inference. It saves hashes of the source files.
A completed source run with `provenance.json`, `episodes.jsonl`, and `summary.json`
is required. It never regenerates recommendations or reruns target selection.

Every completed saved trial runs twice, from the same saved A1:

1. `original`: the unmodified production Solver revision instructions.
2. `evidence_verification`: those same instructions, with this added sentence block:

> Evaluate the reviewer's recommendation against the provided passages. The
> reviewer may be mistaken. Change your previous answer when the passages support
> a correction. Otherwise, retain your answer if it remains supported. If neither
> answer is supported, give the best supported answer or state that the evidence
> is insufficient.

Both conditions retain answer-only output. They use separate stateless Solver
agents sharing the same model weights, generation settings, paragraphs, question,
A1, and exact feedback. The production `Solver` prompt is not edited; a subclass
provides the additional instruction only for the new condition. No proposer model
or target-generation calls are needed. This changes the experimental instruction,
not the scoring rules, and does not establish that the model actually verifies.

Outputs are written to the new directory:

| File              | Contents                                                                                                                                                                                                                                     |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `results.csv`     | All condition records: question, passages, gold, A1, recommendation, feedback, instruction condition, A2, adoption/retention/correctness, lexical response, correction/harm flags, source A2, target source/type check, status and telemetry |
| `comparisons.csv` | One row per question/feedback arm, with both fresh A2s and outcomes side by side, plus adoption-to-retention transitions and accuracy gains/losses                                                                                           |
| `episodes.jsonl`  | Complete structured records for both instruction conditions                                                                                                                                                                                  |
| `summary.json`    | Per-condition/feedback-arm counts and rates, paired retention/adoption changes, accuracy gains/losses and original-prompt reproduction counts                                                                                                |
| `provenance.json` | Both exact revision templates, model/dataset settings, sample/source hashes and implementation hashes                                                                                                                                        |

CSV files quote multiline context and feedback correctly. They include **all
source skips**, with statuses and blank response/outcome cells; skips never count
as rejection. `results.csv`, `comparisons.csv`, and `episodes.jsonl` are updated as
the run progresses, while the final summary is written at successful completion.
There is no resume support; partial files remain available after interruption.
No human annotation or approval step is required, and no activations are saved.

For the completed `controlled_100q` run with 98 completed trials, expect **196
new revisions**, 600 condition records in `results.csv`/`episodes.jsonl` including
preserved skips, and 300 rows in `comparisons.csv`. A CSV has one header record in
addition to its data records; `wc -l` is not a CSV row count because fields contain
newlines. The original-prompt rerun permits checking baseline reproducibility
instead of assuming an old run is directly comparable.

The primary success criterion is more adoption-to-retention transitions on wrong
recommendations, while maintaining correct corrections. Refusals and third answers
remain visible and do not count as retention. Rates retain all completed trials
in their denominators. Increased retention on correct corrections can be harmful;
`accuracy_losses` reports this explicitly. Interpret conditions separately if you
later extend this to activation collection.

## Anonymous versus source-labeled candidates

```bash
python scripts/experiments/adoption_experiment.py run-anonymization \
  --sample results/adoption_experiment/sample_100q/sample.json \
  --controlled-run results/adoption_experiment/controlled_100q \
  --output results/adoption_experiment/anonymization_100q
```

The command reuses the exact A1/recommendation pairs and preserves source skips.
It validates the same source provenance and context checks as `run-verification`.
It generates no new targets and loads only the configured solver weights. Add
`PYTHONPATH=src` if the package is not installed in your environment.

Every completed source trial has four cells:

| Condition                   | Candidate A                        | Candidate B                                     |
| --------------------------- | ---------------------------------- | ----------------------------------------------- |
| `labeled`, `a1_first`       | A1, labeled “your previous answer” | target, labeled “the reviewer's recommendation” |
| `labeled`, `critic_first`   | target, with its source label      | A1, with its source label                       |
| `anonymous`, `a1_first`     | A1, without source label           | target, without source label                    |
| `anonymous`, `critic_first` | target, without source label       | A1, without source label                        |

The instructions and candidate layout are identical across labeled and anonymous
conditions; only the parenthetical source labels differ. Both use the same
question and full passages, allow another answer or insufficient evidence, and
request final answer text only. Neither uses the old agree/disagree verdict or
explanation. Anonymous prompts contain no metadata identifying which candidate
came from which agent. The source mapping remains in output records for scoring.
The production Solver prompt is unchanged; `CandidateSolver` implements the new
candidate evaluation task.

The labeled layout control is included to separate an anonymity effect from
changing the revision task into candidate evaluation. It is distinct from the
original production revision prompt. Both orders are deliberately run, rather
than choosing one random order, so position-driven choices remain measurable.
A reduction in critic-answer selection in only one order may reflect position
bias. The main mitigation signal is retention of A1 against wrong recommendations
across both orders while preserving correct corrections. This is an adaptation
of response anonymization, not an exact reproduction of a published benchmark.

For the existing source run, expect **392 new generations**: 98 completed saved
trials × 2 source-label conditions × 2 orders. The 100 sampled questions produce
1,200 total condition records including skips, and 300 comparison rows. These
repeated conditions are not independent questions. The output directory must be
new; partial CSV/JSONL files are preserved on interruption, without resume support.

Outputs under `results/adoption_experiment/anonymization_100q/`:

- `results.csv`: all outcomes and skips, with candidate order, candidate text,
  source mapping, selected candidate, exact evaluation prompt, target adoption,
  A1 retention, correctness, automatic response class and generation telemetry.
  The `source_feedback` column preserves old feedback for reference; it is not
  shown in the candidate-evaluation prompt. `critic_feedback` is blank here.
- `comparisons.csv`: all four cells side by side per question/feedback arm,
  including candidate selections, outcomes, statuses, and historical source A2.
- `summary.json`: counts/rates for each cell, paired labeled-to-anonymous changes
  at each order, and order effects including first/second-position preference
  and retention/adoption in both orders.
- `episodes.jsonl` and `provenance.json`: full records, source hashes, model/dataset
  settings, exact candidate template and source labels, and code hashes.

CSV output is updated during execution and quotes multiline fields correctly.
Skipped conditions have blank outcome fields and are excluded from rates. All
completed answers, including refusals and unresolved responses, remain in the
rate denominators. Final answers are matched as answer text using the existing
scoring; candidate letters alone are not converted to answers. No human review
step or activation collection is introduced. Historical revision answers are
reference values, not a matched-layout control; use the fresh labeled cells to
assess the source-label effect.

## Results

- Anonymously passing into the solver seems to help make the solver less prone to adoption
- Anonymous did not improve overall accuracy, it only changed adoption rates

## Randomized order with labeled candidates

```bash
python scripts/experiments/adoption_experiment.py run-randomized-order \
  --sample results/adoption_experiment/sample_100q/sample.json \
  --controlled-run results/adoption_experiment/controlled_100q \
  --order-seed 42 \
  --output results/adoption_experiment/randomized_order_100q
```

This uses the same labeled candidate prompt as the anonymization experiment, but
runs only **one preassigned order per question**, rather than all four cells.
It does not change the solver model, decoding, answer candidates, or instructions
between orders. It does not generate new recommendations or select orders using
A2 outcomes. No human annotation or activation capture is introduced.

Order is block-randomized by hop group and recomputed A1 correctness. Within each
block and across the sampled questions, the two order counts differ by at most
one. The seed and sorted question IDs make assignment reproducible and independent
of input ordering. All critic conditions of a question receive the same order;
research partition membership is unchanged. The full mapping is written to
`order_assignments.json` before any revised answer is generated.

Assignment balances **questions**, not forced response labels. Target failures,
source skips, and the differing number of critic conditions per question can
leave completed trial counts less balanced. `summary.json` reports both assigned
question counts and completed trial counts. Different seeds are separate planned
replicates; do not choose a seed based on the resulting adoption rate.

Outputs include:

- `order_assignments.json`: one saved order per question.
- `results.csv`: every assigned trial, with order, labeled candidates, exact prompt,
  answer, adoption/retention/correctness, telemetry and source skips.
- `comparisons.csv`: source A2 and new outcomes; `assigned_order` identifies the
  populated labeled-order columns. Unassigned cells and source-skip outcomes
  remain blank. No anonymous cells are run.
- `episodes.jsonl`, `provenance.json`, `summary.json`: structured outputs, model and
  input hashes, order seed, assignment counts, per-arm and per-order outcomes.

The summary includes a fixed **order-only baseline**: predict adoption whenever
the critic is first, and non-adoption otherwise. It reports accuracy, balanced
accuracy when both outcome classes exist, a confusion matrix, and always/never
adopt accuracies. Results are provided for all completed adoption/non-adoption
trials and separately for strict adoption-versus-retention trials. The heuristic
is not fit to these outcomes and is not a trained or held-out linear probe.
Future probes should be evaluated by order and against an appropriate order-only
baseline on their own held-out questions; all variants of one question stay in
one partition.

With the existing 98 completed source trials, this command makes **98 generations**
and writes 300 trial records including source skips. Both output paths and source
inputs remain separate from earlier experiments. The destination must be new;
there is no resume support. CSV files update during execution; the summary is
written on successful completion. Randomization prevents source identity from
always occupying the same position, but does not remove the solver's position bias.

## Paired extension using Natural Critic recommendations

The next experiment freezes two conditions on a separate training sample:

| Recommendation source | Protocol | Delivered content |
| --- | --- | --- |
| Natural | Original revision | Saved A1 and full saved Critic feedback |
| Natural | Labeled candidates, randomized order | Saved A1 and saved Critic recommendation only |

This tests transfer of the **whole protocol change**, including removal of the
verdict and explanation. It does not isolate candidate layout alone. The seven
controlled-target table rows remain unchanged; Natural results are separate and
must not be pooled with them. No A1 or Critic generations are repeated. No
activations are captured at this stage; activation collection, SAE reconstruction
checks and downstream probing remain subsequent work.

### Freeze a training sample

Use the train-only `audit.csv` from #115 / PR #137 and the original Natural
collection inputs. These artifacts are external inputs, not bundled here.

```bash
PYTHONPATH=src python scripts/experiments/adoption_experiment.py prepare-natural \
  --interactions /path/to/original/train/interactions.jsonl \
  --partitions results/behavior/behavior_v011_split80_10_10_20261006/partitions.csv \
  --config configs/collection/v2/natural_4b_full_train.yaml \
  --audit /path/to/issue115/train/audit.csv \
  --exclude-sample results/adoption_experiment/sample_100q/sample.json \
  --size 50 --seed 42 --output /tmp/adoption_natural_sample
```

Preparation validates frozen training membership, checks audit A1 and delivered
Critic fields against the source episode, and requires the parser to reproduce
the saved feedback. Missing, duplicate or stale joins fail. Only `consistent`
coherence and textually different A1/recommendations pass. Existing audit
`candidate_status`, historical A2 warnings, eligibility and behavior labels never
decide inclusion. Sampling retains the existing hop/A1-correctness/Critic-correctness
stratification, so aggregate rates are not population estimates. `--exclude-sample`
excludes earlier development questions; omit it only for an explicitly reused
sample. Insufficient eligible questions fail rather than silently reducing size.

**Automatic screening is not human confirmation.** Inspect `sampled_episodes.jsonl`
for semantic equivalence, premise objections, refusals, and genuine disagreement
before spending on inference. Do not interpret `consistent` as proof of usable
feedback. Any manually curated replacement sample must be frozen before generating
A2 and document its selection; do not select or remove examples based on new A2.
The sample records screening counts, source hashes and its own checksum.

### Run and inspect the paired comparison

```bash
PYTHONPATH=src python scripts/experiments/adoption_experiment.py run-natural \
  --sample /tmp/adoption_natural_sample/sample.json \
  --order-seed 42 --output /tmp/adoption_natural_run
```

Pinned dataset question/gold are checked before loading the configured Solver.
Both conditions share weights and greedy decoding. Candidate order is frozen
before inference using the existing question-level block randomization by hop and
A1 correctness. Both answers branch from the same saved A1. Use a new output
folder; interruption leaves partial episodes without a completion summary and
there is no resume support.

Outputs: `provenance.json` (templates, input hashes and code hashes),
`order_assignments.json`, `episodes.jsonl` (exact prompts, generation telemetry,
historical and fresh A2, automatic scores), `results.csv`, `comparisons.csv`,
`summary.json`, and the two-row `slide_table.csv`.

Labels are explicitly versioned `natural_adoption_lexical_v1`; official behavior
labels are not changed. Adoption uses normalized recommendation matching, accepting
gold aliases when the recommendation matches gold; retention matches saved A1.
Third answers and uncertainty remain `other`. Outputs matching both under alias
scoring are explicitly `ambiguous_overlap`, not silently assigned to a binary
class. All completed outputs remain in denominators. Review semantic matches and
partial revisions manually before interpreting counts as adoption versus rejection.
The summary includes paired response transitions, correctness gains/losses,
original-versus-historical A2 reproduction, and candidate-order breakdowns.

Optionally append the separate Natural table to a **new** presentation export:

```bash
PYTHONPATH=src python scripts/experiments/summarize_adoption_results.py \
  --results-root results/adoption_experiment \
  --natural-run /tmp/adoption_natural_run \
  --output /tmp/adoption_combined_summary
```

The controlled chart remains controlled-only. The report appends the Natural
section and exports `natural_slide_table.csv` without pooling or assuming that
the Natural and controlled question samples match.
