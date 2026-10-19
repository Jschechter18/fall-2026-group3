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
   same A1 with plausible alternatives of the appropriate answer type. Independently
   review actual explanations and recommendations. This is a separate collection
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
correctness uses automated matching and also needs review.

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

**Automatic labels reuse the existing, known imperfect rules. They are diagnostic
only and cannot validate themselves.** Review `review.csv` using `docs/qc_guide.md`,
adding notes about irrelevant recommendations, contradictions, and truncation.
The file hides arm names and automatic labels, but feedback wording can reveal the
condition, so blinding is partial. Keep `review_private_key.json` from reviewers.
Two independent copies can be compared with `scripts/qc_agreement.py`; adjudicate
eligibility and outcome before drawing a class-balance conclusion. The no-feedback
control should be analyzed as reconsideration, outside the critic-adoption target.

Compare paired outcomes after review and inspect original-arm reproduction first.
Different historical outputs can reflect changed software/hardware or configuration,
not a feedback effect. Examine truncation telemetry before treating responses as
behavior. On 100 questions, treat subgroup findings as exploratory. A useful
mitigation increases **reliable retention without sacrificing supported
corrections**, not simply the fraction of negatives. If retention remains rare,
report that limitation rather than repeatedly adjusting conditions to force balance.
