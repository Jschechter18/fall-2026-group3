# Capstone Proposal

## SAE-MAS: Causal Analysis of Feedback Uptake in Solver-Critic Multi-Agent LLM Systems

### Proposed by: Josh Schechter

#### Advisor: Dr. Amir Jafari

#### The George Washington University, Washington DC

#### Data Science Program

---

## 1 Objective:

Large language model systems increasingly use multiple agents that critique, revise, or verify one another's outputs. A common architecture consists of a **Solver** that produces an initial answer and a **Critic** that evaluates the answer and provides feedback. The Solver can then revise its answer based on the Critic's response.

Although these systems are designed around the assumption that feedback can improve reasoning, considerably less is understood about **what changes internally within the Solver when it receives feedback and decides whether to accept or reject it**.

Sparse Autoencoders (SAEs) provide a potential mechanism for studying this process. SAEs decompose dense LLM activations into high-dimensional sparse feature representations. These sparse representations can be analyzed to identify features associated with particular model behaviors and, importantly, can be manipulated and decoded back into the LLM activation space to test whether those features are causally involved in the behavior.

The primary research question of this project is:

> **What internal representations change when an LLM Solver receives critic feedback, which sparse features predict whether the Solver accepts or rejects that feedback, and are those features causally involved in feedback uptake?**

The project will study this question using a controlled Solver-Critic system. The same initial Solver response will, where possible, be exposed to controlled correct and incorrect critic feedback. Solver activations will be collected before and after feedback, represented using Sparse Autoencoders, and analyzed using interpretable linear probe models.

Candidate SAE features identified by the predictive analysis will then be directly manipulated during subsequent Solver forward passes. If manipulating a feature systematically changes whether the Solver accepts or rejects critic feedback, this provides evidence that the feature is not merely correlated with feedback uptake but is causally involved in the mechanism.

A secondary objective is to compare a **general pretrained SAE** against an **interaction-specific SAE trained on activations collected during Solver-Critic interactions**. This tests whether an SAE trained specifically on the activation distribution produced during feedback processing better isolates features associated with feedback uptake.

### Key Objectives:

1. **Build a controlled Solver-Critic interaction pipeline and activation dataset.**
   - Run benchmark questions through a Solver-Critic-Solver interaction.
   - Cache selected Solver activations before and after critic feedback.
   - Generate controlled correct and incorrect critic feedback using known benchmark ground truth.
   - Record behavioral outcomes including feedback acceptance, answer changes, and final correctness.
   - Record labels such that we have 4 potential classes:
     - 0: Critic feedback helpful, solver accepts
     - 1: Critic feedback helpful, solver rejects
     - 2: Critic feedback harmful, solver accepts
     - 3: Critic feedback harmful, solver rejects

2. **Represent Solver activations using Sparse Autoencoders.**
   <!-- - Load a compatible pretrained SAE for the selected LLM and activation site. -->
   - Train an interaction-specific SAE on cached Solver activations from the Solver-Critic dataset.
   - Evaluate SAE reconstruction quality and sparsity before using the representation for downstream analysis.

3. **Identify sparse features predictive of feedback uptake.**
   - Encode Solver activations into SAE latent feature vectors.
   - Train regularized linear probe models using the latent feature vectors to classify which classes (described in 1) the Solver will yield (basically only cares about accepting or rejecting).
   - Use probe coefficients and held-out validation to select a small set of candidate features associated with feedback uptake (how the solver incorporates the critics feedback).
   - Compare predictive structure discovered by the pretrained and interaction-trained SAEs.

4. **Test whether selected sparse features are causally involved in feedback uptake.**
   - Suppress or amplify selected SAE features during the Solver's processing of critic feedback.
   - Decode the modified SAE representation back into the LLM activation space.
   - Continue the Solver forward pass using the modified activation.
   - Measure whether the intervention changes the probability that the Solver accepts critic feedback.

5. **Characterize when feedback-related internal mechanisms produce useful versus harmful behavior.**
   - Compare interventions under correct and incorrect critic feedback.
   - Determine whether selected features correspond to useful correction uptake, general willingness to revise, resistance to feedback, or other feedback-related behavior.
   - Package the interaction pipeline, activation collection, SAE training/loading, probe analysis, and causal intervention experiments into a reproducible repository.

---

## 2 Dataset:

The project requires two related forms of data:

1. an existing benchmark containing questions with objectively known answers, and
2. a generated Solver-Critic interaction dataset containing behavioral outcomes and Solver activations.

The interaction dataset will be generated by the project rather than obtained from an existing multi-agent benchmark.

### TIER 1 -- BASE REASONING DATASET:

**MuSiQue**

MuSiQue is a multi-hop question-answering benchmark that requires models to combine information across multiple reasoning steps.

Dataset:

https://huggingface.co/datasets/dgslibisey/MuSiQue

MuSiQue provides the underlying questions and known answers required to objectively evaluate the Solver's initial and revised responses.

Additional objectively scored reasoning datasets may be considered if pilot experiments show that MuSiQue does not produce sufficient variation in Solver correctness or feedback acceptance. The final benchmark set should be frozen after the initial pilot.

### TIER 2 -- GENERATED SOLVER-CRITIC INTERACTION DATASET:

Each benchmark question will first be passed through the Solver to generate an initial answer. The same initial Solver response will then be reused across multiple Critic conditions where possible. This creates matched interactions in which the question, initial answer, and initial Solver reasoning remain fixed while the feedback provided to the Solver changes.

The dataset will contain one natural Solver-Critic condition and two controlled feedback conditions.

#### Condition A -- Natural Critic Feedback

The Critic independently evaluates the Solver's initial response and provides whatever feedback it determines is appropriate. The Critic is not instructed to agree or disagree with the Solver and is not given a predetermined answer to advocate.

This condition represents the normal Solver-Critic interaction and will serve as the project's primary observational dataset.

After the interaction, the known benchmark ground truth will be used to determine whether the Critic's feedback was correct/helpful or incorrect/harmful. The Solver's second response will then be evaluated to determine whether it accepted or rejected the Critic's feedback.

This produces naturally occurring examples of:

1. Correct/helpful feedback + Solver accepts
2. Correct/helpful feedback + Solver rejects
3. Incorrect/harmful feedback + Solver accepts
4. Incorrect/harmful feedback + Solver rejects

These interactions will be used to investigate which internal Solver representations are associated with feedback acceptance and rejection under normal multi-agent behavior.

#### Condition B -- Controlled Correct Feedback

The same initial Solver response will be provided to a Critic instructed to advocate for the known ground-truth answer.

This creates a controlled condition in which the feedback presented to the Solver is known to support the correct answer.

This condition measures how the Solver responds when exposed to useful feedback and provides a controlled comparison against the natural Critic condition.

#### Condition C -- Controlled Incorrect Feedback

The same initial Solver response will be provided to a Critic instructed to advocate for a deliberately incorrect answer.

This creates a controlled condition in which the feedback presented to the Solver is known to support an incorrect answer.

This condition measures the Solver's susceptibility to misleading feedback and provides a matched comparison with the controlled correct-feedback condition.

### Paired Experimental Design

Where technically feasible, all three Critic conditions will branch from the same initial Solver response:

`Solver Attempt 1 → Natural Critic → Solver Attempt 2A`

`Solver Attempt 1 → Controlled Correct Critic → Solver Attempt 2B`

`Solver Attempt 1 → Controlled Incorrect Critic → Solver Attempt 2C`

This design holds the original question and Solver Attempt 1 constant while changing the feedback presented to the Solver.

The natural condition provides the primary dataset for studying feedback behavior as it occurs organically within the Solver-Critic system. The controlled conditions provide experimental comparisons that allow the project to determine how feedback correctness affects the Solver's behavior and internal representations.

### Interaction Record

Each feedback event will contain at minimum:

- `question_id`
- `question`
- `ground_truth`
- `solver_attempt_1`
- `solver_attempt_1_correct`
- `critic_condition`
- `critic_feedback`
- `critic_feedback_correct`
- `solver_attempt_2`
- `solver_attempt_2_correct`
- `solver_changed_answer`
- `solver_accepted_feedback`
- `activation_attempt_1`
- `activation_attempt_2`

The primary behavioral target will be:

`solver_accepted_feedback ∈ {0,1}`

Critic correctness will be recorded separately rather than being combined with Solver acceptance into the primary target. This allows the project to distinguish **whether the Solver accepts feedback** from **whether accepting that feedback was appropriate**.

_Important_: The four combinations (2 x 2 table) of feedback correctness and Solver acceptance will still be retained for descriptive analysis:

1. Correct feedback + Solver accepts
2. Correct feedback + Solver rejects
3. Incorrect feedback + Solver accepts
4. Incorrect feedback + Solver rejects

### TIER 3 -- ACTIVATION DATASET:

Selected internal Solver activations will be cached during:

1. **Solver Attempt 1:** before critic feedback
2. **Solver Attempt 2:** while the Solver processes critic feedback and produces its revised answer

The exact transformer layer and activation site will be selected during the initial technical feasibility stage.

The project will initially target **one activation site at one layer**. A second layer may be added only if compute and schedule permit.

These cached activations serve two purposes:

- training the interaction-specific SAE;
- generating SAE latent representations for predictive analysis.

Because both pre-feedback and post-feedback activations are collected, the project can additionally investigate changes in sparse representation such as:

`Δz = z_after_feedback - z_before_feedback`

This provides a direct representation of how the Solver's internal state changes after exposure to critic feedback.

### DATA SPLITS:

Questions must be divided into separate experimental splits before feature selection:

- **Discovery / training split:** SAE training and probe-based feature discovery
- **Validation split:** confirmation that selected features predict feedback uptake on unseen interactions
- **Intervention / test split:** causal intervention experiments

Questions from the intervention/test split must not be used to select candidate SAE features.

This separation prevents the causal experiment from being evaluated on the same interactions used to discover the candidate features.

---

## 3 Rationale:

Multi-agent LLM systems commonly rely on interaction patterns such as critique, debate, verification, and revision. These architectures assume that exposing one model to another agent's feedback can improve the final response.

However, observing that a Solver changes its answer after criticism does not explain **how the feedback is internally processed**.

The Solver may:

- recognize that the critic has identified an error;
- detect disagreement and reconsider its previous reasoning;
- defer to another agent regardless of whether that agent is correct;
- resist external feedback;
- or respond through a distributed mechanism that cannot be represented by a small number of interpretable features.

Behavioral evaluation alone cannot distinguish these possibilities.

Sparse Autoencoders provide a useful framework for investigating this question because they transform dense transformer activations into sparse latent feature representations. Instead of attempting to interpret thousands of dense activation dimensions directly, the project can analyze which sparse features change during feedback processing and which features are predictive of subsequent Solver behavior.

However, **predictive association is not sufficient to establish mechanism**.

A sparse feature may reliably predict feedback acceptance without causing it. The feature could be downstream of the actual mechanism or correlated with another internal process.

The central methodological contribution of this project is therefore the transition from:

> **feature detection:** "This sparse feature is predictive of feedback acceptance."

to:

> **causal intervention:** "Manipulating this sparse feature changes the Solver's probability of accepting feedback."

This distinction is critical for mechanistic interpretability.

The controlled correct-versus-incorrect feedback design further allows the project to distinguish different possible interpretations of a candidate feature.

For example, suppose amplifying a feature increases acceptance of **both correct and incorrect feedback**. The feature may represent general deference or willingness to revise rather than the ability to recognize useful feedback.

Conversely, if manipulating a feature selectively changes acceptance of correct criticism while having little effect on false criticism, the feature may be associated with a more discriminative feedback-evaluation mechanism.

### PRETRAINED SAE VS. INTERACTION-SPECIFIC SAE:

A secondary research question concerns whether general-purpose SAE representations are sufficient for studying multi-agent interactions.

A pretrained SAE has been trained on a broad activation distribution and may already contain features relevant to disagreement, revision, confidence, or correction.

The interaction-specific SAE will instead be trained directly on Solver activations generated during the controlled feedback experiment.

Importantly, this SAE remains **unsupervised**. It is not explicitly trained to identify "feedback acceptance." Rather, it learns a sparse dictionary over the activation distribution produced while the Solver participates in Solver-Critic interactions.

The comparison therefore asks:

> **Does an interaction-specific SAE produce sparse representations that better isolate feedback-related behavioral structure than a general pretrained SAE?**

The same downstream probe and causal intervention protocol can be applied to both representations.

### EXPECTED CONTRIBUTION:

The project aims to provide:

1. An analysis of sparse internal features associated with feedback acceptance and rejection.
2. A comparison between general pretrained and interaction-specific SAE representations.
3. Controlled causal evidence testing whether selected SAE features actually influence feedback uptake.
4. A reproducible framework for studying internal mechanisms of agent-to-agent feedback in multi-agent LLM systems.

A negative result remains scientifically meaningful.

---

## 4 Approach:

### PHASE 1: PROJECT CONCEPTUALIZATION & EXPLORATORY ANALYSIS

#### [Weeks 1-2: Research Design, Literature Review, and Dataset Exploration]

- Finalize the primary research question and experimental objectives.
- Review relevant literature on:
  - Sparse Autoencoders and mechanistic interpretability;
  - SAE feature intervention and causal validation;
  - Solver-Critic and multi-agent LLM systems;
  - LLM response to external feedback and critique.
- Define the Solver-Critic architecture and interaction protocol at a conceptual level.
- Define the primary behavioral outcome: whether the Solver accepts or rejects Critic feedback.
- Define the 2 × 2 behavioral outcome matrix:

|                               | Solver Accepts Feedback | Solver Rejects Feedback |
| ----------------------------- | ----------------------- | ----------------------- |
| **Correct Critic Feedback**   | Correct + Accept        | Correct + Reject        |
| **Incorrect Critic Feedback** | Incorrect + Accept      | Incorrect + Reject      |

- Define the three primary Critic conditions:
  - natural/uncontrolled Critic feedback;
  - controlled correct feedback;
  - controlled incorrect feedback.
- Perform exploratory analysis of MuSiQue and determine:
  - dataset size and structure;
  - available ground-truth fields;
  - question and answer formats;
  - whether answers can be scored reliably;
  - whether the dataset is appropriate for generating controlled incorrect answers;
  - preprocessing or filtering requirements.
- Identify candidate open-weight LLMs based on:
  - instruction-following capability;
  - availability of compatible pretrained SAEs;
  - ability to extract and intervene on internal activations;
  - computational requirements.
- Identify candidate pretrained SAEs and compatible activation sites.
- Define the initial experimental controls, evaluation metrics, and dataset-splitting strategy.
- Produce the finalized experimental design and initial system architecture.

**Milestone:** By the end of Week 2, the research question, experimental design, dataset, behavioral outcomes, candidate model/SAE configuration, and evaluation strategy should be sufficiently defined to begin implementation.

---

### PHASE 2: PARALLEL SYSTEM DEVELOPMENT

#### [Weeks 3-5: Solver-Critic and SAE Development]

Following completion of the experimental design, development will proceed through two parallel workstreams.

#### Workstream A -- Solver-Critic System and Activation Collection

Two team members will focus on implementing the multi-agent experimental system and the pipeline required to collect the internal activations that will eventually be used to train and evaluate the SAEs.

- Implement reusable Solver and Critic components.
- Implement the basic interaction:

  `Question → Solver Attempt 1 → Critic Feedback → Solver Attempt 2`

- Implement the three Critic conditions:
  - natural/uncontrolled Critic feedback;
  - controlled correct feedback;
  - controlled incorrect feedback.
- Implement benchmark loading, preprocessing, and answer scoring.
- Implement interaction logging and behavioral labeling.
- Implement forward hooks for extracting Solver activations from the selected model layer and activation site.
- Verify activation collection during Solver Attempt 1 and Solver Attempt 2.
- Develop the pipeline for storing activation tensors separately from interaction metadata.
- Integrate the selected pretrained SAE.
- Verify that Solver activations can be encoded and reconstructed using the pretrained SAE.
- Measure reconstruction error.
- Develop the mechanism required to inject reconstructed activations back into the Solver.
- Perform an initial SAE intervention proof of concept:

  `Solver Activation → SAE Encode → Modify Latent → SAE Decode → Inject Activation → Continue Generation`

#### Workstream B -- Interaction-Specific SAE Development

One team member will simultaneously develop the custom SAE and its training infrastructure.

- Implement the selected SAE architecture.
- Implement:
  - SAE encoding;
  - SAE decoding;
  - reconstruction loss;
  - sparsity regularization;
  - training and validation loops;
  - checkpointing;
  - experiment configuration;
  - training diagnostics.
- Develop metrics for monitoring:
  - reconstruction error;
  - sparsity;
  - dead-feature rate;
  - feature activation frequency;
  - training stability.
- Test the SAE pipeline using temporary or sample activation tensors with the same expected dimensionality as the final Solver activations.
- Establish initial SAE hyperparameters and training configuration.
- Implement efficient loading of activation vectors generated by the Solver-Critic pipeline.
- Prepare the SAE training pipeline so that training can begin once sufficient real Solver activations have been collected.
- If there are available activations, we can start training immediately.

The custom SAE will remain an **unsupervised representation-learning model**. Feedback-acceptance labels will not be used during SAE training.

#### Integration

As the two workstreams mature, the custom SAE pipeline will be tested on a small sample of actual Solver activations.

This integration will verify:

- activation dimensionality and formatting;
- SAE input compatibility;
- reconstruction behavior;
- activation storage and loading;
- end-to-end SAE encoding and decoding.

**Milestone:** By the end of Week 5, the Solver-Critic activation pipeline and custom SAE training pipeline should both be operational and compatible. The team should also have demonstrated that SAE-based activation intervention is technically feasible using the selected model and activation site.

---

### PHASE 3: FULL INTERACTION & ACTIVATION DATASET GENERATION

#### [Week 6: Integreation Full Forward-Pass Data Collection]

Run the finalized Solver-Critic experiment across the selected benchmark questions and Critic conditions.

For each benchmark question:

1. Generate Solver Attempt 1.
2. Evaluate the initial Solver answer against ground truth.
3. Cache the selected Solver activations.
4. Run the natural/uncontrolled Critic condition.
5. Generate and evaluate the corresponding Solver Attempt 2.
6. Run the controlled correct-feedback condition using the same initial Solver response.
7. Generate and evaluate the corresponding Solver Attempt 2.
8. Run the controlled incorrect-feedback condition using the same initial Solver response.
9. Generate and evaluate the corresponding Solver Attempt 2.
10. Cache the required post-feedback Solver activations.
11. Record Critic correctness, feedback acceptance/rejection, answer changes, and final correctness.
12. Assign each interaction to the appropriate cell of the 2 × 2 behavioral outcome matrix.

Behavioral metadata will be stored separately from high-dimensional activation tensors and linked through stable episode identifiers.

The resulting activation data will provide the inputs required for both the pretrained SAE analysis and interaction-specific SAE training.

Only activation data belonging to the **discovery/training split** will be used to train the custom SAE.

---

### PHASE 5: INTERACTION-SPECIFIC SAE TRAINING & REPRESENTATION GENERATION

#### [Weeks 7-10: SAE Training and Validation]

Once sufficient Solver activations have been collected, training of the interaction-specific SAE will begin using the training infrastructure developed during Weeks 3-5.

#### Interaction-Specific SAE

- Train the SAE on Solver activation vectors from the discovery/training split.
- Monitor:
  - training and validation reconstruction loss;
  - sparsity;
  - dead-feature rate;
  - feature activation frequency;
  - training stability.
- Tune SAE hyperparameters where necessary.
- Save intermediate checkpoints and training diagnostics.
- Evaluate reconstruction quality on held-out activation data.
- Generate sparse representations for the collected Solver-Critic interactions.

The experimental sequence will remain explicitly separated:

`Unsupervised SAE Training → Supervised Feature Discovery → Causal Intervention`

**Milestone:** By the end of Week 10, the team should have a validated interaction-specific SAE and corresponding sparse representations from both the pretrained and interaction-specific SAEs. This concludes the SAE modeling component.

---

### PHASE 6: PREDICTIVE FEATURE DISCOVERY

#### [Weeks 10-11: Predictive Modeling and Feature Selection]

Use the sparse SAE representations to identify internal features associated with whether the Solver accepts or rejects Critic feedback.

The primary behavioral target will be:

`solver_accepted_feedback ∈ {0,1}`

Critic feedback correctness will remain a separate experimental variable rather than being combined with acceptance into a four-class prediction target.

- Train regularized linear classifiers, such as L1 or elastic-net logistic regression, using SAE latent activations as input features.
- Evaluate predictive performance using:
  - AUROC;
  - F1 score;
  - balanced accuracy.
- Analyze predictive performance:
  - across all interactions;
  - under correct feedback;
  - under incorrect feedback;
  - under natural Critic feedback.
- Compare predictive performance between:
  - pretrained SAE features;
  - interaction-specific SAE features.
- Identify candidate features using:
  - coefficient magnitude;
  - coefficient direction;
  - activation frequency;
  - stability across data subsets or random seeds.
- Analyze changes in SAE feature activation before and after Critic feedback where appropriate.

For selected feature \(i\), this may include:

`Δz_i = z_i(after feedback) - z_i(before feedback)`

where \(z_i\) represents the activation of SAE feature \(i\).

#### Held-Out Validation

Before causal intervention:

- Evaluate candidate predictive features on the held-out validation split.
- Determine whether associations between candidate features and feedback acceptance generalize beyond the discovery data.
- Select a small final set of features for causal testing.

Features will not be selected using the intervention/test split.

**Milestone:** By the end of Week 11, the team should have a small validated set of SAE features associated with feedback acceptance or rejection and ready for causal testing.

---

### PHASE 7: CAUSAL SAE INTERVENTION

#### [Weeks 12-13: Feature Suppression and Amplification]

This is the final step of our experiment. The primary causal experiment will test whether features identified during predictive analysis actually influence how the Solver responds to Critic feedback.

For each selected feature:

1. Run the Solver to the selected intervention point.
2. Capture the relevant internal activation.
3. Encode the activation using the SAE.
4. Modify the selected SAE latent.
5. Decode the modified sparse representation.
6. Inject the reconstructed activation back into the Solver.
7. Continue generation.
8. Measure whether Solver feedback acceptance changes.

#### Intervention Conditions

Selected features will be evaluated using:

- **Target-feature suppression:** decrease or remove activation of the selected feature.
- **Target-feature amplification:** increase activation of the selected feature.
- **Intervention magnitude sweep:** evaluate multiple intervention strengths where feasible.

#### Required Controls

The causal experiment will include:

1. **No-intervention baseline**  
   Run the Solver normally without SAE reconstruction or feature modification.

2. **SAE reconstruction-only control**  
   Encode and decode the activation without deliberately modifying a feature. This controls for behavioral changes caused by SAE reconstruction error.

3. **Random-feature intervention control**  
   Apply matched interventions to unrelated SAE features to test whether observed effects are specific to the selected feature rather than a generic consequence of perturbing the SAE representation.

Where feasible, random features will be matched on characteristics such as activation frequency or typical activation magnitude.

#### Primary Outcome

The primary causal outcome will be the change in Solver feedback acceptance under intervention relative to the appropriate control condition.

#### Secondary Outcomes

Secondary outcomes will include:

- final-answer correctness;
- answer-change rate;
- acceptance of correct feedback;
- acceptance of incorrect feedback;
- rejection of correct feedback;
- resistance to incorrect feedback;
- generation coherence and output quality.

Intervention effects will be analyzed separately under correct and incorrect Critic feedback.

For example:

- if amplifying a feature increases acceptance of both correct and incorrect feedback, the feature may represent general deference or willingness to revise;
- if amplification selectively increases acceptance of correct feedback, the feature may be involved in evaluating feedback quality;
- if suppressing a feature reproducibly decreases acceptance relative to matched controls, this provides evidence that the feature is causally involved in feedback acceptance rather than merely correlated with it.

---

### PHASE 8: ANALYSIS & ROBUSTNESS

#### [Week 14: Final Statistical Analysis and Interpretation]

- NOTE: this really should be more about interpretation and analysis. If we did a good job implementing, when we run the experiment the tests will populate csv's with the data we want to see.

- Compare behavioral outcomes across:
  - natural Critic interactions;
  - controlled correct feedback;
  - controlled incorrect feedback.
- Compare pre-feedback and post-feedback sparse representations.
- Compare accepted and rejected feedback interactions.
- Compare predictive feature strength with measured causal intervention effects.
- Compare pretrained and interaction-specific SAE representations.
- Quantify intervention effects relative to:
  - no-intervention behavior;
  - SAE reconstruction-only controls;
  - random-feature controls.
- Evaluate whether intervention effects remain consistent across intervention strengths and data subsets.
- Analyze whether identified features appear related to:
  - general feedback acceptance;
  - resistance to feedback;
  - sensitivity to feedback correctness;
  - or broader behavioral changes unrelated to feedback.
- Report null or inconsistent intervention results explicitly rather than interpreting predictive association alone as evidence of mechanism.

The central distinction will be between:

**Observation:** a sparse feature is associated with Solver feedback acceptance.

and

**Causal evidence:** manipulating that feature produces a reproducible change in Solver feedback acceptance relative to appropriate controls.

**Visualization prep** by this week, we should basically be done. We should be focused on ensuring all of our visualizations for the paper are ready.

---

### PHASE 9: FINAL PAPER & REPRODUCIBILITY

#### [Week 15: Final Documentation, Paper, and Presentation]

- Finalize statistical analyses and figures.
- Document:
  - model configuration;
  - Solver and Critic prompts;
  - Critic conditions;
  - benchmark preprocessing;
  - activation site and layer;
  - pretrained SAE configuration;
  - custom SAE architecture and training procedure;
  - predictive models;
  - feature-selection procedure;
  - intervention magnitudes;
  - random seeds;
  - experimental controls.
- Prepare final tables and visualizations showing:
  - the 2 × 2 behavioral outcome distribution;
  - predictive feature performance;
  - pretrained versus interaction-specific SAE results;
  - causal intervention effects;
  - correct versus incorrect feedback behavior.
- Final organization checks for the codebase into reproducible components for:
  - Solver-Critic orchestration;
  - benchmark processing;
  - interaction generation;
  - activation caching;
  - SAE loading and training;
  - predictive modeling;
  - feature intervention;
  - evaluation and statistical analysis.
- Complete the final capstone paper.
- Prepare the final presentation and project demonstration.

## 5 Timeline:

**Weeks 1-2:** Proposal development, research conceptualization, literature review, dataset selection, EDA, and experimental design.

**Week 3:** Begin parallel development:

- Solver-Critic system, benchmark pipeline, and activation-extraction infrastructure;
- interaction-specific SAE architecture and training infrastructure.

**Week 4:** Continue parallel development. Integrate Solver/Critic conditions, behavioral scoring, activation extraction and storage, SAE training components, and pretrained SAE support. Begin testing the custom SAE on sample or available real activations.

**Week 5:** Complete core Solver-Critic and SAE infrastructure. Perform end-to-end integration and validation of interaction generation, activation extraction, SAE encoding/reconstruction, and activation intervention. Begin preliminary SAE training if sufficient real activations are available.

**Week 6:** Run the finalized Solver-Critic experiment and generate the full behavioral and activation datasets. Finalize discovery/training, validation, and intervention/test splits. Prepare finalized training activations for the interaction-specific SAE.

**Week 7:** Begin full interaction-specific SAE training and hyperparameter evaluation. In parallel, continue development of downstream predictive modeling, feature-selection, causal-intervention, control, and evaluation pipelines.

**Week 8:** Continue interaction-specific SAE training and evaluate reconstruction quality, sparsity, dead-feature rate, and training stability. Generate pretrained SAE representations in parallel. Continue development and testing of downstream experimental infrastructure.

**Week 9:** Finalize and validate the interaction-specific SAE. Generate sparse representations of the Solver-Critic interactions using both pretrained and interaction-specific SAEs. Begin predictive feature analysis as validated representations become available.

**Week 10:** Train regularized predictive models, analyze feedback-related SAE features, compare pretrained and interaction-specific SAE representations, and identify candidate features associated with feedback acceptance and rejection.

**Week 11:** Perform held-out validation of candidate SAE features and select the final feature set for causal testing. Finalize causal intervention conditions, reconstruction-only controls, random-feature controls, and intervention-strength configurations.

**Week 12:** Run target-feature suppression and amplification experiments across correct and incorrect Critic-feedback conditions.

**Week 13:** Complete causal intervention experiments and associated controls. Compare intervention effects across feedback conditions, intervention strengths, and SAE representations. Begin final interpretation and visualization of experimental results.

**Week 14:** Complete statistical analysis and robustness checks. Focus on interpretation of predictive and causal results and finalize tables, figures, and visualizations for the paper.

**Week 15:** Complete the research paper, repository documentation, reproducibility checks, final presentation, and project demonstration.

**TOTAL: 15 weeks**

### KEY MILESTONES:

- **Week 2:** Research question, dataset, eda experimental design, candidate model/SAE configuration, and evaluation strategy finalized.
- **Week 5:** Solver-Critic system, activation pipeline, SAE training pipeline, and end-to-end SAE intervention mechanism operational.
- **Week 6:** Full behavioral and activation datasets generated and experimental splits finalized.
- **Week 9:** Interaction-specific SAE trained and validated; pretrained and interaction-specific sparse representations available for analysis.
- **Week 11:** Candidate feedback-related SAE features selected using held-out validation.
- **Week 13:** Causal feature-intervention experiments and controls complete.
- **Week 14:** Primary statistical analyses, interpretation, and research figures complete.
- **Week 15:** Final paper, reproducible codebase, and presentation complete.

### DELIVERABLES BY WEEK 15:

- Reproducible Solver-Critic experimental framework.
- Behavioral dataset containing natural, controlled-correct, and controlled-incorrect Critic interactions.
- Solver activation dataset collected before and after Critic feedback.
- Pretrained SAE representations of Solver-Critic activations.
- Interaction-specific SAE trained on Solver-Critic activations.
- Predictive feature-selection analysis of feedback acceptance and rejection.
- Held-out evaluation of feedback-related sparse features.
- Controlled SAE feature-intervention experiments, including reconstruction-only and random-feature controls.
- Quantitative analysis of whether selected SAE features are causally involved in feedback acceptance.
- Comparison of pretrained and interaction-specific SAE representations.
- Final research paper/report.
- Documented and reproducible GitHub repository.
- Final project presentation and demonstration.

## 6 Expected Number Students:

**RECOMMENDED: 3 STUDENTS**

### STUDENT 1 -- SOLVER-CRITIC SYSTEM, DATA PIPELINE, AND EXPERIMENT INFRASTRUCTURE

Primary responsibility:

- Design and implement the modular Solver-Critic pipeline.
- Implement benchmark ingestion and evaluation.
- Implement controlled correct/incorrect critic-feedback generation.
- Implement interaction logging and behavioral-label generation.
- Build the activation-caching infrastructure.
- Maintain experiment configuration and reproducibility.
- Support the causal-intervention pipeline by ensuring the forward-pass infrastructure can be reused during Objective 4.

This role owns the foundation of the project. The remaining experiments depend on the interaction and activation datasets produced by this pipeline.

All team members should be involved in designing and validating the controlled interaction protocol because errors in this stage would affect the validity of every downstream experiment.

---

### STUDENT 2 -- SPARSE AUTOENCODER TRAINING AND REPRESENTATION ANALYSIS

Primary responsibility:

- Integrate the pretrained SAE.
- Determine compatible activation sites and layers with the team.
- Implement the interaction-specific SAE.
- Train the SAE on cached Solver activations.
- Evaluate reconstruction loss, sparsity, dead features, and training stability.
- Produce SAE latent representations for downstream predictive modeling.
- Compare pretrained and interaction-specific SAE representations.
- Support intervention experiments requiring SAE encoding, latent manipulation, and decoding.

This is expected to be one of the most technically difficult components because poor SAE reconstruction or unstable sparse representations could compromise downstream feature discovery and intervention.

---

### STUDENT 3 -- PREDICTIVE MODELING, FEATURE SELECTION, AND CAUSAL INTERVENTION

Primary responsibility:

- Build the linear-probe modeling pipeline.
- Train regularized logistic regression models predicting feedback acceptance.
- Evaluate AUROC, F1, and balanced accuracy.
- Analyze probe coefficients and feature stability.
- Select candidate SAE features using the discovery split.
- Validate candidate features on held-out interactions.
- Implement target-feature suppression and amplification experiments.
- Implement random-feature and reconstruction-only controls.
- Analyze causal effects on feedback acceptance and final correctness.

This role owns the transition from correlational feature discovery to causal testing.

---

### SHARED RESPONSIBILITIES:

All three students will contribute to:

- experimental design;
- controlled-feedback protocol;
- selection of activation sites;
- evaluation methodology;
- statistical analysis;
- interpretation of results;
- robustness checks;
- paper writing;
- final presentation;
- repository documentation.

The project should be treated as a single experimental pipeline rather than three independent subprojects. In particular, the behavioral labels created by Student 1, SAE representations produced by Student 2, and candidate features identified by Student 3 must remain aligned through stable episode identifiers and frozen experimental splits.
