"""Linear probes on Solver activations: layer selection (stage 3) and SAE feature selection (stage 5).

``behavior_data`` joins activations, labels and partitions; ``cv_eval`` holds the grouped folds,
bootstrap and metrics both stages share; ``matched_edits`` re-evaluates a feature-selection run at
equal edit sizes. ``data``, ``model`` and ``interpretability`` are earlier probe utilities used only
by archived scripts and ``mas_sae.causal.intervention``.
"""
