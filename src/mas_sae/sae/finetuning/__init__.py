"""Light fine-tuning of a pretrained Gemma Scope (JumpReLU) SAE on our activations.

The entry point is scripts/sae/finetune_pretrained_sae.py; its run() calls these
modules in order:

    data        Split, load_splits, probe_cohort_rows, iterate_batches, features_for
    cache       on-disk cache of a frozen encoder's features (decoder-only mode), encoder_digest
    evaluation  reconstruction metrics (evaluate, evaluate_splits) and the bias-only reference
    trainers    DecoderOnlyTrainer, FullTrainer (JumpReLU straight-through estimators + L0 penalty)
    loop        train(): the epoch loop that keeps the best eligible checkpoint; save_model
    selection   within_l0, eligibility, stability_limits: which epochs may be kept
    reporting   dictionary health and stability (wrapping mas_sae.sae.diagnostics), feature_health.npz
"""
