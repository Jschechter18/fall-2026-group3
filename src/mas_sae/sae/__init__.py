"""Stage 4 of the pipeline: sparse autoencoders (SAEs) on the Solver's residual-stream activations.

    sparse_autoencoder  SparseAutoencoder, the L1 / TopK SAE trained here
    training, tuning    train_sae / finish_run and grid-search studies (scripts/sae/train_sae.py)
    model_runner        one epoch of training or evaluation: losses, AuxK, activity counters
    callbacks           best-checkpoint saving and loading, early stopping
    training_state      epoch-boundary state (incl. RNG) for continuing a trial
    pretrained          Gemma Scope JumpReLU SAEs behind the same encoder/decoder interface
    finetuning          light fine-tuning of a Gemma Scope SAE (finetune_pretrained_sae.py)
    diagnostics         dictionary health and stability beyond reconstruction error
    refinement          feature-revival continuations of a TopK trial (uses resampling)

Later stages load SAEs with callbacks.checkpointing.load_sae_checkpoint (trained here) or
pretrained.load_gemma_scope_sae (Gemma Scope layout, including fine-tuned checkpoints).
"""
