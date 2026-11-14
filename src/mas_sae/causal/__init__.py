"""The earlier causal pipeline (issue #90), replaced by the stage-6 intervention (mas_sae.intervention).

``replay`` regenerates the Solver's A2 with an SAE spliced into the residual stream; ``distributed``
and ``distributed_cli`` run it as prepare / map / reduce stages; ``semantic`` labels the replays with
Claude; ``prefix`` reuses the shared prefill; ``prepare`` builds the label and reference inputs;
``smoke`` holds tiny CPU stand-ins; ``intervention`` is an early synthetic-model experiment.
``offline`` is still used by feature selection (stage 5).
"""
