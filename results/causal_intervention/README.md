# Synthetic smoke test, not a result

`20260922T145224717439Z/` is not a causal result. It comes from the 2026-09-22 smoke test of the early
mock intervention pipeline (`src/mas_sae/causal/intervention.py`), which reads a probe on synthetic latents
instead of regenerating the Solver's answer. Its flip rates (about 0.5 for every feature) describe that
synthetic setup. Do not cite them.

The real causal experiment is the causal replay (`scripts/causal_replay.py`, `src/mas_sae/causal/replay.py`),
which regenerates the Solver's A2 with the edit applied; its runs are under `results/causal_replay/`.
