# Control 1: each SAE spliced in with no ablation, layer 25 (#220)

Run B of #220: config `configs/causal_reconstruction_control_layer25.yaml` (main 6a0883c), on the first 250 episodes of the seed-42 order (a subset of the TopK run's).

- **What it measures:** the Solver run on `decode(encode(x))` at the final prompt token of the revise prompt, against the plain replay, for TopK, the original Gemma Scope SAE and its decoder-fine-tuned copy.
- **Reported separately:** these results are reported separately from the error-preserving causal effects.
- **Descriptive only:** the 1-draw ablation rows of the reconstruction splice are descriptive.

**Run:**
- 2,500 generations.
- The replay check passed: 250/250 replays exactly match the stored answers.
- Claude labels, status `complete_with_exclusions`. One episode was excluded for an empty answer under ablation; it is the same episode the TopK run excluded.
- 249 episodes analysed.

**Main table:** `reconstruction_damage.csv`. Every SAE's no-ablation Δ adoption CI includes 0.

| SAE | Reconstruction error |
|---|---|
| TopK | 4.2% |
| original pretrained | 13.0% |
| fine-tuned | 7.8% |

**Full run** (per-row data and the labeling audit): `s3://dats-capstone/shared/group3/results/causal_reconstruction_control_layer25/20261103T202307480433Z/` (25 files).
