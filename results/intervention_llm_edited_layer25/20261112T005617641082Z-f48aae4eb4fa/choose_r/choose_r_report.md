# Choose r: edit-size report

50 choose-r episodes; 0 excluded for labeling failures. Baseline flip rate 0.0% (should be ~0).

Choose r and write it into the config as `experiment.edit_size`. Suggested rule: the smallest size where targeted − random is clearly above 0 for at least one SAE, with out and degenerate answers under 5%. Degenerate: empty, 5+ repeats of one token, or the token budget reached.

## Flip rates

| Edit size r | Representation | Arm | N | Targeted flip % | Random flip % | Δ (pp) | Δ 95% CI | % out | % degenerate | Meets suggested rule |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | LLM-edited SAE | decrease | 50 | 8 | 7.333 | 0.6667 | [-6.0, +7.3] | 5.5 | 0 | False |
| 0.2 | LLM-edited SAE | increase | 50 | 10 | 5.333 | 4.667 | [+0.0, +10.7] | 4 | 0 | False |
| 0.2 | LLM-edited SAE | both | 50 | 10 | 6 | 4 | [-2.7, +11.3] | 6 | 0 | False |
| 0.2 | Mechanical edit | decrease | 50 | 8 | 7.333 | 0.6667 | [-4.7, +6.0] | 3.5 | 0 | False |
| 0.2 | Mechanical edit | increase | 50 | 10 | 5.333 | 4.667 | [+0.0, +10.7] | 1.5 | 0 | False |
| 0.2 | Mechanical edit | both | 50 | 10 | 6 | 4 | [-2.7, +11.3] | 4.5 | 0 | False |
| 0.2 | Random edit | decrease | 50 | 2 | 7.333 | -5.333 | [-11.3, -0.7] | 1 | 0 | False |
| 0.2 | Random edit | increase | 50 | 8 | 5.333 | 2.667 | [-0.7, +7.3] | 0 | 0 | False |
| 0.2 | Random edit | both | 50 | 4 | 6 | -2 | [-6.7, +2.0] | 1.5 | 0 | False |

## Examples at r = 0.2

| Representation | Arm | Stored label | Baseline A2 | Edited A2 | New label |
|---|---|---|---|---|---|
| LLM-edited SAE | increase | retention | Candidate B | The evidence is insufficient. | out |
| LLM-edited SAE | both | retention | Candidate B | The evidence is insufficient. | out |
| Mechanical edit | decrease | retention | Candidate B | The evidence is insufficient. | out |
| Mechanical edit | increase | retention | Candidate B | The evidence is insufficient. | out |
| Mechanical edit | both | retention | Candidate B | The evidence is insufficient. | out |
