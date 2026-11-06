# Choose r: edit-size report

45 choose-r episodes; 5 excluded for labeling failures. Baseline flip rate 0.0% (should be ~0).

Choose r and write it into the config as `experiment.edit_size`. Suggested rule: the smallest size where targeted − random is clearly above 0 for at least one SAE, with out and degenerate answers under 5%. Degenerate: empty, 5+ repeats of one token, or the token budget reached.

## Flip rates

| Edit size r | Representation | Arm | N | Targeted flip % | Random flip % | Δ (pp) | Δ 95% CI | % out | % degenerate | Meets suggested rule |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.3 | L1 SAE | decrease | 45 | 4.444 | 5.926 | -1.481 | [-9.6, +5.9] | 16.11 | 0 | False |
| 0.3 | L1 SAE | increase | 45 | 8.889 | 6.667 | 2.222 | [-5.2, +11.1] | 4.444 | 0 | False |
| 0.3 | L1 SAE | both | 45 | 11.11 | 8.889 | 2.222 | [-5.9, +11.1] | 21.11 | 1.111 | False |
| 0.3 | Pretrained SAE | decrease | 45 | 8.889 | 7.407 | 1.481 | [-2.2, +5.9] | 2.778 | 0 | False |
| 0.3 | Pretrained SAE | increase | 45 | 8.889 | 3.704 | 5.185 | [+0.0, +11.9] | 2.778 | 0 | False |
| 0.3 | Pretrained SAE | both | 45 | 6.667 | 5.185 | 1.481 | [-3.7, +8.1] | 3.889 | 0 | False |
| 0.3 | Fine-tuned decoder | decrease | 45 | 6.667 | 9.63 | -2.963 | [-9.6, +4.4] | 3.889 | 0 | False |
| 0.3 | Fine-tuned decoder | increase | 45 | 8.889 | 4.444 | 4.444 | [-0.7, +11.1] | 2.222 | 0 | False |
| 0.3 | Fine-tuned decoder | both | 45 | 4.444 | 8.889 | -4.444 | [-10.4, +1.5] | 3.333 | 0 | False |
| 0.5 | L1 SAE | decrease | 45 | 8.889 | 5.185 | 3.704 | [-4.4, +12.6] | 30.56 | 0 | False |
| 0.5 | L1 SAE | increase | 45 | 24.44 | 17.04 | 7.407 | [-4.4, +20.0] | 23.89 | 1.111 | False |
| 0.5 | L1 SAE | both | 45 | 15.56 | 11.85 | 3.704 | [-5.9, +13.3] | 35.56 | 0.5556 | False |
| 0.5 | Pretrained SAE | decrease | 45 | 8.889 | 10.37 | -1.481 | [-7.4, +3.7] | 12.78 | 0.5556 | False |
| 0.5 | Pretrained SAE | increase | 45 | 8.889 | 7.407 | 1.481 | [-5.2, +8.9] | 10 | 0 | False |
| 0.5 | Pretrained SAE | both | 45 | 6.667 | 7.407 | -0.7407 | [-8.1, +6.7] | 12.22 | 0 | False |
| 0.5 | Fine-tuned decoder | decrease | 45 | 8.889 | 14.07 | -5.185 | [-13.3, +2.2] | 12.22 | 2.222 | False |
| 0.5 | Fine-tuned decoder | increase | 45 | 11.11 | 6.667 | 4.444 | [-2.2, +11.9] | 9.444 | 0.5556 | False |
| 0.5 | Fine-tuned decoder | both | 45 | 2.222 | 6.667 | -4.444 | [-10.4, +0.7] | 15.56 | 1.111 | False |

## Examples at r = 0.3

| Representation | Arm | Stored label | Baseline A2 | Edited A2 | New label |
|---|---|---|---|---|---|
| Fine-tuned decoder | decrease | retention | Tony Curran | Justin Crow | out |
| L1 SAE | decrease | retention | Candidate B | 13 | retention |
| L1 SAE | both | retention | Candidate B | The evidence is insufficient. | out |
| Pretrained SAE | decrease | retention | Candidate B | 0 | adoption |
| Pretrained SAE | increase | retention | Candidate B | 0 | adoption |

## Examples at r = 0.5

| Representation | Arm | Stored label | Baseline A2 | Edited A2 | New label |
|---|---|---|---|---|---|
| L1 SAE | decrease | retention | Tony Curran | 1.  Tony Curran | retention |
| L1 SAE | increase | retention | Tony Curran | None | adoption |
| L1 SAE | both | retention | Tony Curran | ...Tony Curran | retention |
| Fine-tuned decoder | decrease | retention | Tony Curran | This evidence is insufficient. | out |
| L1 SAE | decrease | retention | Candidate B | 13 | retention |
