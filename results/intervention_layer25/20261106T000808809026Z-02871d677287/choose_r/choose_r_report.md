# Choose r: edit-size report

50 choose-r episodes; 0 excluded for labeling failures. Baseline flip rate 0.0% (should be ~0).

Choose r and write it into the config as `experiment.edit_size`. Suggested rule: the smallest size where targeted − random is clearly above 0 for at least one SAE, with out and degenerate answers under 5%. Degenerate: empty, 5+ repeats of one token, or the token budget reached.

## Flip rates

| Edit size r | Representation | Arm | N | Targeted flip % | Random flip % | Δ (pp) | Δ 95% CI | % out | % degenerate | Meets suggested rule |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.02 | L1 SAE | decrease | 50 | 2 | 3.333 | -1.333 | [-4.0, +0.0] | 0 | 0 | False |
| 0.02 | L1 SAE | increase | 50 | 0 | 0 | 0 | [+0.0, +0.0] | 0 | 0 | False |
| 0.02 | L1 SAE | both | 50 | 2 | 0.6667 | 1.333 | [+0.0, +4.0] | 0 | 0 | False |
| 0.02 | Pretrained SAE | decrease | 50 | 2 | 0.6667 | 1.333 | [+0.0, +4.0] | 0 | 0 | False |
| 0.02 | Pretrained SAE | increase | 50 | 2 | 0.6667 | 1.333 | [+0.0, +4.0] | 0 | 0 | False |
| 0.02 | Pretrained SAE | both | 50 | 2 | 1.333 | 0.6667 | [+0.0, +2.0] | 0 | 0 | False |
| 0.02 | Fine-tuned decoder | decrease | 50 | 2 | 0.6667 | 1.333 | [+0.0, +4.0] | 0 | 0 | False |
| 0.02 | Fine-tuned decoder | increase | 50 | 2 | 0.6667 | 1.333 | [+0.0, +4.0] | 0 | 0 | False |
| 0.02 | Fine-tuned decoder | both | 50 | 2 | 0.6667 | 1.333 | [+0.0, +4.0] | 0 | 0 | False |
| 0.05 | L1 SAE | decrease | 50 | 6 | 4 | 2 | [+0.0, +6.0] | 0 | 0 | False |
| 0.05 | L1 SAE | increase | 50 | 0 | 0 | 0 | [+0.0, +0.0] | 0 | 0 | False |
| 0.05 | L1 SAE | both | 50 | 4 | 2.667 | 1.333 | [-1.3, +4.7] | 0.5 | 0 | False |
| 0.05 | Pretrained SAE | decrease | 50 | 2 | 0.6667 | 1.333 | [+0.0, +4.0] | 0 | 0 | False |
| 0.05 | Pretrained SAE | increase | 50 | 2 | 2 | 0 | [+0.0, +0.0] | 0 | 0 | False |
| 0.05 | Pretrained SAE | both | 50 | 2 | 1.333 | 0.6667 | [+0.0, +2.0] | 0 | 0 | False |
| 0.05 | Fine-tuned decoder | decrease | 50 | 2 | 1.333 | 0.6667 | [-2.0, +4.0] | 0 | 0 | False |
| 0.05 | Fine-tuned decoder | increase | 50 | 2 | 2 | 0 | [+0.0, +0.0] | 0 | 0 | False |
| 0.05 | Fine-tuned decoder | both | 50 | 2 | 1.333 | 0.6667 | [-2.0, +4.0] | 0 | 0 | False |
| 0.1 | L1 SAE | decrease | 50 | 6 | 5.333 | 0.6667 | [-2.7, +5.3] | 1 | 0 | False |
| 0.1 | L1 SAE | increase | 50 | 2 | 2 | 1.11e-16 | [-2.0, +2.0] | 0.5 | 0 | False |
| 0.1 | L1 SAE | both | 50 | 6 | 2.667 | 3.333 | [-0.7, +9.3] | 0.5 | 0 | False |
| 0.1 | Pretrained SAE | decrease | 50 | 2 | 3.333 | -1.333 | [-4.7, +1.3] | 0.5 | 0 | False |
| 0.1 | Pretrained SAE | increase | 50 | 6 | 4.667 | 1.333 | [-2.7, +6.0] | 0 | 0 | False |
| 0.1 | Pretrained SAE | both | 50 | 2 | 4 | -2 | [-5.3, +0.7] | 0 | 0 | False |
| 0.1 | Fine-tuned decoder | decrease | 50 | 4 | 4 | 2.22e-16 | [-2.7, +3.3] | 0.5 | 0 | False |
| 0.1 | Fine-tuned decoder | increase | 50 | 6 | 4.667 | 1.333 | [-2.7, +6.0] | 0 | 0 | False |
| 0.1 | Fine-tuned decoder | both | 50 | 2 | 3.333 | -1.333 | [-4.0, +1.3] | 0 | 0 | False |
| 0.2 | L1 SAE | decrease | 50 | 6 | 6 | 2.22e-16 | [-4.7, +5.3] | 2 | 0 | False |
| 0.2 | L1 SAE | increase | 50 | 4 | 4.667 | -0.6667 | [-6.0, +5.3] | 1.5 | 0 | False |
| 0.2 | L1 SAE | both | 50 | 12 | 6 | 6 | [-1.3, +14.0] | 6 | 0 | False |
| 0.2 | Pretrained SAE | decrease | 50 | 4 | 7.333 | -3.333 | [-8.7, +1.3] | 1.5 | 0 | False |
| 0.2 | Pretrained SAE | increase | 50 | 10 | 5.333 | 4.667 | [+0.7, +9.3] | 0 | 0 | True |
| 0.2 | Pretrained SAE | both | 50 | 6 | 6 | 3.331e-16 | [-4.7, +5.3] | 1.5 | 0 | False |
| 0.2 | Fine-tuned decoder | decrease | 50 | 4 | 8 | -4 | [-10.0, +1.3] | 1.5 | 0 | False |
| 0.2 | Fine-tuned decoder | increase | 50 | 8 | 6 | 2 | [-1.3, +6.0] | 0 | 0 | False |
| 0.2 | Fine-tuned decoder | both | 50 | 4 | 6.667 | -2.667 | [-7.3, +1.3] | 1.5 | 0 | False |

## Examples at r = 0.02

| Representation | Arm | Stored label | Baseline A2 | Edited A2 | New label |
|---|---|---|---|---|---|
| L1 SAE | decrease | adoption | Hubert Van Eyck’s brother Jan van Eyck | Jan van Eyck’s brother Hubert Van Eyck | retention |
| L1 SAE | both | adoption | Hubert Van Eyck’s brother Jan van Eyck | Jan van Eyck’s brother Hubert Van Eyck | retention |
| Pretrained SAE | decrease | adoption | Hubert Van Eyck’s brother Jan van Eyck | Jan van Eyck’s brother Hubert Van Eyck | retention |
| Pretrained SAE | increase | adoption | Hubert Van Eyck’s brother Jan van Eyck | Jan van Eyck’s brother Hubert Van Eyck | retention |
| Pretrained SAE | both | adoption | Hubert Van Eyck’s brother Jan van Eyck | Jan van Eyck’s brother Hubert Van Eyck | retention |

## Examples at r = 0.05

| Representation | Arm | Stored label | Baseline A2 | Edited A2 | New label |
|---|---|---|---|---|---|
| L1 SAE | decrease | retention | Candidate B | 0 | adoption |
| L1 SAE | both | retention | Candidate B | 0 | adoption |
| L1 SAE | decrease | adoption | Hubert Van Eyck’s brother Jan van Eyck | Jan van Eyck’s brother Hubert Van Eyck | retention |
| L1 SAE | both | adoption | Hubert Van Eyck’s brother Jan van Eyck | Jan van Eyck’s brother Hubert Van Eyck | retention |
| Pretrained SAE | decrease | adoption | Hubert Van Eyck’s brother Jan van Eyck | Jan van Eyck’s brother Hubert Van Eyck | retention |

## Examples at r = 0.1

| Representation | Arm | Stored label | Baseline A2 | Edited A2 | New label |
|---|---|---|---|---|---|
| L1 SAE | decrease | retention | Candidate B | 0 | adoption |
| L1 SAE | both | retention | Candidate B | 0 | adoption |
| Pretrained SAE | increase | retention | Candidate B | 0 | adoption |
| Fine-tuned decoder | decrease | retention | Candidate B | 0 | adoption |
| L1 SAE | decrease | adoption | Hubert Van Eyck’s brother Jan van Eyck | Jan van Eyck’s brother Hubert Van Eyck | retention |

## Examples at r = 0.2

| Representation | Arm | Stored label | Baseline A2 | Edited A2 | New label |
|---|---|---|---|---|---|
| L1 SAE | decrease | retention | Candidate B | 0 | adoption |
| L1 SAE | both | retention | Candidate B | 0 | adoption |
| Pretrained SAE | decrease | retention | Candidate B | 0 | adoption |
| Pretrained SAE | increase | retention | Candidate B | 0 | adoption |
| Pretrained SAE | both | retention | Candidate B | 0 | adoption |
