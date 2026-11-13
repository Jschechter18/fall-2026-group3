# Human audit of the Claude labels (#237): results

100 validation episodes, balanced across Claude's four labels and the two sources, each labeled blind by one person with the rules Claude was given. Claude labels: `us.anthropic.claude-sonnet-4-5-20250929-v1:0`, prompt `claude_binary_v3_candidates`.

| | Value |
|---|---|
| Human–Claude agreement | 82/100 = 82.0% (95% CI 73.3%–88.3%) |
| Cohen's kappa | 0.760 |

## Confusion matrix (rows: human label, columns: Claude label)

| Human \ Claude | adoption | retention | agreement | out | Total |
|---|---|---|---|---|---|
| adoption | 22 | 1 | 0 | 5 | 28 |
| retention | 1 | 20 | 0 | 3 | 24 |
| agreement | 2 | 4 | 25 | 2 | 33 |
| out | 0 | 0 | 0 | 15 | 15 |

## Precision, recall and F1 by class (human label as reference)

| Label | Precision | Recall | F1 | Claude used it | Human used it |
|---|---|---|---|---|---|
| adoption | 88.0% | 78.6% | 0.830 | 25 | 28 |
| retention | 80.0% | 83.3% | 0.816 | 25 | 24 |
| agreement | 100.0% | 75.8% | 0.862 | 25 | 33 |
| out | 60.0% | 100.0% | 0.750 | 25 | 15 |

## Disagreements (18)

| Item | Source | Claude | Human | Annotator's note |
|---|---|---|---|---|
| H002 | musique | out | adoption |  |
| H004 | musique | out | adoption |  |
| H006 | synthetic | out | agreement |  |
| H009 | synthetic | out | retention |  |
| H018 | musique | out | adoption |  |
| H046 | musique | retention | adoption |  |
| H049 | synthetic | out | retention |  |
| H057 | synthetic | retention | agreement |  |
| H058 | synthetic | retention | agreement |  |
| H060 | synthetic | adoption | agreement |  |
| H063 | synthetic | out | agreement |  |
| H065 | musique | out | retention | Republican governor of Georgia is Nathan Deal. The A2 answer still refers to the actual person and it doesnt matter that they didnt answer the question right because it simply changed how it refered to the same person |
| H076 | musique | out | adoption |  |
| H082 | musique | adoption | retention |  |
| H091 | synthetic | retention | agreement |  |
| H094 | musique | out | adoption |  |
| H096 | musique | retention | agreement |  |
| H099 | musique | adoption | agreement |  |

Full text and Claude's reasons: `disagreements.csv`.

## Limits

- The sample is balanced by Claude's label and source, so these figures describe the audit sample, not Claude's accuracy across the whole dataset.
- Each example was labeled by one person, so this measures human–Claude agreement, not agreement between annotators.
- Left out before sampling: 2 × `None:incomplete_response:refusal`.
