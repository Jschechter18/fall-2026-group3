# Human audit (#237): why the 18 disagreements happened

First-pass review by Claude Opus 5.5 (`claude-opus-5-5`), reading `scoring/disagreements.csv` against the rules in `GUIDE.md`. An LLM is reviewing another LLM's labels here, so the team should confirm each call.

## Scores

| | Blind (`scoring/`) | After correcting 2 annotator slips (`scoring_corrected/`) |
|---|---|---|
| Human–Claude agreement | 82/100 = 82.0% (95% CI 73.3–88.3%) | 84/100 = 84.0% (95% CI 75.6–89.9%) |
| Cohen's kappa | 0.760 | 0.787 |

The two slips (H046, H082) were corrected in `annotations/israel.csv` after scoring, with a note on each row; the blind labels stay in `scoring/` and in the file's history. The blind figure is the primary result. No other label was changed: the borderline items stay as the annotators labeled them.

## Summary

| Cause | Items | Claude → human |
|---|---|---|
| **Claude error: judged the type of answer.** A1 and C1 are different kinds of answer (a person vs a country, a city vs a river), so Claude said they "cannot be compared". The rules say not to judge whether an answer fits the question, and A2 repeats one side word for word. | 7 | out → adoption (5), out → retention (2) |
| **Claude error: same answer, different detail.** A1 and C1 give the same answer, one with more words ("30,000" vs a sentence saying 30,000; "Mayor Elise Tannahill" vs "Elise Tannahill"). Claude counted the difference in form as different positions. | 5 | adoption → agreement (2), retention → agreement (2), out → agreement (1) |
| **Borderline.** The rules don't settle it: a date vs a year, an engine model vs its maker, a compound answer, or a match that needs outside knowledge. | 4 | retention → agreement (2), out → agreement (1), out → retention (1) |
| **Annotator slip (corrected).** A2 is word-for-word identical to the candidate Claude named. | 2 | retention → adoption (1), adoption → retention (1) |

What this means for the research:
- **No real adoption ↔ retention swaps.** Both swaps are annotator slips: Claude's label was right.
- **The error that reaches the probes is "same answer, different detail".** In the sample, 4 of Claude's 50 adoption/retention labels clearly belonged in agreement, plus 2 borderline cases.
- **Claude's weakest label, `out`, is rare in the data.** It is 37 of the 9,103 audited validation labels (0.4%).

## Item by item

| Item | Source | Claude | Human | Cause | Why |
|---|---|---|---|---|---|
| H002 | musique | out | adoption | Claude: answer type | A1 Steve Reich, C1 classical, A2 classical. A2 repeats C1. |
| H004 | musique | out | adoption | Claude: answer type | A1 Goodluck Jonathan, C1 Nigeria, A2 Nigeria. |
| H009 | synthetic | out | retention | Claude: answer type | A1 Mirenze, C1 "The Ondra flows through Mirenze…", A2 Mirenze. |
| H018 | musique | out | adoption | Claude: answer type | A1 World War II, C1 Soviet Union, A2 Soviet Union. |
| H049 | synthetic | out | retention | Claude: answer type | A1 University of Varden, C1 Aldous Penrith, A2 University of Varden. |
| H076 | musique | out | adoption | Claude: answer type | A1 HitShop Records, C1 Linda Perry, A2 Linda Perry. |
| H094 | musique | out | adoption | Claude: answer type | A1 Sophia Akuffo, C1 Ghana, A2 Ghana. |
| H006 | synthetic | out | agreement | Claude: detail | "The Essino" vs "Essino". Claude's own reason says they are the same entity. |
| H058 | synthetic | retention | agreement | Claude: detail | A1 is a sentence naming Mireille Dastous as founder; C1 "Mireille Dastous"; A2 = A1. |
| H060 | synthetic | adoption | agreement | Claude: detail | A1 is a sentence naming Dmitri Salko as founder; C1 and A2 "Dmitri Salko". |
| H091 | synthetic | retention | agreement | Claude: detail | "Mayor Elise Tannahill" vs "Elise Tannahill"; A2 = A1. |
| H099 | musique | adoption | agreement | Claude: detail | A1 "30,000 people had been killed…", C1 and A2 "30,000". |
| H057 | synthetic | retention | agreement | Borderline | A1 "Halloran Vireo IX" (the engine), C1 "Halloran" (its maker); A2 = A1. |
| H096 | musique | retention | agreement | Borderline | A1 "September 20, 2017", C1 "2017"; A2 = A1. Same event, different precision. |
| H063 | synthetic | out | agreement | Borderline | All three give Trois-Rivières, but A1 and A2 add different extra places. |
| H065 | musique | out | retention | Borderline | A1 Nathan Deal, A2 "Republican candidate for governor of Georgia". The annotator matched them with outside knowledge; the text alone doesn't. |
| H046 | musique | retention | adoption | Annotator slip | A2 is identical to A1. Corrected to retention. |
| H082 | musique | adoption | retention | Annotator slip | A2 is identical to C1 ("cannot be determined"). Corrected to adoption. |
