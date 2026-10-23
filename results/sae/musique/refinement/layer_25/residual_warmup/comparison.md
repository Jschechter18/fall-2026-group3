# Refinement comparison

Final epochs; no automatic checkpoint promotion.

| Seed | Branch | Validation MSE | Change vs matched control | Train active | Validation active | Net gain | Displaced |
|---|---|---:|---:|---:|---:|---:|---:|
| 42 | warmup_control | 0.0057680248 | +0.000% | 0 | 1 | 1 | 7 |
| 42 | residual_warmup | 0.0057663006 | -0.030% | 432 | 299 | 429 | 9 |

Activity columns track the same candidate IDs in controls and treatments.
Review repeated firing in comparison.json and rerun downstream feature analysis before selecting a checkpoint.
