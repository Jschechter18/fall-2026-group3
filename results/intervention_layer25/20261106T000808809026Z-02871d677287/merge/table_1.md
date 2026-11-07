| Representation | Stratum (baseline → target) | Arm | N | Targeted flip % | Random flip % (mean of draws) | Δ (pp) [95% CI] | p | p (Holm) | Δ log p(baseline first token) |
|---|---|---|---|---|---|---|---|---|---|
| Raw activations | adoption → retention | decrease adoption features | 496 | 10.28 | 4.099 | +6.2 [+3.9, +8.6] | 9.999e-05 | 0.0005999 | -0.3122 |
| Raw activations | adoption → retention | increase retention features | 496 | 8.266 | 6.855 | +1.4 [-1.0, +3.9] | 0.2895 | 0.2895 | 0.3882 |
| Raw activations | adoption → retention | both | 496 | 11.09 | 5.847 | +5.2 [+2.6, +8.0] | 0.0003 | 0.0015 | -0.1838 |
| Raw activations | retention → adoption | decrease retention features | 487 | 5.955 | 4.107 | +1.8 [-0.3, +4.1] | 0.1058 | 0.2116 | -6.624 |
| Raw activations | retention → adoption | increase adoption features | 487 | 5.339 | 3.354 | +2.0 [+0.1, +4.0] | 0.05339 | 0.1602 | -0.1579 |
| Raw activations | retention → adoption | both | 487 | 5.955 | 3.08 | +2.9 [+0.9, +5.0] | 0.006199 | 0.0248 | -5.765 |
| L1 SAE | adoption → retention | decrease adoption features | 496 | 9.879 | 5.847 | +4.0 [+1.7, +6.5] | 0.0009999 | 0.005999 | 0.1286 |
| L1 SAE | adoption → retention | increase retention features | 496 | 4.435 | 5.847 | -1.4 [-3.2, +0.4] | 0.1524 | 0.47 | 0.365 |
| L1 SAE | adoption → retention | both | 496 | 8.468 | 6.922 | +1.5 [-0.9, +4.0] | 0.2412 | 0.4824 | 0.8002 |
| L1 SAE | retention → adoption | decrease retention features | 487 | 3.491 | 3.97 | -0.5 [-2.3, +1.3] | 0.645 | 0.645 | 0.5876 |
| L1 SAE | retention → adoption | increase adoption features | 487 | 6.776 | 5.27 | +1.5 [-0.3, +3.4] | 0.1175 | 0.47 | -0.003434 |
| L1 SAE | retention → adoption | both | 487 | 4.312 | 6.16 | -1.8 [-3.7, +0.1] | 0.06369 | 0.3185 | 0.9227 |
| Pretrained SAE | adoption → retention | decrease adoption features | 496 | 5.847 | 4.099 | +1.7 [-0.1, +3.7] | 0.09659 | 0.2631 | -7.229 |
| Pretrained SAE | adoption → retention | increase retention features | 496 | 5.645 | 3.696 | +1.9 [+0.4, +3.6] | 0.0268 | 0.134 | 0.2439 |
| Pretrained SAE | adoption → retention | both | 496 | 8.266 | 4.839 | +3.4 [+1.3, +5.6] | 0.0039 | 0.0234 | -4.684 |
| Pretrained SAE | retention → adoption | decrease retention features | 487 | 4.312 | 2.738 | +1.6 [+0.1, +3.1] | 0.0426 | 0.1704 | 0.7493 |
| Pretrained SAE | retention → adoption | increase adoption features | 487 | 2.053 | 1.711 | +0.3 [-0.5, +1.2] | 0.5367 | 0.5367 | 0.02595 |
| Pretrained SAE | retention → adoption | both | 487 | 4.107 | 2.601 | +1.5 [-0.1, +3.2] | 0.08769 | 0.2631 | 0.4186 |
| Fine-tuned decoder | adoption → retention | decrease adoption features | 496 | 5.242 | 4.032 | +1.2 [-0.5, +3.0] | 0.2287 | 0.4574 | -4.832 |
| Fine-tuned decoder | adoption → retention | increase retention features | 496 | 5.04 | 4.099 | +0.9 [-0.8, +2.8] | 0.3335 | 0.4574 | 0.1238 |
| Fine-tuned decoder | adoption → retention | both | 496 | 7.056 | 4.839 | +2.2 [+0.2, +4.3] | 0.043 | 0.172 | -1.049 |
| Fine-tuned decoder | retention → adoption | decrease retention features | 487 | 4.517 | 2.533 | +2.0 [+0.4, +3.7] | 0.0203 | 0.1218 | -0.2686 |
| Fine-tuned decoder | retention → adoption | increase adoption features | 487 | 3.285 | 1.985 | +1.3 [-0.1, +2.7] | 0.09339 | 0.2802 | 0.02046 |
| Fine-tuned decoder | retention → adoption | both | 487 | 4.107 | 2.259 | +1.8 [+0.3, +3.6] | 0.0284 | 0.142 | 0.2619 |
| Fully fine-tuned † | adoption → retention | decrease adoption features | 496 | 5.242 | 4.503 | +0.7 [-1.1, +2.7] | 0.4854 | 1 | -2.28 |
| Fully fine-tuned † | adoption → retention | increase retention features | 496 | 7.46 | 4.032 | +3.4 [+1.5, +5.4] | 0.0008999 | 0.005399 | -0.06022 |
| Fully fine-tuned † | adoption → retention | both | 496 | 8.266 | 5.376 | +2.9 [+0.9, +5.0] | 0.006699 | 0.0335 | -6.98 |
| Fully fine-tuned † | retention → adoption | decrease retention features | 487 | 2.669 | 2.738 | -0.1 [-1.5, +1.4] | 1 | 1 | 0.981 |
| Fully fine-tuned † | retention → adoption | increase adoption features | 487 | 2.464 | 2.053 | +0.4 [-0.8, +1.8] | 0.6163 | 1 | 0.06771 |
| Fully fine-tuned † | retention → adoption | both | 487 | 4.107 | 3.012 | +1.1 [-0.4, +2.7] | 0.206 | 0.8239 | 0.2093 |
