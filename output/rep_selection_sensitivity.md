# Representative-selection sensitivity (5k fixture)

Graph built once: k=30, seed=20260909, nnz=251140.
Reference: centroid-nearest r=30 per class. Perturbation: 10 random r=30 selections per class.

## Per-class summary

| state | n cells | median Spearman | 5–95% Spearman | mean MAE | 5–95% MAE | max L∞ | mean overlap vs ref | random–random Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1q_inflammatory | 905 | 0.592 | [0.473, 0.696] | 0.047 | [0.043, 0.051] | 0.921 | 0.02 | 0.02 |
| DAM_like | 453 | 0.327 | [0.169, 0.556] | 0.055 | [0.045, 0.070] | 0.936 | 0.05 | 0.04 |
| Homeostatic | 1864 | 0.116 | [0.001, 0.365] | 0.091 | [0.074, 0.108] | 0.865 | 0.01 | 0.01 |
| IFN_responsive | 1354 | 0.238 | [0.099, 0.426] | 0.060 | [0.045, 0.077] | 0.860 | 0.02 | 0.01 |
| PU1low_lymphoid | 371 | 0.623 | [0.536, 0.718] | 0.049 | [0.042, 0.055] | 0.886 | 0.04 | 0.04 |
| Proliferating | 53 | 0.572 | [0.538, 0.660] | 0.065 | [0.060, 0.068] | 0.952 | 0.42 | 0.40 |

## Overall

- Median per-state Spearman: 0.450
- Median per-state MAE: 0.058
- Max L∞ across all states and draws: 0.952
