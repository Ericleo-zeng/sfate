# C 75k c_layer_validation.py

##
- scipy 1.15.3 / numpy 2.4.2 /  20260909 /  BLAS=1
- CellRank  step12 postmortem  75k  ~405GB  cellrank 2.1.0 vs  2.3.2
- <mnt> h5py  anndata backed output/
- S3–S53.6sS1/S2  RSS ≈1.89GB [ time -v] 295MBtime -v  output/c_layer_time_v.txt
- swap`NAME           TYPE       SIZE USED PRIO ; /swapfile      file      15.6G 348K   -1 ; /dev/nvme0n1p7 partition 30.5G 256K   -1`

## sfate 75k S1/S2
|  | (s) |
|---|---|
| load latent+obsh5py  | 0.0 |
| build_transition_graph k=30 | 65.0 |
| absorption_probabilities 6  | 3915.2 |

- 1  0
- **75k GMRES design P0 ** [49951, 49921, 49942, 49912, 49911, 137] max 1.89e-07f64  5

## S3sfate vs step12 fate_* barcode
- sfate 74984 ∩ CR 74984 →  74984
|  | Spearman | Pearson |
|---|---|---|
| IFN_responsive | 0.3199 | 0.0085 |
| C1q_inflammatory | 0.3282 | 0.0624 |
| Homeostatic | -0.5676 | -0.0854 |

-  -0.5676 /  0.3199CR  fate ['DAM_like', 'PU1low_lymphoid', 'Proliferating']
- CR commitment_score 0.916≥0.8  0.998
- C ——B GPCCA  DAM/PU1low/Proliferating

## S4
- 12  delta **9/15**
|  |  |  | sfate  | step12  |  |
|---|---|---|---|---|---|
| IFN_responsive | 5xFAD−control | 3,6,8 | [1, -1, 1] | [1, -1, -1] | 2/3 |
| IFN_responsive | cKO−5xFAD | 3,6,8 | [1, 1, None] | [1, 1, None] | 2/2 |
| C1q_inflammatory | 5xFAD−control | 3,6,8 | [1, 1, 1] | [1, 1, -1] | 2/3 |
| C1q_inflammatory | cKO−5xFAD | 3,6,8 | [1, -1, None] | [1, -1, None] | 2/2 |
| Homeostatic | 5xFAD−control | 3,6,8 | [1, -1, -1] | [-1, -1, 1] | 1/3 |
| Homeostatic | cKO−5xFAD | 3,6,8 | [1, 1, None] | [-1, -1, None] | 0/2 |

## GPCCA  75k S5
- step12 ['IFN_responsive_2', 'IFN_responsive_1', 'IFN_responsive_3', 'C1q_inflammatory', 'Homeostatic']unsupported['DAM_like', 'PU1low_lymphoid']fate ['fate_IFN_responsive', 'fate_C1q_inflammatory', 'fate_Homeostatic']
- sfate6 ['C1q_inflammatory', 'DAM_like', 'Homeostatic', 'IFN_responsive', 'PU1low_lymphoid', 'Proliferating']——

###  × macrostate 75kmacrostates_fwd  30  74804 ——CR n_cells=30
|  \  | C1q_inflammatory | Homeostatic | IFN_responsive_1 | IFN_responsive_2 | IFN_responsive_3 | Proliferating |
|---|---|---|---|---|---|---|
| C1q_inflammatory | 13 | 0 | 0 | 0 | 0 | 0 |
| DAM_like | 8 | 8 | 2 | 4 | 6 | 0 |
| Homeostatic | 3 | 15 | 5 | 3 | 5 | 0 |
| IFN_responsive | 5 | 5 | 21 | 21 | 18 | 0 |
| PU1low_lymphoid | 0 | 2 | 2 | 1 | 0 | 0 |
| Proliferating | 1 | 0 | 0 | 1 | 1 | 30 |

###
- B  5kGPCCA  1  Spearman min = 0.136
- C  75kstep12  5 IFN  3 DAM_like/PU1low_lymphoid unsupportedProliferating  fate  75k  = **** 5k DAM/PU1low IFN
-  postmortem §6.3 GPCCA  Schur ——6  2–3  transition/unsupportedsfate  6
