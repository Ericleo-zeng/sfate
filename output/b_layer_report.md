# B sfate vs CellRank b_layer_validation.py

##
- cellrank **2.1.0** 2.3.2****GPCCA  fate  2.1.0  SLEPc/ pygpcca 1.0.4/ scanpy 1.11.5 / scipy 1.15.3 / numpy 2.4.2
- fixture_5kn=5000k=30 20260909 BLAS=1
- swap`NAME           TYPE       SIZE USED PRIO ; /swapfile      file      15.6G 348K   -1 ; /dev/nvme0n1p7 partition 30.5G 256K   -1`
-  61.3stracemalloc  350MB output/b_layer_time_v.txt

##
|  |  | (s) |
|---|---|---|
| CellRank | neighbors→ConnectivityKernel→GPCCA( n_states=6eigengap  1 )→→fate(gmres+ilu) | neighbors 16.4 / kernel 0.0 / gpcca_fit 6.3 / terminal 0.0 / fate 0.2 |
| sfate | kNN →absorption_probabilities | graph 35.0 / solve 0.7 |

## B ②
- CellRank 6['DAM_like', 'IFN_responsive_1', 'IFN_responsive_2', 'C1q_inflammatory', 'Homeostatic_1', 'Homeostatic_2']
- CellRank stability≥0.96['IFN_responsive_1']
- sfate  6 ['C1q_inflammatory', 'DAM_like', 'Homeostatic', 'IFN_responsive', 'PU1low_lymphoid', 'Proliferating']
- sfate GMRES /[28, 28, 30, 28, 27, 26] max1.41e-07f64 0
- GPCCA  kernel  fit=True=True

##
### stability≥0.96
- 1['IFN_responsive_1']
-  <2Spearman

###
- 6['DAM_like', 'IFN_responsive_1', 'IFN_responsive_2', 'C1q_inflammatory', 'Homeostatic_1', 'Homeostatic_2']

|  | CR  | Spearman | Pearson |  |
|---|---|---|---|---|
| C1q_inflammatory | C1q_inflammatory | 0.1492 | 0.0573 | 135 |
| DAM_like | DAM_like | -0.3918 | -0.0487 | 62 |
| Homeostatic | Homeostatic_2 | 0.3845 | 0.1302 | 999 |
| IFN_responsive | Homeostatic_1 | -0.0098 | -0.0191 | 234 |
| PU1low_lymphoid | IFN_responsive_2 | -0.3034 | -0.0180 | 24 |
| Proliferating | IFN_responsive_1 | 0.2534 | 0.2499 | 2 |

-  Spearman  = **-0.3918**argmax ARI = 0.2315n=180= 0.1833
-  JaccardC1q_inflammatory 0.009DAM_like 0.000Homeostatic 0.648IFN_responsive 0.051PU1low_lymphoid 0.005Proliferating 0.411
-

### 2×2 ×  sfate
- ** 1E1 **sfate (P_cr, T_cr top-30 ) vs CellRank fateL∞ = 1.07e-06—— A  CR 2.1.0  top-30 CR  A  0  max 1.55e-07
- ** 2=**P_cr vs P_sf  Spearman  = **0.8299**C1q_inflammatory 0.940DAM_like 0.899Homeostatic 0.830IFN_responsive 0.879PU1low_lymphoid 0.926Proliferating 0.903
- ** 3=P_cr**T_cr vs T_sf  Spearman  = **0.1359**DAM_like→DAM_like 0.164IFN_responsive_1→Proliferating 0.157IFN_responsive_2→PU1low_lymphoid 0.436C1q_inflammatory→C1q_inflammatory 0.636Homeostatic_1→IFN_responsive 0.136Homeostatic_2→Homeostatic 0.223
-  2  ⇒  3  ⇒

## E4 >0.95  / 0.90–0.95  / <0.90

**CR  = 1['IFN_responsive_1']Spearman —— B GPCCA **

## E2
1. **A ** P  sfate vs spsolve(f64) L∞=3.2e-7 PASSoutput/benchmark_m2.md—— bugB
2. ****sfate  scVI latent kNNgaussian max f32CellRank  scanpy connectivities + density_normalize
3. **** 6  vs GPCCA  n_states=6eigengap  1 stability≥0.96sfate 6  vs CR  1 ['IFN_responsive_1']
4. **** gmres tol=1e-6CR  ilu use_petsc=FalseA

###
1. ****CR  + CR sfate  CellRank fate L∞ = 1.07e-06 1—— A  a3 E1  CR
2. ****CR  vs sfate  fate Spearman min = 0.830 2—— X_scVI  kNN/
3. ****GPCCA  vs  fate Spearman min = 0.136 3——min Spearman = -0.3918
4. **GPCCA  5k ** stability≥0.96  1 IFN_responsive_1 75k postmortemDAM/PU1low  transition/unsupported—— postmortem §6.3  C  75k
5. ****E4  <2
