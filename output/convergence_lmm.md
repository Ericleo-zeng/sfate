# Convergence & sensitivity re-analysisanalyze_convergence_lmm.py

## r  LMMSpearman ~ log2(r) + (1|state))

- n = 306 states × 5 r levels
- β per doubling = 0.1149 (95% CI 0.0980–0.1318)
- per-state mean |ΔP| first < 0.05 at r = 10
- per-state mean |ΔP| first < 0.02 at r = 100

```
        Mixed Linear Model Regression Results
======================================================
Model:            MixedLM Dependent Variable: spearman
No. Observations: 30      Method:             REML
No. Groups:       6       Scale:              0.0028
Min. group size:  5       Log-Likelihood:     31.4099
Max. group size:  5       Converged:          Yes
Mean group size:  5.0
------------------------------------------------------
             Coef. Std.Err.   z    P>|z| [0.025 0.975]
------------------------------------------------------
Intercept    0.031    0.061  0.511 0.609 -0.088  0.150
log2r        0.115    0.009 13.353 0.000  0.098  0.132
Group Var    0.011    0.145
======================================================

```

## kNN k  k=30

- MAE range vs k=30: 9.42e-03–1.57e-02

| k | graph_s | solve_s | nnz | nnz/(n·k) | L∞ vs k=30 | MAE vs k=30 | Spearman min | Spearman median |
|---|---|---|---|---|---|---|---|---|
| 15 | 26.42 | 0.66 | 126,328 | 1.6844 | 5.08e-01 | 1.57e-02 | 0.8841 | 0.9182 |
| 50 | 1.51 | 0.79 | 416,328 | 1.6653 | 1.89e-01 | 9.42e-03 | 0.9193 | 0.9490 |
| 100 | 2.59 | 1.33 | 821,986 | 1.644 | 2.54e-01 | 1.55e-02 | 0.8350 | 0.8949 |

- Note: k=100 shows non-monotonic Spearman decline relative to k=50
