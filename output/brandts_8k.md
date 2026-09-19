# brandts  + 2.1.0 exp_brandts_points.py

##
- cellrank **2.1.0**T7  2.3.2—— E4 / pygpcca 1.0.4 / scipy 1.15.3 / numpy 2.4.2 /  BLAS=1 / seed 42
-  latenttests/conftest.pyd=10n_terminal=3scanpy neighbors(k=30) → ConnectivityKernel → `compute_schur(n_components=10, method='brandts')`→ compute_macrostates(n_states=3)
- tracemalloc  T7 + RSS 250ms output/brandts_8k_time_v.txt

##

| n | Schur (s) | macro (s) | tracemalloc (MB) | RSS (MB) | (B/n²) |
|---|---|---|---|---|---|
| 2000 | 4.07 | 0.26 | 288.3 | 1010.5 | 72.1 |
| 5000 | 43.9 | 1.67 | 1800.6 | 2322.6 | 72.0 |
| 8000 | 167.76 | 5.71 | 4609.0 | 4978.4 | 72.0 |

## E4 2.1.0  vs T7@2.3.2

| n | 2.3.2T7 | 2.1.0 |
|---|---|---|
| 2000 | 288.2 MB | 288.3 MB |
| 5000 | 1800.3 MB | 1800.6 MB |

## E3 n=8000

- n=8000tracemalloc  4609.0 MB 72.0 B/n²72·n²  4608 MB
-  ∈ [64, 80] B/n² n²
