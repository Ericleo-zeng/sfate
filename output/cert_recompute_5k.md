# cert_recompute_5k.py 

-  2026-09-20seed=20260909

## a2 n=1600 f64 GMRES tol=1e-12 + ILU

| solve_mode |  max | κ∞(I−Q) | 10·κ∞·backward | L∞ vs spsolve |  | PASS |
|---|---|---|---|---|---|---|
| aggregated | 1.35e-13 | 5.01e+02 | 6.78e-10 | 8.78e-13 | 1e-12 | ✅ |
| columns | 2.08e-13 | 5.01e+02 | 1.04e-09 | 9.28e-13 | 1e-10 | ✅ |

## a3 fixture_5kf32 GMRES tol=1e-6columns

|  |  |
|---|---|
| L∞ vs f64  | 1.74e-06 |
|  max  | 2.59e-07 |
|  η=max ‖Ax−b‖/(‖A‖‖x‖+‖b‖) | 3.27e-07 |
| κ∞(I−Q)M- | 3.21e+01 |
| 10·κ∞·η | 1.05e-04 |
|  L∞ ≤ 1e-5 |  5.7× |
| PASS | ✅ |

## 

- ≤1e-12  GMRES f64 
- tol=1e-6, f32 ~1e-7 
  L∞ ≤ 1e-5 ——1e-12 1e-5 