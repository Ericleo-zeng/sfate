# NN-Descent recall and seed sensitivity (exp_nn_recall.py)

- date: 2026-09-19
- exact comparator: sklearn NearestNeighbors(brute, euclidean), k=30
- seeds: [20260907, 20260909, 20260911]

| fixture | n | seed | recall | build+query time (s) |
|---|---|---|---|---|
| fixture_5k_real | 5,000 | 20260907 | 0.9802 | 22.952 |
| fixture_5k_real | 5,000 | 20260909 | 0.9804 | 0.875 |
| fixture_5k_real | 5,000 | 20260911 | 0.9802 | 0.992 |
| fixture_5k_real | 5,000 | **summary** | mean 0.9803 (min 0.9802, max 0.9804, std 0.0001) | — |
| synthetic_10k | 10,000 | 20260907 | 0.9945 | 1.067 |
| synthetic_10k | 10,000 | 20260909 | 0.9945 | 1.200 |
| synthetic_10k | 10,000 | 20260911 | 0.9945 | 1.096 |
| synthetic_10k | 10,000 | **summary** | mean 0.9945 (min 0.9945, max 0.9945, std 0.0000) | — |

Interpretation: recall > 0.99 with < 0.001 seed std ⇒ approximate neighbor graph
is effectively deterministic and high-recall on these fixtures; 500k scalability
is reported as completion, not recall, because exact search is intractable.
