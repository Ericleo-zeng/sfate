#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_knn_k_sensitivity.py — P1-E5kNN k=15/30/50 5k fixture
 k  +  k=30 L∞ +  Spearman
output/knn_k_sensitivity.md / .json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SEED = 20260909
KS = [15, 30, 50]


def main() -> int:
    sys.path.insert(0, "src")
    import anndata as ad
    from sfate import absorption_probabilities, build_transition_graph

    fx = ad.read_h5ad("output/fixture_5k.h5ad")
    X = np.asarray(fx.obsm["X_scVI"], dtype=np.float32)
    states = np.asarray(fx.obs["microglia_state"].astype(str))
    n = X.shape[0]
    X64 = X.astype(np.float64)
    classes = np.unique(states)

    labels = np.array([None] * n, dtype=object)
    for c in classes:
        idx = np.nonzero(states == c)[0]
        cen = X64[idx].mean(axis=0)
        dist = np.linalg.norm(X64[idx] - cen, axis=1)
        for i in idx[np.argsort(dist, kind="stable")[:30]]:
            labels[i] = c

    sols = {}
    for k in KS:
        t0 = time.perf_counter()
        P = build_transition_graph(X, k=k, random_state=SEED)
        tg = time.perf_counter() - t0
        t0 = time.perf_counter()
        B, info = absorption_probabilities(P, labels, tol=1e-6,
                                           return_info=True)
        ts = time.perf_counter() - t0
        sols[k] = B
        print(f"k={k}: graph {tg:.1f}s solve {ts:.1f}s nnz={P.nnz} "
              f"={int(info['n_f64_fallback'])}", flush=True)

    ref = sols[30].astype(np.float64)
    rows = []
    for k in KS:
        B = sols[k].astype(np.float64)
        linf = float(np.abs(B - ref).max())
        rho = [float(spearmanr(B[:, j], ref[:, j]).statistic)
               for j in range(len(classes))]
        rows.append({"k": k, "Linf_vs_k30": linf,
                     "spearman_min": float(min(rho)),
                     "spearman_median": float(np.median(rho))})
        print(f"k={k}: L∞={linf:.2e} Spearman min={min(rho):.4f} "
              f"median={np.median(rho):.4f}", flush=True)

    Path("output/knn_k_sensitivity.json").write_text(json.dumps({
        "env": {"seed": SEED, "r": 30}, "reference": "k=30", "rows": rows},
        ensure_ascii=False, indent=2))

    L = ["# E5kNN k 5k fixtureexp_knn_k_sensitivity.py\n",
         f"- r=30 /seed {SEED} = k=30\n",
         "| k | L∞ vs k=30 | Spearman min | Spearman median |", "|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['k']} | {r['Linf_vs_k30']:.2e} | "
                 f"{r['spearman_min']:.4f} | {r['spearman_median']:.4f} |")
    Path("output/knn_k_sensitivity.md").write_text("\n".join(L), encoding="utf-8")
    print("written: output/knn_k_sensitivity.md", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
