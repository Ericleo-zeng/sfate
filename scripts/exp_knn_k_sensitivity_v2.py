#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_knn_k_sensitivity_v2.py — P1-③kNN k 5k fixturek=15/30/50/100
r=30 / = k=30
output/knn_k_sensitivity_v2.md / .json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SEED = 20260909
KS = [15, 30, 50, 100]
R = 30


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
        for i in idx[np.argsort(dist, kind="stable")[:R]]:
            labels[i] = c

    sols = {}
    graph_times = {}
    solve_times = {}
    nnzs = {}
    for k in KS:
        t0 = time.perf_counter()
        P = build_transition_graph(X, k=k, random_state=SEED)
        graph_times[k] = time.perf_counter() - t0
        t0 = time.perf_counter()
        B, info = absorption_probabilities(P, labels, tol=1e-6,
                                           return_info=True)
        solve_times[k] = time.perf_counter() - t0
        sols[k] = B
        nnzs[k] = P.nnz
        print(f"k={k}: graph {graph_times[k]:.1f}s solve {solve_times[k]:.1f}s nnz={P.nnz} "
              f"fallback={int(info['n_f64_fallback'])}", flush=True)

    ref = sols[30].astype(np.float64)
    rows = []
    for k in KS:
        B64 = sols[k].astype(np.float64)
        diff = np.abs(B64 - ref)
        linf = float(diff.max())
        mae = float(diff.mean())
        rho = [float(spearmanr(B64[:, j], ref[:, j]).statistic)
               for j in range(len(classes))]
        rows.append({
            "k": k,
            "graph_s": round(graph_times[k], 2),
            "solve_s": round(solve_times[k], 2),
            "nnz": int(nnzs[k]),
            "nnz_per_nk": round(nnzs[k] / (n * k), 4),
            "Linf_vs_k30": linf,
            "MAE_vs_k30": mae,
            "spearman_min": float(min(rho)),
            "spearman_median": float(np.median(rho)),
        })
        print(f"k={k}: L∞={linf:.2e} MAE={mae:.2e} "
              f"Spearman min={min(rho):.4f} median={np.median(rho):.4f}",
              flush=True)

    out = {
        "env": {"seed": SEED, "r": R, "reference": "k=30"},
        "classes": classes.tolist(),
        "rows": rows,
    }
    Path("output/knn_k_sensitivity_v2.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2))

    lines = [
        "# P1-③kNN k 5k fixtureexp_knn_k_sensitivity_v2.py\n",
        f"- r={R} /seed {SEED} = k=30\n",
        "| k | graph_s | solve_s | nnz | nnz/(n·k) | L∞ vs k=30 | MAE vs k=30 | Spearman min | Spearman median |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['k']} | {r['graph_s']} | {r['solve_s']} | {r['nnz']:,} | "
            f"{r['nnz_per_nk']} | {r['Linf_vs_k30']:.2e} | {r['MAE_vs_k30']:.2e} | "
            f"{r['spearman_min']:.4f} | {r['spearman_median']:.4f} |"
        )
    Path("output/knn_k_sensitivity_v2.md").write_text("\n".join(lines), encoding="utf-8")
    print("written: output/knn_k_sensitivity_v2.md", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
