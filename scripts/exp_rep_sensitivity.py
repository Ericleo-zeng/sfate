#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_rep_sensitivity.py — P1-E25k fixturer=10/30/50/100
 r  sfate columns+ r=100
L∞ +  Spearmanoutput/rep_sensitivity.md / .json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SEED = 20260909
RS = [10, 30, 50, 100]


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

    t0 = time.perf_counter()
    P = build_transition_graph(X, k=30, random_state=SEED)
    t_graph = time.perf_counter() - t0

    def make_labels(r: int):
        labels = np.array([None] * n, dtype=object)
        for c in classes:
            idx = np.nonzero(states == c)[0]
            cen = X64[idx].mean(axis=0)
            dist = np.linalg.norm(X64[idx] - cen, axis=1)
            for i in idx[np.argsort(dist, kind="stable")[:r]]:
                labels[i] = c
        return labels

    sols = {}
    for r in RS:
        t0 = time.perf_counter()
        B, info = absorption_probabilities(P, make_labels(r), tol=1e-6,
                                           return_info=True)
        sols[r] = B
        print(f"r={r}: solve {time.perf_counter()-t0:.1f}s "
              f"={int(info['n_f64_fallback'])}", flush=True)

    ref = sols[100].astype(np.float64)
    rows = []
    for r in RS:
        B = sols[r].astype(np.float64)
        linf = float(np.abs(B - ref).max())
        rho = [float(spearmanr(B[:, j], ref[:, j]).statistic)
               for j in range(len(classes))]
        rows.append({"r": r, "Linf_vs_r100": linf,
                     "spearman_min": float(min(rho)),
                     "spearman_median": float(np.median(rho))})
        print(f"r={r}: L∞={linf:.2e} Spearman min={min(rho):.4f} "
              f"median={np.median(rho):.4f}", flush=True)

    Path("output/rep_sensitivity.json").write_text(json.dumps({
        "env": {"seed": SEED, "k": 30, "graph_s": round(t_graph, 1)},
        "reference": "r=100", "rows": rows}, ensure_ascii=False, indent=2))

    L = ["# E25k fixtureexp_rep_sensitivity.py\n",
         f"- k=30seed {SEED}columns+ = r=100\n",
         "| r | L∞ vs r=100 | Spearman min | Spearman median |", "|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['r']} | {r['Linf_vs_r100']:.2e} | "
                 f"{r['spearman_min']:.4f} | {r['spearman_median']:.4f} |")
    L.append("\n r=10/30/50  r=100  Spearman min ≥ 0.99"
             " modeling choice r ")
    Path("output/rep_sensitivity.md").write_text("\n".join(L), encoding="utf-8")
    print("written: output/rep_sensitivity.md", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
