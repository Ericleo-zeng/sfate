#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_rep_sensitivity_v2.py — P1-②5k fixturer=10/20/30/50/100/200
columns+ = r=200
output/rep_sensitivity_v2.md / .json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SEED = 20260909
RS = [10, 20, 30, 50, 100, 200]


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
    solve_times = {}
    for r in RS:
        t0 = time.perf_counter()
        B, info = absorption_probabilities(P, make_labels(r), tol=1e-6,
                                           return_info=True)
        sols[r] = B
        solve_times[r] = time.perf_counter() - t0
        print(f"r={r}: solve {solve_times[r]:.1f}s "
              f"fallback={int(info['n_f64_fallback'])}", flush=True)

    ref = sols[200].astype(np.float64)
    rows = []
    for r in RS:
        B = sols[r].astype(np.float64)
        diff = np.abs(B - ref)
        linf = float(diff.max())
        mae = float(diff.mean())
        p95 = float(np.percentile(diff, 95))
        rho = [float(spearmanr(B[:, j], ref[:, j]).statistic)
               for j in range(len(classes))]
        mean_prob_diff = [float(np.abs(B[:, j].mean() - ref[:, j].mean()))
                          for j in range(len(classes))]
        rows.append({
            "r": r,
            "Linf_vs_r200": linf,
            "MAE_vs_r200": mae,
            "p95_abs_diff": p95,
            "spearman_min": float(min(rho)),
            "spearman_median": float(np.median(rho)),
            "spearman_per_state": rho,
            "mean_prob_diff_per_state": mean_prob_diff,
            "solve_s": round(solve_times[r], 1),
        })
        print(f"r={r}: L∞={linf:.2e} MAE={mae:.2e} p95={p95:.2e} "
              f"Spearman min={min(rho):.4f} median={np.median(rho):.4f}",
              flush=True)

    out = {
        "env": {"seed": SEED, "k": 30, "graph_s": round(t_graph, 1),
                "reference": "r=200"},
        "classes": classes.tolist(),
        "rows": rows,
    }
    Path("output/rep_sensitivity_v2.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2))

    lines = [
        "# P1-②5k fixtureexp_rep_sensitivity_v2.py\n",
        f"- k=30seed {SEED}columns+ = r=200\n",
        "| r | L∞ vs r=200 | MAE vs r=200 | p95 abs diff | Spearman min | Spearman median | solve_s |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['r']} | {r['Linf_vs_r200']:.2e} | {r['MAE_vs_r200']:.2e} | "
            f"{r['p95_abs_diff']:.2e} | {r['spearman_min']:.4f} | "
            f"{r['spearman_median']:.4f} | {r['solve_s']} |"
        )
    lines.append("\n## vs r=200\n")
    lines.append("| r | " + " | ".join(classes) + " |")
    lines.append("|---|" + "---|" * len(classes))
    for r in rows:
        vals = " | ".join(f"{d:.4f}" for d in r["mean_prob_diff_per_state"])
        lines.append(f"| {r['r']} | {vals} |")
    lines.append("\nlandscape  r r=30  CellRank ")
    Path("output/rep_sensitivity_v2.md").write_text("\n".join(lines), encoding="utf-8")
    print("written: output/rep_sensitivity_v2.md", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
