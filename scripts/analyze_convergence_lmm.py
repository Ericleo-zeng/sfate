#!/usr/bin/env python3
"""analyze_convergence_lmm.py —  r/k  JSON

1. LMM: Spearman ~ log2(r) + (1|state) on rep_sensitivity_v2.json.
2.  <0.05 / <0.02  r
3. k
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"


def lmm_rep_sensitivity() -> dict:
    data = json.loads((OUT / "rep_sensitivity_v2.json").read_text(encoding="utf-8"))
    classes = data["classes"]
    rows = []
    for row in data["rows"]:
        if row["r"] == 200:
            continue
        for j, c in enumerate(classes):
            rows.append({
                "state": c,
                "r": row["r"],
                "log2r": np.log2(row["r"]),
                "spearman": row["spearman_per_state"][j],
                "mean_prob_diff": row["mean_prob_diff_per_state"][j],
            })
    df = pd.DataFrame(rows)

    # MixedLM via formula
    model = smf.mixedlm("spearman ~ log2r", df, groups=df["state"])
    result = model.fit()
    beta = float(result.params["log2r"])
    ci = result.conf_int().loc["log2r"].values
    beta_lo, beta_hi = float(ci[0]), float(ci[1])

    # thresholds across all states
    thresh = {}
    for thr in (0.05, 0.02):
        ok_r = []
        for r, grp in df.groupby("r"):
            if (grp["mean_prob_diff"] < thr).all():
                ok_r.append(r)
        thresh[thr] = min(ok_r) if ok_r else None

    return {
        "n_obs": int(len(df)),
        "beta_per_doubling": beta,
        "beta_95ci_lo": beta_lo,
        "beta_95ci_hi": beta_hi,
        "r_mean_diff_below_0.05": thresh[0.05],
        "r_mean_diff_below_0.02": thresh[0.02],
        "model_summary": result.summary().as_text(),
    }


def summarize_k_sensitivity() -> dict:
    data = json.loads((OUT / "knn_k_sensitivity_v2.json").read_text(encoding="utf-8"))
    ref_row = next(r for r in data["rows"] if r["k"] == 30)
    others = [r for r in data["rows"] if r["k"] != 30]
    return {
        "reference_k": 30,
        "reference_nnz": ref_row["nnz"],
        "rows": others,
        "mae_range": [float(min(r["MAE_vs_k30"] for r in others)),
                      float(max(r["MAE_vs_k30"] for r in others))],
        "note": "k=100 shows non-monotonic Spearman decline relative to k=50",
    }


def main() -> int:
    lmm = lmm_rep_sensitivity()
    ks = summarize_k_sensitivity()
    result = {"rep_sensitivity_lmm": lmm, "knn_k_sensitivity_summary": ks}
    (OUT / "convergence_lmm.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Convergence & sensitivity re-analysisanalyze_convergence_lmm.py",
        "",
        "## r  LMMSpearman ~ log2(r) + (1|state))",
        "",
        f"- n = {lmm['n_obs']}6 states × 5 r levels",
        f"- β per doubling = {lmm['beta_per_doubling']:.4f} "
        f"(95% CI {lmm['beta_95ci_lo']:.4f}–{lmm['beta_95ci_hi']:.4f})",
        f"- per-state mean |ΔP| first < 0.05 at r = {lmm['r_mean_diff_below_0.05']}",
        f"- per-state mean |ΔP| first < 0.02 at r = {lmm['r_mean_diff_below_0.02']}",
        "",
        "```",
        lmm["model_summary"],
        "```",
        "",
        "## kNN k  k=30",
        "",
        f"- MAE range vs k=30: {ks['mae_range'][0]:.2e}–{ks['mae_range'][1]:.2e}",
        "",
        "| k | graph_s | solve_s | nnz | nnz/(n·k) | L∞ vs k=30 | MAE vs k=30 | Spearman min | Spearman median |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in ks["rows"]:
        lines.append(
            f"| {r['k']} | {r['graph_s']} | {r['solve_s']} | {r['nnz']:,} | "
            f"{r['nnz_per_nk']} | {r['Linf_vs_k30']:.2e} | {r['MAE_vs_k30']:.2e} | "
            f"{r['spearman_min']:.4f} | {r['spearman_median']:.4f} |"
        )
    lines += ["", f"- Note: {ks['note']}", ""]
    (OUT / "convergence_lmm.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
