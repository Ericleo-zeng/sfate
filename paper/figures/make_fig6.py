#!/usr/bin/env python3
"""make_fig6.py — Sample-level direction consistency between sfate and archived CellRank analysis."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
C_LAYER_MD = ROOT / "output/c_layer_report.md"
TSV = ROOT / "output/fate_probabilities/fate_by_sample_sfate_vs_cellrank.tsv"
OUT_PNG = Path(__file__).resolve().parent / "fig6.png"
OUT_PDF = Path(__file__).resolve().parent / "fig6.pdf"

C_CLIFF = "#a40000"
C_SF = "#2e7d32"
C_GRAY = "#c8c8c8"

AGES = [3, 6, 8]
CONTRASTS = ["5xFAD - control", "5xFAD;Cd28-cKO - 5xFAD"]
CONTRAST_SHORT = {"5xFAD - control": "5xFAD−control",
                  "5xFAD;Cd28-cKO - 5xFAD": "cKO−5xFAD"}


def parse_s4_scores():
    text = C_LAYER_MD.read_text(encoding="utf-8")
    m = re.search(r" delta \*\*(\d+)/(\d+)\*\*", text)
    total = (int(m.group(1)), int(m.group(2)))
    per = {}
    for ln in text.splitlines():
        mm = re.match(r"\| [\w_]+ \| (5xFAD−control|cKO−5xFAD) \| [\d,]+ \|"
                      r" .*? \| .*? \| (\d+)/(\d) \|", ln.strip())
        if mm:
            c, a, b = mm.group(1), int(mm.group(2)), int(mm.group(3))
            per.setdefault(c, [0, 0])
            per[c][0] += a
            per[c][1] += b
    return total, {k: (v[0], v[1]) for k, v in per.items()}


def main() -> int:
    df = pd.read_csv(TSV, sep="\t", comment="#")
    sm = df[df.record == "sample_mean"]
    ds = df[df.record == "genotype_delta_sign"]

    fates = sorted(sm["fate"].unique())
    grid = np.full((len(fates) * len(CONTRASTS), len(AGES)), np.nan)
    row_labels = []
    for ci, c in enumerate(CONTRASTS):
        for fi, f in enumerate(fates):
            r = ci * len(fates) + fi
            row_labels.append(f)
            for ai, a in enumerate(AGES):
                hit = ds[(ds.genotype == c) & (ds.fate == f) & (ds.age == a)]
                if len(hit) == 1:
                    grid[r, ai] = 1.0 if bool(hit["sign_agree"].iloc[0]) else 0.0
    agree_by_contrast = {
        CONTRAST_SHORT[c]: (
            int(np.nansum(grid[ci * 3:(ci + 1) * 3])),
            int(np.isfinite(grid[ci * 3:(ci + 1) * 3]).sum()))
        for ci, c in enumerate(CONTRASTS)}
    total_grid = int(np.nansum(grid))
    (s4_num, s4_den), s4_per = parse_s4_scores()
    assert (s4_num, s4_den) == (9, 15)
    assert agree_by_contrast["5xFAD−control"] == s4_per["5xFAD−control"]
    assert agree_by_contrast["cKO−5xFAD"] == s4_per["cKO−5xFAD"]
    assert sm["sample"].nunique() == 12 and len(fates) == 3

    fig = plt.figure(figsize=(15.4, 5.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 1.0, 1.0, 1.0], wspace=0.32)
    axa = fig.add_subplot(gs[0, 0])
    axb = [fig.add_subplot(gs[0, i + 1]) for i in range(3)]

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([C_CLIFF, C_SF])
    rgba = cmap(np.nan_to_num(grid, nan=0.0))
    rgba[~np.isfinite(grid)] = (*[v / 255 for v in (200, 200, 200)], 1.0)
    axa.imshow(rgba, aspect="auto", vmin=0, vmax=1)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            v = grid[r, c]
            txt = "✓" if v == 1 else ("✗" if v == 0 else "NA")
            tc = "white" if np.isfinite(v) else "#666666"
            axa.text(c, r, txt, ha="center", va="center", fontsize=11,
                     weight="bold", color=tc)
    axa.set_xticks(range(3), [f"{a}M" for a in AGES], fontsize=9.5)
    axa.set_yticks(range(len(row_labels)), row_labels, fontsize=8.6)
    axa.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    axa.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    axa.grid(which="minor", color="white", lw=1.6)
    axa.tick_params(which="minor", length=0)
    axa.text(2.75, 1.0, "5xFAD−control\n5/9", fontsize=9.5, va="center", color="#333333")
    axa.text(2.75, 4.0, "cKO−5xFAD\n4/6", fontsize=10, va="center",
             weight="bold", color=C_SF)
    axa.set_xlim(-0.5, 3.75)
    axa.set_title("(a) genotype-delta sign agreement\n"
                  "total 9/15 · cKO arm: 4/4 on IFN/C1q,\n"
                  "0/2 opposite on Homeostatic",
                  fontsize=10.5, weight="bold")
    axa.set_xlabel("age", fontsize=9.5)

    markers = {"control": "o", "5xFAD": "^", "5xFAD;Cd28-cKO": "s"}
    age_colors = {3: "#3465a4", 6: "#8f6202", 8: "#a40000"}
    for ax, f in zip(axb, fates):
        sub = sm[sm.fate == f]
        for gname, mk in markers.items():
            for age, col in age_colors.items():
                ss = sub[(sub.genotype == gname) & (sub.age == age)]
                if len(ss):
                    ax.scatter(ss["cellrank_mean"], ss["sfate_mean"],
                               marker=mk, s=64, color=col,
                               edgecolor="white", lw=0.8, zorder=3)
        x0, x1 = sub["cellrank_mean"].min(), sub["cellrank_mean"].max()
        y0, y1 = sub["sfate_mean"].min(), sub["sfate_mean"].max()
        px, py = (x1 - x0) * 0.35, (y1 - y0) * 0.35
        ax.set_xlim(x0 - px, x1 + px)
        ax.set_ylim(y0 - py, y1 + py)
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], ls="--", color="#888888", lw=1.1)
        ax.set_title(f, fontsize=10, weight="bold")
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8.5)
        ax.locator_params(axis="x", nbins=4)
    axb[0].set_ylabel("sfate sample_mean", fontsize=9.5)
    for ax in axb:
        ax.set_xlabel("CellRank sample_mean", fontsize=9.5)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=mk, ls="", color="#555555",
                      markeredgecolor="white", markersize=8, label=g)
               for g, mk in markers.items()]
    handles += [Line2D([], [], marker="o", ls="", color=col, markersize=8,
                       label=f"{a}M") for a, col in age_colors.items()]
    axb[0].legend(handles=handles, loc="lower left", fontsize=7.6,
                  framealpha=0.95, borderpad=0.6, labelspacing=0.4)
    axb[0].set_title(f"{fates[0]}\n(b) per-sample mean fate probability",
                     fontsize=10, weight="bold")

    fig.text(0.01, 0.028,
             "Sign agreement on genotype delta direction; grey = arm not "
             "comparable (no cKO sample at 8M).",
             fontsize=7.6, color="#666666")
    fig.text(0.01, 0.005,
             "Sources: output/c_layer_report.md (S4) · "
             "output/fate_probabilities/fate_by_sample_sfate_vs_cellrank.tsv",
             fontsize=6.8, color="#888888")
    fig.tight_layout(rect=(0.05, 0.06, 1, 1))
    fig.savefig(OUT_PNG, dpi=300, facecolor="white")
    fig.savefig(OUT_PDF, facecolor="white")
    print(f"written: {OUT_PNG}\nwritten: {OUT_PDF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
