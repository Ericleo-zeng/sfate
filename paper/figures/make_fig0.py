#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fig0.py — Fig0: pipeline schematic graphical abstract

 paper/PLAN.md §3 Fig0  docs/step12_postmortem.md §5
 Fig1 assert
- 7272·n²         : docs/cellrank-bottleneck.md §2.2
- 1.89 GB @ 75ksfate     : output/c_layer_report.mdC11
- 4.33 GB @ 500ksfate      : output/benchmark_m1_smoke.json
- 794.7 / 7.090        : output/benchmark_m1_smoke.md §
- 405 GB @ 75kbrandts      : 72 × 74,984² Bdocs/step12_postmortem.md §3
-            : output/benchmark_m1_smoke.json Fig1

paper/figures/fig0.png300 dpi16:9+ fig0.pdf
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
BOTTLENECK_MD = ROOT / "docs/cellrank-bottleneck.md"
M1_MD = ROOT / "output/benchmark_m1_smoke.md"
M1_JSON = ROOT / "output/benchmark_m1_smoke.json"
C_LAYER_MD = ROOT / "output/c_layer_report.md"
OUT_PNG = Path(__file__).resolve().parent / "fig0.png"
OUT_PDF = Path(__file__).resolve().parent / "fig0.pdf"

C_CLIFF = "#a40000"
C_CLIFF_BG = "#f7e3e3"
C_SF = "#2e7d32"
C_SF_BG = "#e4f0e2"
C_GRAY = "#666666"

ASSERT_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    ASSERT_RESULTS.append((name, ok, detail))
    print(f"[assert {'PASS' if ok else 'FAIL'}] {name}"
          + (f"  ({detail})" if detail else ""))
    return ok


def box(ax, x, y, w, h, text, fc, ec, fs=9.5, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.014",
        linewidth=1.5, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, weight=weight, zorder=3, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color="#555555"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=15, linewidth=1.7,
                                 color=color, zorder=1))


def main() -> int:
    try:
        # ----  Fig1 ----
        coef72 = float(re.search(r"≈\s*\*?\*?\s*(72)·n² ",
                                 BOTTLENECK_MD.read_text(encoding="utf-8")).group(1))
        gb75_full = float(re.search(r" RSS ≈([\d.]+)\s*GB",
                                    C_LAYER_MD.read_text(encoding="utf-8")).group(1))
        m1 = json.loads(M1_JSON.read_text())
        tiers = {int(r["n"]): float(r["rss_peak_mb_max"]) / 1000.0
                 for r in m1["results"]}
        gb500 = round(tiers[500_000], 2)
        mm = re.search(r"mem\(n\) = ([\d.]+) \+ ([\d.]+) MB/",
                       M1_MD.read_text(encoding="utf-8"))
        intercept, slope = float(mm.group(1)), float(mm.group(2))
        gb405 = coef72 * 74_984**2 / 1e9

        ok = all([
            check("F0-1: 72  == 72", coef72 == 72.0, f"{coef72}"),
            check("F0-2: 1.89GB@75k ", abs(gb75_full - 1.89) < 0.005,
                  f"{gb75_full}"),
            check("F0-3: 4.33GB@500k ", abs(gb500 - 4.33) < 0.005, f"{gb500}"),
            check("F0-4:  794.7/7.090 ",
                  abs(intercept - 794.7) < 0.05 and abs(slope - 7.090) < 0.0005,
                  f"{intercept}/{slope}"),
        ])
        if not ok:
            raise ValueError("assert ")

        # =================  =================
        fig = plt.figure(figsize=(12.8, 7.2))
        ax = fig.add_axes([0, 0.06, 1, 0.94])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        W, H = 0.34, 0.075
        XL, XR = 0.055, 0.605
        ys = [0.86, 0.70, 0.54, 0.38, 0.24]

        def caption(x, y_top, lines, color, weight="normal", fs=8.2):
            ax.text(x + W / 2, y_top - 0.028, lines, ha="center", va="center",
                    fontsize=fs, color=color, weight=weight, linespacing=1.3,
                    bbox=dict(facecolor="white", alpha=0.88,
                              edgecolor="none", pad=1.4), zorder=4)

        # ----  CellRank ----
        ax.text(XL + W / 2, 0.942, "CellRank 2 / GPCCA", ha="center",
                fontsize=13, weight="bold", color=C_CLIFF)
        left_nodes = [
            "RNA velocity\n+ VelocityKernel (n×g dense f64)",
            "GPCCA\nSchur decomposition",
            "automatic macrostate\n& terminal-state curation",
            "fate probabilities\nGMRES + ILU",
        ]
        left_ys = ys[:4]
        for i, t in enumerate(left_nodes):
            box(ax, XL, left_ys[i], W, H, t, C_CLIFF_BG, C_CLIFF,
                weight="bold" if i == 1 else "normal")
            if i:
                arrow(ax, XL + W / 2, left_ys[i - 1], XL + W / 2,
                      left_ys[i] + H, color=C_CLIFF)
        caption(XL, left_ys[1],
                f"peak ≈ {coef72:.0f}·n² B (Θ(n²)) · "
                f"n=75k → ~{gb405:.0f} GB (infeasible)",
                C_CLIFF, weight="bold", fs=8.6)

        #
        axl = fig.add_axes([0.075, 0.088, 0.26, 0.125])
        n_grid = np.linspace(0, 1e5, 100)
        axl.plot(n_grid / 1000, coef72 * n_grid**2 / 1e9, color=C_CLIFF, lw=2)
        axl.set_title("memory: 72·n² (Θ(n²))", fontsize=8.5, color=C_CLIFF)
        axl.set_xlabel("n (×1000 cells)", fontsize=7.5)
        axl.set_ylabel("GB", fontsize=7.5)
        axl.tick_params(labelsize=7)
        axl.set_ylim(0, 760)
        axl.axhline(30, color="#999999", ls=":", lw=1)
        axl.text(2, 33, "30 GB", fontsize=6.5, color="#999999")
        axl.plot([100], [coef72 * 1e5**2 / 1e9], "o", ms=4, color=C_CLIFF)
        axl.annotate(f"{coef72 * 1e5**2 / 1e9:.0f} GB @ 100k",
                     xy=(100, coef72 * 1e5**2 / 1e9), xytext=(52, 690),
                     fontsize=6.5, color=C_CLIFF,
                     arrowprops=dict(arrowstyle="->", color=C_CLIFF, lw=0.8))

        # ----  sfate ----
        ax.text(XR + W / 2, 0.942, "scalable-fate (sfate)", ha="center",
                fontsize=13, weight="bold", color=C_SF)
        right_nodes = [
            "scVI latent (n × 30, f32)\n[data graph]",
            "approximate kNN (pynndescent, k=30)\nGaussian · max-symmetrized · CSR f32\n[data graph: diffusion-type, no velocity direction]",
            "annotation-driven absorbing states\n(no curation) [target semantics]",
            "(I−Q)X = R\n[computational core]",
            "column-wise GMRES + blocking\n[scaling strategy]",
        ]
        ys5 = [0.88, 0.72, 0.56, 0.40, 0.25]
        for i, t in enumerate(right_nodes):
            box(ax, XR, ys5[i], W, H, t, C_SF_BG, C_SF,
                fs=8.6,
                weight="bold" if i == 2 else "normal")
            if i:
                arrow(ax, XR + W / 2, ys5[i - 1], XR + W / 2, ys5[i] + H,
                      color=C_SF)
        caption(XR, ys5[2],
                f"mem(n) = {intercept:.1f} + {slope:.3f} MB per 1000 cells "
                f"(Θ(n))", C_SF, weight="bold", fs=8.6)
        caption(XR, ys5[4],
                f"n=75k → ~{gb75_full:.1f} GB · "
                f"n=500k → {gb500:.2f} GB (measured)", C_SF, fs=8.6)

        #  +
        axr = fig.add_axes([0.665, 0.088, 0.26, 0.125])
        ns = np.array(sorted(tiers), dtype=float) / 1000.0
        vs = np.array([tiers[int(n * 1000)] for n in ns])
        n_fit = np.linspace(0, 550, 10)
        axr.plot(n_fit, (intercept + slope * n_fit) / 1000.0, color=C_SF,
                 lw=2)
        axr.plot(ns, vs, "o", color=C_SF, ms=5)
        offsets = [(-4, 2), (10, 6), (6, 5), (-12, 6)]
        has = ["right", "center", "center", "center"]
        for (n_, v_), (dx, dy), ha_ in zip(zip(ns, vs), offsets, has):
            axr.annotate(f"{int(n_)}k", (n_, v_), textcoords="offset points",
                         xytext=(dx, dy), fontsize=6.5, ha=ha_,
                         color=C_SF)
        axr.set_title("memory: linear (4 tiers measured)", fontsize=8.5,
                      color=C_SF)
        axr.set_xlabel("n (×1000 cells)", fontsize=7.5)
        axr.set_ylabel("GB", fontsize=7.5)
        axr.tick_params(labelsize=7)
        axr.set_ylim(0, 5.2)

        # ----  0.24–0.935 0.5875----
        ax.add_patch(FancyBboxPatch((0.455, 0.3875), 0.09, 0.40,
                                    boxstyle="round,pad=0.008",
                                    facecolor="#f2f2f2", edgecolor="#aaaaaa",
                                    lw=1.2, zorder=2))
        ax.text(0.5, 0.7075, "Memory", ha="center", fontsize=9, weight="bold")
        ax.text(0.5, 0.6525, "Θ(n²)", ha="center", fontsize=10, color=C_CLIFF,
                weight="bold")
        ax.text(0.5, 0.6125, "vs", ha="center", fontsize=8, color=C_GRAY)
        ax.text(0.5, 0.5775, "Θ(n)", ha="center", fontsize=10, color=C_SF,
                weight="bold")
        ax.text(0.5, 0.5125, "Terminal\nstates", ha="center", fontsize=9,
                weight="bold", linespacing=1.25)
        ax.text(0.5, 0.4525, "curation", ha="center", fontsize=8.5,
                color=C_CLIFF)
        ax.text(0.5, 0.4225, "vs", ha="center", fontsize=8, color=C_GRAY)
        ax.text(0.5, 0.3925, "annotation", ha="center", fontsize=8.5,
                color=C_SF)

        fig.text(0.5, 0.978,
                 "Two routes to single-cell fate probabilities",
                 ha="center", fontsize=14, weight="bold")

        # ----  footer ----
        fig.text(0.01, 0.028,
                 "Schematic; memory annotations from cited benchmarks.",
                 fontsize=7.6, color="#666666")
        fig.text(0.01, 0.005,
                 "Sources: docs/cellrank-bottleneck.md · "
                 "docs/step12_postmortem.md · output/benchmark_m1_smoke.md · "
                 "output/c_layer_report.md",
                 fontsize=6.8, color="#888888")
        fig.savefig(OUT_PNG, dpi=300, facecolor="white",
                    bbox_inches="tight", pad_inches=0.18)
        fig.savefig(OUT_PDF, facecolor="white",
                    bbox_inches="tight", pad_inches=0.18)
        print(f"written: {OUT_PNG}\nwritten: {OUT_PDF}")
    except Exception as e:  # noqa: BLE001
        print(f"[warning] fig0 /assert : {e}", file=sys.stderr)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.55, "Fig0 [pending]", ha="center", fontsize=14,
                color=C_CLIFF, weight="bold")
        ax.text(0.5, 0.4, f"data parse/assert failed:\n{e}", ha="center",
                fontsize=8, color="#888888")
        fig.savefig(OUT_PNG, dpi=300, facecolor="white")
        print(f"written(pending): {OUT_PNG}")

    n_fail = sum(1 for _, ok, _ in ASSERT_RESULTS if not ok)
    print(f"assert : {len(ASSERT_RESULTS) - n_fail}/{len(ASSERT_RESULTS)} PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
