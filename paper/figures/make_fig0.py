#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fig0.py — Figure 0: conceptual decomposition of the two pipelines.

Layout (round-4): shared upstream latent/kNN graph, then two optional
upstream-state-discovery routes that both feed the same downstream
absorption system.  Emphasizes that the absorption mathematics is shared
and that spectral terminal-state curation is optional when targets are
supplied externally.

Outputs: paper/figures/fig0.png (300 dpi) + fig0.pdf.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
M1_JSON = ROOT / "output/benchmark_m1_smoke.json"
OUT_PNG = Path(__file__).resolve().parent / "fig0.png"
OUT_PDF = Path(__file__).resolve().parent / "fig0.pdf"

C_LEFT = "#7f7f7f"          # gray for spectral/optional route
C_LEFT_BG = "#f2f2f2"
C_RIGHT = "#2e7d32"         # green for sfate route
C_RIGHT_BG = "#e4f0e2"
C_SHARED = "#1f4e79"        # blue for shared graph/absorption
C_SHARED_BG = "#e8f0f8"
C_TEXT = "#333333"


def box(ax, x, y, w, h, text, fc, ec, fs=9.0, weight="normal", linespacing=1.25):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.014",
        linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, weight=weight, color=C_TEXT, zorder=3,
            linespacing=linespacing)


def arrow(ax, x1, y1, x2, y2, color="#555555", lw=1.5, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=14, linewidth=lw,
                                 color=color, zorder=1))


def main() -> int:
    try:
        m1 = json.loads(M1_JSON.read_text())
        tiers = {int(r["n"]): float(r["rss_peak_mb_max"]) / 1000.0
                 for r in m1["results"]}
        gb500 = round(tiers[500_000], 2)
    except Exception as e:
        print(f"[warning] could not read 500k graph benchmark: {e}",
              file=sys.stderr)
        gb500 = 4.03

    fig = plt.figure(figsize=(12.0, 7.2))
    ax = fig.add_axes([0.0, 0.05, 1.0, 0.93])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---- shared upstream ----
    box_w, box_h = 0.36, 0.09
    cx = 0.5
    y_z = 0.86
    y_graph = 0.72
    box(ax, cx - box_w / 2, y_z, box_w, box_h,
        "Latent representation Z\n(n × d, here 30-d scVI)",
        C_SHARED_BG, C_SHARED, weight="bold")
    box(ax, cx - box_w / 2, y_graph, box_w, box_h,
        "Sparse kNN graph + diffusion transition matrix P\n(max-symmetrized, row-stochastic CSR)",
        C_SHARED_BG, C_SHARED, fs=8.8, weight="bold")
    arrow(ax, cx, y_z, cx, y_graph + box_h, color=C_SHARED)

    # ---- fork labels ----
    fork_y = 0.58
    left_x, right_x = 0.22, 0.78
    branch_w, branch_h = 0.34, 0.22

    # left branch: spectral route (optional)
    ax.add_patch(FancyBboxPatch(
        (left_x - branch_w / 2, fork_y - branch_h / 2),
        branch_w, branch_h,
        boxstyle="round,pad=0.012,rounding_size=0.016",
        facecolor=C_LEFT_BG, edgecolor=C_LEFT, linewidth=2.0,
        linestyle="--", zorder=2))
    ax.text(left_x, fork_y + 0.075, "CellRank / GPCCA spectral route",
            ha="center", va="center", fontsize=10.5, weight="bold",
            color=C_LEFT, zorder=3)
    ax.text(left_x, fork_y + 0.015,
            "partial real Schur decomposition\n↓\n"
            "metastable macrostates & memberships χ\n↓\n"
            "terminal-state identification (stability criterion)",
            ha="center", va="center", fontsize=8.7, color=C_TEXT,
            linespacing=1.2, zorder=3)
    ax.text(left_x, fork_y - 0.078,
            "optional when targets are externally specified;\n"
            "requires PETSc/SLEPc for sparse route;\n"
            "O(n²) if dense Brandts fallback",
            ha="center", va="center", fontsize=7.0, color=C_LEFT,
            linespacing=1.15, zorder=3)

    # right branch: sfate explicit targets
    ax.add_patch(FancyBboxPatch(
        (right_x - branch_w / 2, fork_y - branch_h / 2),
        branch_w, branch_h,
        boxstyle="round,pad=0.012,rounding_size=0.016",
        facecolor=C_RIGHT_BG, edgecolor=C_RIGHT, linewidth=2.0,
        zorder=2))
    ax.text(right_x, fork_y + 0.075, "sfate explicit-target route",
            ha="center", va="center", fontsize=10.5, weight="bold",
            color=C_RIGHT, zorder=3)
    ax.text(right_x, fork_y + 0.015,
            "annotation-defined target states\n↓\n"
            "centroid-nearest r absorbing representatives",
            ha="center", va="center", fontsize=8.7, color=C_TEXT,
            linespacing=1.2, zorder=3)

    # arrows from graph to branches
    arrow(ax, cx - 0.08, y_graph, left_x, fork_y + branch_h / 2,
          color="#777777")
    arrow(ax, cx + 0.08, y_graph, right_x, fork_y + branch_h / 2,
          color=C_RIGHT)

    # fork subtitle (centered in the gap between branches and absorbing chain)
    ax.text(cx, 0.43,
            "Upstream state discovery is optional; the downstream absorption "
            "system is the shared mathematical object.",
            ha="center", va="center", fontsize=9.5, color="#444444",
            style="italic", zorder=3,
            bbox=dict(facecolor="white", alpha=0.95, edgecolor="none",
                      pad=3.0))

    # ---- shared downstream ----
    y_abs = 0.30
    y_system = 0.18
    y_fate = 0.06
    down_w, down_h = 0.44, 0.09
    box(ax, cx - down_w / 2, y_abs, down_w, down_h,
        "Absorbing chain  P_abs = (Q  R; 0  I)",
        C_SHARED_BG, C_SHARED, fs=9.5, weight="bold")
    box(ax, cx - down_w / 2, y_system, down_w, down_h,
        "Linear absorption system  (I − Q) X = R",
        C_SHARED_BG, C_SHARED, fs=10.0, weight="bold")
    box(ax, cx - down_w / 2, y_fate, down_w, down_h,
        "Fate probabilities  F",
        C_SHARED_BG, C_SHARED, fs=10.0, weight="bold")

    # arrows from branches to absorbing chain (routed around center subtitle)
    arrow(ax, left_x, fork_y - branch_h / 2, cx - 0.14, y_abs + down_h,
          color="#777777")
    arrow(ax, right_x, fork_y - branch_h / 2, cx + 0.14, y_abs + down_h,
          color=C_RIGHT)
    # vertical chain
    arrow(ax, cx, y_abs, cx, y_system + down_h, color=C_SHARED)
    arrow(ax, cx, y_system, cx, y_fate + down_h, color=C_SHARED)

    # ---- memory annotations ----
    ax.text(0.02, 0.70,
            "~405 GB @ 75k\n[BACK-CALCULATED]\n(dense Brandts fallback)",
            ha="left", va="center", fontsize=7.8, color=C_LEFT,
            linespacing=1.15,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                      pad=2), zorder=4)
    ax.text(0.98, 0.70,
            f"{gb500:.2f} GB @ 500k\n(graph-construction only)",
            ha="right", va="center", fontsize=7.8, color=C_RIGHT,
            linespacing=1.15,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                      pad=2), zorder=4)
    ax.text(0.98, 0.12,
            "~1.76 GiB @ 75k\n[MEASURED]\n(full sfate pipeline)",
            ha="right", va="center", fontsize=7.8, color=C_RIGHT,
            linespacing=1.15,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                      pad=2), zorder=4)

    fig.savefig(OUT_PNG, dpi=300, facecolor="white",
                bbox_inches="tight", pad_inches=0.18)
    fig.savefig(OUT_PDF, facecolor="white",
                bbox_inches="tight", pad_inches=0.18)
    print(f"written: {OUT_PNG}\nwritten: {OUT_PDF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
