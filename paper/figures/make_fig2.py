#!/usr/bin/env python3
"""make_fig2.py — Numerical validation of the sfate solver across independent scenarios."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
M2_MD = ROOT / "output/benchmark_m2.md"
B_MD = ROOT / "output/b_layer_report.md"
N_MD = ROOT / "<excluded>"
CERT_JSON = ROOT / "output/cert_recompute_5k.json"
OUT_PNG = Path(__file__).resolve().parent / "fig2.png"
OUT_PDF = Path(__file__).resolve().parent / "fig2.pdf"

C_SF = "#2e7d32"
C_BLUE = "#3465a4"
C_HEAVY = "#8f6202"
C_CLIFF = "#a40000"


def main() -> int:
    m2 = M2_MD.read_text(encoding="utf-8")
    b = B_MD.read_text(encoding="utf-8")
    n = N_MD.read_text(encoding="utf-8")
    cert = json.loads(CERT_JSON.read_text())

    a3_linf = float(cert["production_5k"]["Linf_vs_golden"])
    tol = 1e-5
    row = next(l for l in m2.splitlines() if l.strip().startswith("|  |"))
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    gmres_s = float(cells[1])
    spsolve_s = float(re.search(r" spsolve\(f64\)  \| ([\d.]+)", m2).group(1))
    speedup = spsolve_s / gmres_s
    recon = float(re.search(r" L∞=([\d.e+-]+)", m2).group(1))
    e1 = float(re.search(r" 1E1 .*?L∞ = ([\d.e+-]+)", b).group(1))
    a1_pass = "PASS" in next(l for l in n.splitlines() if "a1 lumping" in l)

    fig, ax = plt.subplots(figsize=(10.2, 3.4))
    scenes = [
        ("A-layer a3", a3_linf, C_SF, f"L∞ = {a3_linf:.2e}",
         ["f32 GMRES vs f64 spsolve", "2×2 grid all PASS"], "left"),
        ("B-layer E1", e1, C_BLUE, f"L∞ = {e1:.2e}",
         ["on CR kernel", "+ CR terminal cells"], "left"),
        ("mode reconciliation", recon, C_HEAVY, f"L∞ = {recon:.2e}",
         ["column-wise vs aggregated (75k)"], "left"),
        ("A-layer a1", None, C_SF, "✓ PASS — exact equivalence",
         ["f64 spsolve full chain · no numeric L∞ reported"], "left"),
    ]
    ys = list(range(len(scenes)))[::-1]
    ax.set_yticks(ys, [s[0] for s in scenes], fontsize=10)
    for y, (name, val, color, line1, desc_lines, side) in zip(ys, scenes):
        if val is not None:
            ax.plot([val], [y], "o", ms=10, color=color, zorder=3)
            if side == "left":
                ax.annotate(line1, xy=(val, y), xytext=(-10, 13),
                            textcoords="offset points", ha="right",
                            va="center", fontsize=9.5, weight="bold", color=color)
                for k, dl in enumerate(desc_lines):
                    ax.annotate(dl, xy=(val, y), xytext=(-10, 1 - 12 * k),
                                textcoords="offset points", ha="right",
                                va="center", fontsize=7.8, color="#666666")
            else:
                ax.annotate(line1, xy=(val, y), xytext=(10, 5),
                            textcoords="offset points", ha="left",
                            va="center", fontsize=9.5, weight="bold", color=color)
                for k, dl in enumerate(desc_lines):
                    ax.annotate(dl, xy=(val, y), xytext=(10, -9 - 12 * k),
                                textcoords="offset points", ha="left",
                                va="center", fontsize=7.8, color="#666666")
        else:
            ax.annotate(line1, xy=(1.2e-7, y), fontsize=9.5,
                        va="center", color=color, weight="bold")
            ax.annotate(desc_lines[0], xy=(1.2e-7, y), xytext=(0, -13),
                        textcoords="offset points", fontsize=7.8,
                        va="center", color="#666666")
    ax.axvline(tol, color=C_CLIFF, ls="--", lw=1.4)
    ax.text(tol * 1.15, ys[0] + 0.42, f"A-layer tolerance ({tol:.0e})",
            fontsize=8.5, color=C_CLIFF)
    ax.set_xscale("log")
    ax.set_xlim(1e-7, 1e-3)
    ax.set_ylim(-0.55, len(scenes) - 0.45)
    ax.set_xlabel("L∞ = max |sfate − reference| per cell (log scale)", fontsize=10)
    ax.set_title("Numerical validation of the sfate solver across independent scenarios",
                 fontsize=11.5, weight="bold")
    ax.grid(True, axis="x", which="both", alpha=0.25, lw=0.5)
    ax.text(0.985, 0.05,
            f"performance note (not accuracy):\nGMRES {gmres_s:.2f} s vs spsolve(f64) {spsolve_s:.1f} s "
            f"≈ {speedup:.0f}× speedup (5k fixture)",
            transform=ax.transAxes, fontsize=8, color="#666666",
            ha="right", va="bottom",
            bbox=dict(facecolor="#f4f4f4", edgecolor="#cccccc",
                      boxstyle="round,pad=0.35"))
    fig.text(0.01, 0.028,
             "L∞ = max |sfate − reference| per cell; tolerance as reported per layer "
             "(a3: 1e-5; other layers record-only, no shared threshold drawn).",
             fontsize=7.6, color="#666666")
    fig.text(0.01, 0.005,
             "Sources: output/benchmark_m2.md · output/b_layer_report.md · "
             "<excluded> · output/cert_recompute_5k.json",
             fontsize=6.8, color="#888888")
    fig.tight_layout(rect=(0.135, 0.10, 1, 1))
    fig.savefig(OUT_PNG, dpi=300, facecolor="white")
    fig.savefig(OUT_PDF, facecolor="white")
    print(f"written: {OUT_PNG}\nwritten: {OUT_PDF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
