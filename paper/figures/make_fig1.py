#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fig1.py — Fig1: log-logbrandts  Schur vs
krylov-Schur vs sfate


- brandts tracemalloc  288.2MB@2000 / 1800.3MB@5000
  72·n² n=10000  OOM killer 4GB
  research/exp_t7/results.csvexp4  parse_brandts()
   research/T7_experiments.md §exp4
- krylov /usr/bin/time -v Maximum RSSKiB→GB
  output/slepc_krylov_100k_time_v.txt1,607,860 KiB
  output/slepc_krylov_500k_time_v.txt3,849,808 KiB
- sfate RSS  MB****+
  output/benchmark_m1_smoke.jsonfixture_5k_real / synthetic_10k /
  real_75k_full / synthetic_500k
- sfate 75k time -v  ≈1.89GB
  output/c_layer_report.md §C11
- 405GB@75k 72 × 74,984² B docs/step12_postmortem.md §3
- 30GB docs/step12_postmortem.md §3 ~30GB

paper/figures/fig1.png
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "fig1.png"

T7_CSV = ROOT / "research/exp_t7/results.csv"
M1_JSON = ROOT / "output/benchmark_m1_smoke.json"
KRYLOV_TIME_V = {
    100_000: ROOT / "output/slepc_krylov_100k_time_v.txt",
    500_000: ROOT / "output/slepc_krylov_500k_time_v.txt",
}
N_75K = 74_984
WORKSTATION_GB = 30.0
C_LAYER_MD = ROOT / "output/c_layer_report.md"


def parse_brandts() -> dict[int, float]:
    """exp4 gpcca_fit  'gpcca_fit'  -2=n+6=tracemalloc  MB
    results.csv  exp
     n=8000  output/brandts_8k.jsonE3cellrank 2.1.0 """
    pts: dict[int, float] = {}
    with T7_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if "gpcca_fit" in row:
                i = row.index("gpcca_fit")
                n = int(float(row[i - 2]))
                pts[n] = float(row[i + 6]) / 1000.0  # MB → GB
    b8 = json.loads((ROOT / "output/brandts_8k.json").read_text())
    for r in b8["results"]:
        pts[int(r["n"])] = float(r["tracemalloc_mb"]) / 1000.0  # 2.1.0
    assert abs(pts[2000] - 0.2883) < 5e-4 and abs(pts[5000] - 1.8006) < 5e-4 \
        and abs(pts[8000] - 4.609) < 0.005, pts
    return pts


def parse_krylov() -> dict[int, float]:
    pts: dict[int, float] = {}
    for n, path in KRYLOV_TIME_V.items():
        m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)",
                      path.read_text(encoding="utf-8", errors="replace"))
        pts[n] = int(m.group(1)) / 1024 / 1024  # KiB → GiB
    return pts


def parse_sfate() -> dict[int, float]:
    d = json.loads(M1_JSON.read_text())
    return {int(r["n"]): float(r["rss_peak_mb_max"]) / 1000.0 / 1.073741824
            for r in d["results"]}  # MB(SI) → GiB


def parse_sfate_full_pipeline() -> float:
    """75k  RSS ≈1.89GB [time -v ]C11→ GiB"""
    m = re.search(r" RSS ≈([\d.]+)\s*GB",
                  C_LAYER_MD.read_text(encoding="utf-8"))
    assert m, "c_layer_report.md  75k "
    return float(m.group(1)) / 1.073741824  # GB → GiB


GB2GiB = 1 / 1.073741824


def main() -> None:
    brandts = {n: v * GB2GiB for n, v in parse_brandts().items()}  # GB→GiB
    krylov = parse_krylov()
    sfate = parse_sfate()
    print("brandts measured (GiB):", {k: round(v, 3) for k, v in brandts.items()})
    print("krylov measured (GiB):", {k: round(v, 3) for k, v in krylov.items()})
    print("sfate measured (GiB):", {k: round(v, 3) for k, v in sfate.items()})

    # sfate  benchmark_m1_smoke.md §
    ns = np.array(sorted(sfate), dtype=float)
    ys = np.array([sfate[int(n)] for n in ns])
    b, a = np.polyfit(ns / 1000.0, ys, 1)  # mem(GB) = a + b·(n/1000)
    r = np.corrcoef(ns / 1000.0, ys)[0, 1]
    print(f"sfate fit: mem(GB) = {a:.1f} + {b:.4f}·n/1000, R² = {r**2:.4f}, "
          f"pred(1M) = {a + b * 1000:.2f} GB")

    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.5e3, 1.6e6)
    ax.set_ylim(0.1, 2e5)
    ax.set_xlabel("number of cells n (log scale)", fontsize=11)
    ax.set_ylabel("peak memory (GiB, log scale)", fontsize=11)
    ax.set_title("Peak memory scaling: dense Schur (brandts) vs krylov-Schur vs sfate",
                 fontsize=12.5, weight="bold")

    n_grid = np.logspace(np.log10(1.8e3), np.log10(1.05e6), 200)

    C_CLIFF, C_HEAVY, C_SF = "#a40000", "#8f6202", "#2e7d32"

    # --- brandts +  n²  ---
    ax.plot(n_grid, 72 * n_grid**2 / 1e9 * GB2GiB, ls="--", lw=1.8, color=C_CLIFF,
            label="brandts dense Schur: empirical n² model\n(72 B/cell²; dashed = back-calculated)")
    bn = sorted(brandts)
    ax.plot(bn, [brandts[n] for n in bn], "s", ms=8, color=C_CLIFF,
            label="brandts measured (3 pts; 2.1.0/2.3.2 identical)")
    ax.plot([10_000], [72 * 10_000**2 / 1e9 * GB2GiB], "x", ms=11, mew=2.4, color=C_CLIFF)
    ax.annotate("n = 10k: OOM-killed\n(4 GB sandbox)", xy=(10_000, 7.2 * GB2GiB),
                xytext=(1.3e4, 1.5), fontsize=8.5, color=C_CLIFF,
                arrowprops=dict(arrowstyle="->", color=C_CLIFF, lw=1.1))
    # 405GB@75k =377 GiB
    gb_405 = 72 * N_75K**2 / 1e9 * GB2GiB
    ax.plot([N_75K], [gb_405], "o", ms=7, mfc="none", mew=2.0, color=C_CLIFF)
    ax.annotate(f"{gb_405:.0f} GiB @ 75k (=405 GB)\n(back-calculated)",
                xy=(N_75K, gb_405), xytext=(3.1e4, 2.2e3),
                fontsize=9, color=C_CLIFF, weight="bold",
                arrowprops=dict(arrowstyle="->", color=C_CLIFF, lw=1.2))

    # --- krylov ---
    kn = sorted(krylov)
    ax.plot(kn, [krylov[n] for n in kn], "^", ms=9, color=C_HEAVY,
            label="krylov-Schur measured (PETSc/SLEPc)")

    # --- sfate+ 75k  +  ---
    ax.plot(ns, ys, "o", ms=8, color=C_SF,
            label="sfate measured (4 tiers, graph construction)")
    gb_full = parse_sfate_full_pipeline()
    ax.plot([N_75K], [gb_full], "o", ms=9, mfc="none", mew=2.0, color=C_SF,
            label=f"sfate full pipeline incl. solve ({gb_full:.2f} GiB @ 75k)")
    n_fit = np.logspace(np.log10(ns.min()), np.log10(1.05e6), 100)
    ax.plot(n_fit, a + b * n_fit / 1000.0, ls="--", lw=1.8, color=C_SF,
            label=f"sfate linear fit (R² = {r**2:.4f})\n(dashed beyond 500k = extrapolated)")
    ax.annotate(f"{a + b * 1000:.2f} GiB @ 1M\n[extrapolated]",
                xy=(1e6, a + b * 1000), xytext=(2.1e5, 0.35),
                fontsize=8.5, color=C_SF,
                arrowprops=dict(arrowstyle="->", color=C_SF, lw=1.1))

    # --- 30 GiB  ---
    ax.axhline(WORKSTATION_GB, color="#555555", lw=1.4, ls=":")
    ax.text(2.0e5, WORKSTATION_GB * 1.22,
            "workstation physical RAM (~30 GiB)", fontsize=9, color="#555555")

    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(loc="upper left", fontsize=8.8, framealpha=0.95)

    fig.text(0.01, 0.005,
             "Sources: research/exp_t7/results.csv + output/brandts_8k.json (brandts) · "
             "output/slepc_krylov_{100k,500k}_time_v.txt (krylov) · "
             "output/benchmark_m1_smoke.json (sfate) · output/c_layer_report.md (75k full)",
             fontsize=6.8, color="#888888")
    fig.text(0.01, 0.028,
             "Solid points: graph construction; open circle: full pipeline incl. solve. "
             "sfate 500k full-pipeline solve pending (Limitations).",
             fontsize=7.6, color="#666666")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(OUT, dpi=200, facecolor="white")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
