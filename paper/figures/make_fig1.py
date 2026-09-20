#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fig1.py — Fig1: 内存定律三方对比（log-log）：brandts 稠密 Schur vs
krylov-Schur vs sfate。

数字来源（脚本现场读取，禁止凭记忆）：
- brandts 实测两点（tracemalloc 峰值 288.2MB@2000 / 1800.3MB@5000，
  72·n² 定律拟合基础；n=10000 被 OOM killer 终止（4GB 沙箱））：
  research/exp_t7/results.csv（exp4 行，列位见 parse_brandts()；
  口径释义见 research/T7_experiments.md §exp4 表）
- krylov 实测两点（/usr/bin/time -v Maximum RSS，KiB→GB）：
  output/slepc_krylov_100k_time_v.txt（1,607,860 KiB）
  output/slepc_krylov_500k_time_v.txt（3,849,808 KiB）
- sfate 实测四档（RSS 采样峰值 MB，**建图阶段口径**）+ 线性拟合（脚本内按四点重拟合）：
  output/benchmark_m1_smoke.json（fixture_5k_real / synthetic_10k /
  real_75k_full / synthetic_500k）
- sfate 75k 全管线（含求解，time -v 近似口径 ≈1.89GB，空心圈）：
  output/c_layer_report.md §环境指纹与口径（C11）
- 405GB@75k 回算点：72 × 74,984² B（定律来源 docs/step12_postmortem.md §3）
- 30GB 物理内存线：docs/step12_postmortem.md §3（本机 ~30GB）

输出：paper/figures/fig1.png（与脚本同名可追溯）。
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
    """exp4 gpcca_fit 行：定位 'gpcca_fit' 字段，相对列位 -2=n，+6=tracemalloc 峰值 MB
    （results.csv 各 exp 行前列数不齐，必须相对定位）。
    第三点 n=8000 现读 output/brandts_8k.json（E3，cellrank 2.1.0 复跑同机实测）。"""
    pts: dict[int, float] = {}
    with T7_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if "gpcca_fit" in row:
                i = row.index("gpcca_fit")
                n = int(float(row[i - 2]))
                pts[n] = float(row[i + 6]) / 1000.0  # MB → GB
    b8 = json.loads((ROOT / "output/brandts_8k.json").read_text())
    for r in b8["results"]:
        pts[int(r["n"])] = float(r["tracemalloc_mb"]) / 1000.0  # 2.1.0 值覆盖
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
    """75k 全链（含求解）进程峰值 RSS ≈1.89GB [time -v 近似口径]（C11）→ GiB。"""
    m = re.search(r"全链（含求解）进程峰值 RSS ≈([\d.]+)\s*GB",
                  C_LAYER_MD.read_text(encoding="utf-8"))
    assert m, "c_layer_report.md 中未找到 75k 全链峰值口径行"
    return float(m.group(1)) / 1.073741824  # GB → GiB


GB2GiB = 1 / 1.073741824


def main() -> None:
    brandts = {n: v * GB2GiB for n, v in parse_brandts().items()}  # GB→GiB
    krylov = parse_krylov()
    sfate = parse_sfate()
    print("brandts measured (GiB):", {k: round(v, 3) for k, v in brandts.items()})
    print("krylov measured (GiB):", {k: round(v, 3) for k, v in krylov.items()})
    print("sfate measured (GiB):", {k: round(v, 3) for k, v in sfate.items()})

    # sfate 线性拟合（四点重拟合，与 benchmark_m1_smoke.md §线性外推 同口径）
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

    # --- brandts：实测三点 + 经验 n² 模型虚线 ---
    ax.plot(n_grid, 72 * n_grid**2 / 1e9 * GB2GiB, ls="--", lw=1.8, color=C_CLIFF,
            label="brandts dense Schur: empirical n² model\n(72 B/cell²; dashed = back-calculated)")
    bn = sorted(brandts)
    ax.plot(bn, [brandts[n] for n in bn], "s", ms=8, color=C_CLIFF,
            label="brandts measured (3 pts; 2.1.0/2.3.2 identical)")
    ax.plot([10_000], [72 * 10_000**2 / 1e9 * GB2GiB], "x", ms=11, mew=2.4, color=C_CLIFF)
    ax.annotate("n = 10k: OOM-killed\n(4 GB sandbox)", xy=(10_000, 7.2 * GB2GiB),
                xytext=(1.3e4, 1.5), fontsize=8.5, color=C_CLIFF,
                arrowprops=dict(arrowstyle="->", color=C_CLIFF, lw=1.1))
    # 405GB@75k 回算点（=377 GiB）
    gb_405 = 72 * N_75K**2 / 1e9 * GB2GiB
    ax.plot([N_75K], [gb_405], "o", ms=7, mfc="none", mew=2.0, color=C_CLIFF)
    ax.annotate(f"{gb_405:.0f} GiB @ 75k (=405 GB)\n(back-calculated)",
                xy=(N_75K, gb_405), xytext=(3.1e4, 2.2e3),
                fontsize=9, color=C_CLIFF, weight="bold",
                arrowprops=dict(arrowstyle="->", color=C_CLIFF, lw=1.2))

    # --- krylov：实测两点 ---
    kn = sorted(krylov)
    ax.plot(kn, [krylov[n] for n in kn], "^", ms=9, color=C_HEAVY,
            label="krylov-Schur measured (PETSc/SLEPc)")

    # --- sfate：实测四点（建图口径）+ 75k 全管线空心圈 + 线性拟合虚线 ---
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
    ax.text(ns[-1], ys[-1] * 1.35,
            "graph only; absorption failed (Results 2)",
            ha="center", va="bottom", fontsize=8, color=C_SF)

    # --- 30 GiB 物理线 ---
    ax.axhline(WORKSTATION_GB, color="#555555", lw=1.4, ls=":")
    ax.text(2.0e5, WORKSTATION_GB * 1.22,
            "workstation physical RAM (~30 GiB)", fontsize=9, color="#555555")

    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(loc="upper left", fontsize=8.8, framealpha=0.95)

    fig.text(0.01, 0.028,
             "Solid points: graph construction; open circle: full pipeline incl. solve. "
             "The sfate 500k absorption solve was attempted and did not converge within the iteration cap (Results 2).",
             fontsize=7.6, color="#666666")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    OUT_PDF = OUT.with_suffix(".pdf")
    fig.savefig(OUT, dpi=200, facecolor="white")
    fig.savefig(OUT_PDF, facecolor="white")
    print(f"written: {OUT}\nwritten: {OUT_PDF}")


if __name__ == "__main__":
    main()
