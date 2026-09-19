#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fig4.py — Fig4: P0 5k  / 75k  / 75k + ILU inset

 + assert = output/benchmark_m2.md §P0
- output/benchmark_m2.md §P05k 26–30 / 0.73s75k  5/6  5
   3915s75k  ~59  12029.0s
- output/c_layer_report.md §S1/S2 [49951,...,137]5/6 ≥ 4.99
   3915.2s m2
- <excluded>ILU inset fill=1 132s / fill=10 380s /
   0.1s "52 /0.1s"  smoke
- 135× = 3915/29.0

assert
- P1: 5k  == (26, 30)  ∈ [0.70, 0.75]
- P2: 75k  ~59 ∈ [58,60]  == 120
- P3: 29.0s ∈ [28.5, 29.5]  3915/29.0 ∈ [134, 136]
- P4:  5/6  ≥ 499005  m2/c_layer ±1s
- P5: ILU  132/380/0.1±2%

paper/figures/fig4.png300 dpi+ fig4.pdf warning+[pending]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
M2_MD = ROOT / "output/benchmark_m2.md"
C_MD = ROOT / "output/c_layer_report.md"
INC_MD = ROOT / "<excluded>"
OUT_PNG = Path(__file__).resolve().parent / "fig4.png"
OUT_PDF = Path(__file__).resolve().parent / "fig4.pdf"

C_CLIFF = "#a40000"
C_HEAVY = "#8f6202"
C_SF = "#2e7d32"
C_BLUE = "#3465a4"

ASSERT_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    ASSERT_RESULTS.append((name, ok, detail))
    print(f"[assert {'PASS' if ok else 'FAIL'}] {name}"
          + (f"  ({detail})" if detail else ""))
    return ok


def main() -> int:
    try:
        m2 = M2_MD.read_text(encoding="utf-8")
        c = C_MD.read_text(encoding="utf-8")
        inc = INC_MD.read_text(encoding="utf-8")

        # ---- m2 §P0  ----
        it5k_lo, it5k_hi, t5k = re.search(
            r"(\d+)–(\d+) /([\d.]+)s", m2).groups()
        it5k_lo, it5k_hi, t5k = int(it5k_lo), int(it5k_hi), float(t5k)
        agg_frac = re.search(r"(\d)/(\d)  (\d+) ", m2).groups()
        agg_bad, agg_total, agg_cap = int(agg_frac[0]), int(agg_frac[1]), int(agg_frac[2])
        t_agg = float(re.search(r" (\d+)sP0 ", m2).group(1))
        it_col, it_col_peak, t_col = re.search(
            r"~(\d+) / (\d+)\*\*([\d.]+)s\*\*", m2).groups()
        it_col, it_col_peak, t_col = int(it_col), int(it_col_peak), float(t_col)
        speedup = t_agg / t_col

        # ---- c_layer  ----
        agg_iters = [int(v) for v in re.search(
            r" \[([\d, ]+)\]", c).group(1).split(",")]
        t_agg_c = float(re.search(
            r"absorption_probabilities 6  \| ([\d.]+)", c).group(1))

        # ---- ILU inset<excluded>----
        ilu1 = float(re.search(r"fill=1  (\d+)s", inc).group(1))
        ilu10 = float(re.search(r"fill=10  (\d+)s", inc).group(1))
        ilu0 = float(re.search(r" GMRES  \d+  ([\d.]+)s",
                               inc).group(1))

        ok1 = check("P1: 5k  == (26,30)  ∈ [0.70,0.75]",
                    (it5k_lo, it5k_hi) == (26, 30) and 0.70 <= t5k <= 0.75,
                    f"({it5k_lo},{it5k_hi}), {t5k}s")
        ok2 = check("P2: 75k  ~59 ∈ [58,60]  == 120",
                    58 <= it_col <= 60 and it_col_peak == 120,
                    f"{it_col}/{it_col_peak}")
        ok3 = check("P3: 29.0s ∈ [28.5,29.5]  135× ∈ [134,136]",
                    28.5 <= t_col <= 29.5 and 134 <= speedup <= 136,
                    f"{t_col}s, {speedup:.1f}×")
        ok4 = check("P4:  5/6  ≥49900  m2/c_layer ",
                    agg_bad == 5 and agg_total == 6
                    and sum(1 for v in agg_iters if v >= 49900) == 5
                    and abs(t_agg - t_agg_c) <= 1.0,
                    f"iters={agg_iters}, {t_agg} vs {t_agg_c}")
        ok5 = check("P5: ILU 132/380/0.1±2%",
                    abs(ilu1 - 132) / 132 <= 0.02
                    and abs(ilu10 - 380) / 380 <= 0.02
                    and abs(ilu0 - 0.1) / 0.1 <= 0.02,
                    f"{ilu1}/{ilu10}/{ilu0}")
        if not (ok1 and ok2 and ok3 and ok4 and ok5):
            raise ValueError("assert ")

        # =================  2/3 + inset =================
        fig = plt.figure(figsize=(11.4, 5.2))
        ax = fig.add_axes([0.075, 0.115, 0.585, 0.775])
        configs = ["5k column-wise", "75k aggregated RHS", "75k column-wise"]
        xs = np.arange(3)
        w = 0.36
        it_mean = [np.mean([it5k_lo, it5k_hi]), agg_cap * 1e4, it_col]
        it_err_lo = [it_mean[0] - it5k_lo, 0, 0]
        it_err_hi = [it5k_hi - it_mean[0], 0, it_col_peak - it_col]
        times = [t5k, t_agg, t_col]

        ax.set_yscale("log")
        ax.set_ylim(1, 3e6)
        ax.set_ylabel("GMRES iterations per solve (log)", fontsize=10)
        ax.set_xticks(xs, configs, fontsize=10)
        b1 = ax.bar(xs - w / 2, it_mean, w, color=[C_SF, C_CLIFF, C_SF],
                    alpha=0.85, edgecolor="white",
                    yerr=[it_err_lo, it_err_hi], capsize=5,
                    error_kw=dict(lw=1.4, ecolor="#333333"))
        b1[1].set_hatch("//")
        b1[1].set_alpha(0.55)
        ax.text(xs[0] - w / 2, it_mean[0] * 3.2, f"{it5k_lo}–{it5k_hi}",
                ha="center", fontsize=9, color=C_SF, weight="bold")
        ax.text(xs[1] - w / 2, agg_cap * 1e4 * 1.5,
                f"≥{agg_cap}×10⁴ (cap)\ndid not converge\n({agg_bad}/{agg_total} classes)",
                ha="center", va="bottom", fontsize=8.5, color=C_CLIFF,
                weight="bold", linespacing=1.35)
        ax.text(xs[2] - w / 2, it_col * 1.6, f"~{it_col} (peak {it_col_peak})",
                ha="center", fontsize=9, color=C_SF, weight="bold")

        ax2 = ax.twinx()
        ax2.set_yscale("log")
        ax2.set_ylim(0.1, 3e5)
        ax2.set_ylabel("wall time (s, log)", fontsize=10)
        b2 = ax2.bar(xs + w / 2, times, w,
                     color=[C_BLUE, C_BLUE, C_BLUE], alpha=0.45,
                     edgecolor=C_BLUE)
        for x, t in zip(xs, times):
            ax2.text(x + w / 2, t * 1.5, f"{t:g} s", ha="center",
                     fontsize=9, color=C_BLUE, weight="bold")
        # 135×
        ax2.annotate("", xy=(xs[2] + w / 2, t_col * 3), xytext=(xs[1] + w / 2, t_agg * 0.5),
                     arrowprops=dict(arrowstyle="->", color=C_HEAVY, lw=1.6,
                                     connectionstyle="arc3,rad=-0.25"))
        ax2.text((xs[1] + xs[2]) / 2 + w / 2, t_agg * 0.16,
                 f"{speedup:.0f}×\n({t_agg:g} s → {t_col:g} s)",
                 ha="center", fontsize=9.5, color=C_HEAVY, weight="bold",
                 linespacing=1.35)

        ax.set_title("(a) GMRES behavior across three configurations on the 75k absorption system",
                     fontsize=11.5, weight="bold")
        #
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor=C_SF, alpha=0.85, label="iterations (left axis)"),
            Patch(facecolor=C_BLUE, alpha=0.45, edgecolor=C_BLUE,
                  label="wall time (right axis)"),
        ], loc="upper left", fontsize=8.5, framealpha=0.95)
        ax.text(0.985, 0.955, "500k convergence: pending (Limitations)",
                transform=ax.transAxes, fontsize=8.5, color=C_CLIFF,
                ha="right", va="top",
                bbox=dict(facecolor="#fdf3f3", edgecolor=C_CLIFF,
                          boxstyle="round,pad=0.3", lw=0.8))

        # ---- (b) ILU inset----
        axin = fig.add_axes([0.715, 0.30, 0.255, 0.42])
        ilu_vals = [ilu1, ilu10, ilu0]
        axin.set_yscale("log")
        axin.set_ylim(0.05, 1e3)
        axin.bar([0, 1, 2], ilu_vals, color=[C_CLIFF, C_CLIFF, C_SF],
                 alpha=0.75, width=0.55)
        for x, v in zip([0, 1, 2], ilu_vals):
            axin.text(x, v * 1.7, f"{v:g} s", ha="center", fontsize=8,
                      weight="bold",
                      color=C_CLIFF if x < 2 else C_SF)
        axin.set_xticks([0, 1, 2], ["spilu\nfill=1", "spilu\nfill=10", "no\nprecond."],
                        fontsize=7.5)
        axin.set_title("(b) ILU construction cost vs one\nunpreconditioned column solve",
                       fontsize=8.5, color=C_CLIFF)
        axin.tick_params(labelsize=7)
        axin.grid(True, axis="y", alpha=0.25, lw=0.5)

        # ----  footer ----
        fig.text(0.01, 0.028,
                 "135× is a configuration-level difference (aggregated+ILU vs column-wise+no-ILU); "
                 "both RHS organization and preconditioning changed, so no single-factor causal "
                 "attribution is implied. Iterations = GMRES per solve.",
                 fontsize=7.6, color="#666666")
        fig.text(0.01, 0.005,
                 "Sources: output/benchmark_m2.md (P0) · "
                 "output/c_layer_report.md (S1/S2) · "
                 "<excluded> (ILU inset only)",
                 fontsize=6.8, color="#888888")
        fig.savefig(OUT_PNG, dpi=300, facecolor="white")
        fig.savefig(OUT_PDF, facecolor="white")
        print(f"written: {OUT_PNG}\nwritten: {OUT_PDF}")
    except Exception as e:  # noqa: BLE001
        print(f"[warning] fig2 /assert : {e}", file=sys.stderr)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.55, "Fig4 [pending]", ha="center", fontsize=14,
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
