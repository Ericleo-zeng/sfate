#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fig3.py — Fig3: Sensitivity to automatic terminal-state curation.

数据源（均从文件正则现读 + assert 验证，禁止硬编码）：
- (a) 75k 注释态 × macrostate 列联表：output/c_layer_report.md §S5
  （"注释态 × macrostate 列联" markdown 表；口径：每宏观态仅 30 代表细胞，
  其余 74,804 未指派——CR n_cells=30 口径）
- (b) 归因消融 min Spearman 及逐类/逐对数值：output/b_layer_report.md
  §归因消融（对比 2 核效应 / 对比 3 终末态集效应，2×2 统一 sfate 求解器）

assert 清单（任一失败：打印 warning + 对应面板标 [pending]，不崩溃）：
- A1: DAM_like 行 == [8, 8, 2, 4, 6, 0]
- A2: 列联表每列之和 == 30（30 代表/宏观态口径）
- A3: 对比 2 min ∈ [0.829, 0.830] 且逐类 6 个值与正文一致（重算 min 吻合）
- A4: 对比 3 min ∈ [0.135, 0.136] 且逐对 6 个值重算 min 吻合

输出：paper/figures/fig3.png（300 dpi）+ fig3.pdf；渲染后打印全部 assert 状态。
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
C_LAYER_MD = ROOT / "output/c_layer_report.md"
B_LAYER_MD = ROOT / "output/b_layer_report.md"
OUT_PNG = Path(__file__).resolve().parent / "fig3.png"
OUT_PDF = Path(__file__).resolve().parent / "fig3.pdf"

C_CLIFF = "#a40000"
C_HEAVY = "#8f6202"
C_SF = "#2e7d32"

ASSERT_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    ASSERT_RESULTS.append((name, ok, detail))
    tag = "PASS" if ok else "FAIL"
    print(f"[assert {tag}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


# ---------------------------------------------------------------------------
# 解析 (a)：c_layer_report.md §S5 列联表
# ---------------------------------------------------------------------------

def parse_contingency() -> tuple[list[str], list[str], np.ndarray]:
    text = C_LAYER_MD.read_text(encoding="utf-8")
    lines = text.splitlines()
    header_i = next(i for i, ln in enumerate(lines)
                    if ln.strip().startswith("| 注释态"))
    cols = [c.strip() for c in lines[header_i].strip().strip("|").split("|")][1:]
    rows: list[str] = []
    mat: list[list[int]] = []
    for ln in lines[header_i + 2:]:  # 跳过分隔行
        if not ln.strip().startswith("|"):
            break
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        rows.append(cells[0])
        mat.append([int(x) for x in cells[1:]])
    return rows, cols, np.array(mat, dtype=int)


# ---------------------------------------------------------------------------
# 解析 (b)：b_layer_report.md 归因消融 对比 2 / 对比 3
# ---------------------------------------------------------------------------

def parse_ablation(text: str, tag: str) -> tuple[float, list[tuple[str, float]]]:
    """返回 (正文 min, [(名称, 值), ...])。tag 为例 '对比 2'。"""
    ln = next(l for l in text.splitlines() if f"**{tag}" in l and "Spearman" in l)
    m_min = re.search(r"最小值\s*=\s*\*\*([\d.]+)\*\*", ln)
    stated_min = float(m_min.group(1))
    paren = re.search(r"（(.+?)）", ln.split("最小值")[1]).group(1)
    pairs = [(name, float(v)) for name, v in
             re.findall(r"([A-Za-z0-9_→]+?)\s+([\d.]+)(?=，|）|$)", paren)]
    return stated_min, pairs


# ---------------------------------------------------------------------------
# 面板渲染
# ---------------------------------------------------------------------------

def panel_a(ax, rows, cols, mat) -> None:
    im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=30, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=28, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(rows)), rows, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8.5,
                    color="white" if v >= 18 else "#333333")
    # DAM_like 行高亮（散布 5 宏观态）
    dam_i = rows.index("DAM_like")
    ax.add_patch(plt.Rectangle((-0.5, dam_i - 0.5), mat.shape[1], 1,
                               fill=False, edgecolor=C_CLIFF, lw=2.4))
    ax.get_yticklabels()[dam_i].set_color(C_CLIFF)
    ax.get_yticklabels()[dam_i].set_weight("bold")
    ax.set_title("(a) 75k annotated-state × macrostate contingency\n"
                 "DAM_like scattered across 5 macrostates; "
                 "PU1low_lymphoid nearly absent",
                 fontsize=10, weight="bold")
    ax.set_xlabel("GPCCA macrostate (30 representatives each)", fontsize=9.5)
    ax.set_ylabel("annotated state", fontsize=9.5)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("representative cells", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    ax.text(0.0, -0.34,
            "n_cells=30 per macrostate (180 total); "
            "74,804 of 74,984 cells unassigned",
            transform=ax.transAxes, fontsize=8, color="#666666")


def panel_b(ax, ab2, ab3) -> None:
    min2, pts2 = ab2
    min3, pts3 = ab3
    labels = ["kernel effect\n(fixed terminal cells)", "terminal-state-set effect\n(fixed kernel)"]
    mins = [min2, min3]
    colors = [C_HEAVY, C_CLIFF]
    bars = ax.bar([0, 1], mins, width=0.45, color=colors, alpha=0.75,
                  edgecolor=colors, lw=1.4)
    # 条高与现读值一致性校验（渲染日志可见）
    heights = [b.get_height() for b in bars]
    ok_h = all(abs(h - m) < 1e-12 for h, m in zip(heights, mins))
    check("A5: 消融主条高度 == 现读 min 值",
          ok_h, f"bar heights={heights}, mins={mins}")
    if not ok_h:
        raise ValueError(f"条高与读值不一致: {heights} vs {mins}")
    rng = np.random.default_rng(20260909)
    for x, pts, c in [(0, pts2, C_HEAVY), (1, pts3, C_CLIFF)]:
        vals = [v for _, v in pts]
        jit = rng.uniform(-0.13, 0.13, len(vals))
        ax.scatter(np.full(len(vals), x) + jit, vals, s=42, color=c,
                   edgecolor="white", lw=0.7, zorder=3)
    for x, m in zip([0, 1], mins):
        if m > 0.35:  # 高条：竖排白字内置
            ax.text(x, m / 2, f"min = {m:.3f}", ha="center", va="center",
                    fontsize=10.5, weight="bold", color="white", rotation=90)
        else:  # 矮条：右侧横排
            ax.text(x + 0.26, m, f"min = {m:.3f}", ha="left", va="center",
                    fontsize=10.5, weight="bold", color=colors[x])
    ax.axhline(0.95, color=C_SF, ls="--", lw=1.4)
    ax.text(1.30, 0.917, "E4 gate (0.95)", fontsize=8, color=C_SF,
            va="top", ha="left")
    ax.set_xticks([0, 1], labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel("per-class Spearman (min over classes)", fontsize=9.5)
    ax.set_title("(b) 5k 2×2 ablation (unified sfate solver):\n"
                 "terminal-state-set contrast larger than kernel contrast",
                 fontsize=10, weight="bold")
    ax.grid(True, axis="y", alpha=0.25, lw=0.5)


def panel_pending(ax, label: str, err: Exception) -> None:
    ax.axis("off")
    ax.text(0.5, 0.55, f"{label} [pending]", ha="center", fontsize=13,
            color=C_CLIFF, weight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.42, f"data parse/assert failed:\n{err}", ha="center",
            fontsize=8, color="#888888", transform=ax.transAxes)


# ---------------------------------------------------------------------------

def main() -> int:
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.4, 6.2),
                                   gridspec_kw={"width_ratios": [1.15, 1.0]})

    # ---- (a) 列联热图 ----
    try:
        rows, cols, mat = parse_contingency()
        ok1 = check("A1: DAM_like 行 == [8,8,2,4,6,0]",
                    mat[rows.index("DAM_like")].tolist() == [8, 8, 2, 4, 6, 0],
                    str(mat[rows.index("DAM_like")].tolist()))
        ok2 = check("A2: 列联表每列之和 == 30（30 代表/宏观态口径）",
                    bool((mat.sum(axis=0) == 30).all()),
                    str(mat.sum(axis=0).tolist()))
        if not (ok1 and ok2):
            raise ValueError("列联表 assert 未过")
        panel_a(axa, rows, cols, mat)
    except Exception as e:  # noqa: BLE001
        print(f"[warning] panel (a) 数据缺失/assert 失败: {e}", file=sys.stderr)
        panel_pending(axa, "(a) 75k contingency", e)

    # ---- (b) 消融条形图 ----
    try:
        text = B_LAYER_MD.read_text(encoding="utf-8")
        min2, pts2 = parse_ablation(text, "对比 2")
        min3, pts3 = parse_ablation(text, "对比 3")
        ok3 = check("A3: 对比 2 min ∈ [0.829,0.830] 且逐类重算吻合",
                    0.829 <= min2 <= 0.830 and len(pts2) == 6
                    and abs(min(v for _, v in pts2) - min2) < 5e-4,
                    f"min={min2}, per-class={[v for _, v in pts2]}")
        ok4 = check("A4: 对比 3 min ∈ [0.135,0.136] 且逐对重算吻合",
                    0.135 <= min3 <= 0.136 and len(pts3) == 6
                    and abs(min(v for _, v in pts3) - min3) < 5e-4,
                    f"min={min3}, per-pair={[v for _, v in pts3]}")
        if not (ok3 and ok4):
            raise ValueError("消融 assert 未过")
        panel_b(axb, (min2, pts2), (min3, pts3))
    except Exception as e:  # noqa: BLE001
        print(f"[warning] panel (b) 数据缺失/assert 失败: {e}", file=sys.stderr)
        panel_pending(axb, "(b) 5k ablation", e)

    # ---- 双行 footer（与 fig1 同款）----
    fig.text(0.01, 0.028,
             "Contingency: annotated states × GPCCA macrostates (30 representatives each); "
             "Spearman = per-class minimum; minima are highlighted for orientation, not effect size.",
             fontsize=7.6, color="#666666")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT_PNG, dpi=300, facecolor="white")
    fig.savefig(OUT_PDF, facecolor="white")
    print(f"written: {OUT_PNG}\nwritten: {OUT_PDF}")
    n_fail = sum(1 for _, ok, _ in ASSERT_RESULTS if not ok)
    print(f"assert 汇总: {len(ASSERT_RESULTS) - n_fail}/{len(ASSERT_RESULTS)} PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
