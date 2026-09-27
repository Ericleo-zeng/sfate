#!/usr/bin/env python
"""Figure 7 (three-panel) — Layer B2 audit: sfate vs CellRank fate concordance.

Panel A  S_fate heatmap: Spearman between CR macro-state fate columns (rows,
         natural macro order, Hungarian-matched) and sfate arm fate columns
         (cols, M3 order); rows reordered by matched arm -> diagonal = same-arm.
Panel B  Per-arm full-cell fate Spearman (bars) with across-assignment-rule
         bands (B2_AUDIT_SESSION_20260927 Results-6 table; DAM zero-variance).
Panel C  Self-absorption block: mean absorption into target arm within cells
         of each annotated state; recomputed and verified against
         output/audit_b2_testB_log.md (transcribed 4dp reference below).

Authoritative input: output/audit_b2_step3b_final_v3.npz
  (all arrays follow the `states` M3 convention; produced and self-verified by
   audit_b2_fix_v2_metadata.py — spearman / S_fate repaired there).

Output: output/figure7_b2_audit.png (300 dpi) and .pdf
"""

from pathlib import Path

import anndata as ad
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

CWD = Path(__file__).resolve().parent.parent.parent.parent  # Cellrank重构/
NPZ = CWD / "output" / "audit_b2_step3b_final_v3.npz"
FATE75K = CWD / "output" / "fate_full_75k.h5ad"
ANNOT = "/mnt/t9/datasets/ad_microglia_fate_landscape_output/06_annotation/adata_annotated.h5ad"
OUT_PNG = CWD / "output" / "figure7_b2_audit.png"
OUT_PDF = CWD / "output" / "figure7_b2_audit.pdf"

M3 = ["C1q_inflammatory", "DAM_like", "Homeostatic",
      "IFN_responsive", "Proliferating", "PU1low_lymphoid"]
SHORT = ["C1q", "DAM", "Homeo", "IFN", "Prolif", "PU1low"]
IDX = {a: j for j, a in enumerate(M3)}
DISCORDANT = "PU1low_lymphoid"  # fate-correlation discordant arm (Results 6)

# Across-assignment-rule bands, transcribed from B2_AUDIT_SESSION_20260927
# section 3 ("每臂 fate Spearman 带"); DAM = zero-variance anchor.
BAND = {"C1q_inflammatory": (0.66, 0.68), "DAM_like": (0.648, 0.648),
        "Homeostatic": (0.65, 0.67), "IFN_responsive": (0.55, 0.64),
        "Proliferating": (0.49, 0.62), "PU1low_lymphoid": (-0.73, -0.51)}

# Transcribed from output/audit_b2_testB_log.md self_absorption dict + the
# into-Prolif column. NOTE (2026-09-27 session): the log's PRINTED 6x6 block
# matrix is row-scrambled (print-time bug, likely hand-assembled at 13:12);
# its dict values and global means are correct and verified against _v3.
EXPECTED_SELF = np.array([0.1935, 0.1770, 0.2252, 0.1948, 0.1469, 0.1875])
EXPECTED_PROLIF_COL = np.array([0.0578, 0.0570, 0.0579, 0.0580, 0.1469, 0.0563])


def _to_str(a: np.ndarray) -> np.ndarray:
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def read_obs_column(f: h5py.File, key: str) -> np.ndarray:
    """Read an obs column from an on-disk h5ad as unicode strings.

    Handles both layouts: plain string dataset, or anndata categorical group
    (children 'codes'/'categories', or attrs pointing at sibling datasets).
    """
    g = f["obs"]
    item = g[key]
    if isinstance(item, h5py.Dataset):
        return _to_str(np.asarray(item))
    if "codes" in item.keys() and "categories" in item.keys():
        return _to_str(np.asarray(item["categories"])[np.asarray(item["codes"])])
    cats = np.asarray(g[item.attrs["categories"]])
    return _to_str(cats[np.asarray(item["codes"])])


def load_annotations_in_prod_order() -> np.ndarray:
    """microglia_state for the 74,984 cells, in fate-h5ad (npz) row order."""
    with h5py.File(ANNOT, "r") as f:
        gobs = f["obs"]
        idx_ann = np.asarray(gobs[gobs.attrs["_index"]][...]).astype(str)
        ann = read_obs_column(f, "microglia_state")
    prod_index = np.asarray(ad.read_h5ad(FATE75K).obs_names).astype(str)
    lut = {b: i for i, b in enumerate(idx_ann)}
    sel = np.array([lut[b] for b in prod_index])
    assert ann.shape[0] == idx_ann.shape[0], \
        f"annotation column length {ann.shape[0]} != index length {idx_ann.shape[0]}"
    return ann[sel]


def panel_a(S: np.ndarray, spearman: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Hungarian-match S_fate rows (macros) to arms; reorder rows to arm order."""
    rows, cols = linear_sum_assignment(-S)
    assert sorted(cols.tolist()) == list(range(6)), "matching not a bijection"
    row_to_arm = dict(zip(rows.tolist(), cols.tolist()))
    for r, j in row_to_arm.items():
        assert abs(S[r, j] - spearman[j]) < 1e-9, (r, j, S[r, j], spearman[j])
    print("[ok] panel A: Hungarian diagonal == per-arm spearman")
    arm_order = [row_to_arm_inv for row_to_arm_inv in
                 sorted(row_to_arm, key=lambda r: row_to_arm[r])]
    return S[arm_order, :], arm_order


def panel_c(F_sf: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Self-absorption block; assert against verified testB log values.

    The log's printed 6x6 matrix is row-scrambled, so we assert the two
    quantities that are independently verified: the self-absorption
    diagonal (log dict) and the into-Prolif column (all->Prolif ~ 0.057).
    """
    block = np.zeros((6, 6))
    for i, a in enumerate(M3):
        m = labels == a
        assert m.sum() > 0, a
        block[i] = F_sf[m].mean(axis=0)
    self_abs = np.diag(block)
    err = np.abs(self_abs - EXPECTED_SELF).max()
    assert err < 1e-4, f"self-absorption mismatch vs testB log dict: {err}"
    err2 = np.abs(block[:, IDX["Proliferating"]] - EXPECTED_PROLIF_COL).max()
    assert err2 < 1e-4, f"into-Prolif column mismatch: {err2}"
    print(f"[ok] panel C: self-absorption == testB log dict (max |diff| = {err:.2e}); "
          f"into-Prolif col ok ({err2:.2e}); "
          f"self-absorption {dict(zip(SHORT, np.round(self_abs, 4)))}")
    return block


def main() -> None:
    plt.rcParams.update({          # 全局出版字号，三栏+色标统一
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })
    z = np.load(NPZ, allow_pickle=True)
    states = [str(s) for s in z["states"]]
    assert states == M3, states
    F_sf = np.asarray(z["F_sfate"], dtype=np.float64)
    S = np.asarray(z["S_fate"], dtype=np.float64)
    sp = np.asarray(z["spearman"], dtype=np.float64)

    S_ord, arm_rows = panel_a(S, sp)
    labels = load_annotations_in_prod_order()
    block = panel_c(F_sf, labels)

    fig = plt.figure(figsize=(11.5, 3.9), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.15])

    # ---- Panel A: S_fate heatmap -------------------------------------------
    axA = fig.add_subplot(gs[0, 0])
    im = axA.imshow(S_ord, cmap="RdBu_r", vmin=-0.75, vmax=0.75)
    axA.set_xticks(range(6), SHORT)
    plt.setp(axA.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    axA.set_yticks(range(6),
                   [f"CR macro {r}\n({SHORT[i]})" for i, r in enumerate(arm_rows)])
    for i in range(6):
        for j in range(6):
            axA.text(j, i, f"{S_ord[i, j]:+.2f}", ha="center", va="center",
                     fontsize=7, color="black" if abs(S_ord[i, j]) < 0.55 else "white")
    fig.colorbar(im, ax=axA, shrink=0.85, label="Spearman")
    axA.set_title("A  Cross-tool Spearman\n(CR macro × sfate arm)", fontsize=10)

    # ---- Panel B: per-arm concordance --------------------------------------
    axB = fig.add_subplot(gs[0, 1])
    x = np.arange(6)
    colors = ["#b2182b" if a == DISCORDANT else "#2166ac" for a in M3]
    axB.bar(x, sp, color=colors, width=0.62)
    for j, a in enumerate(M3):
        lo, hi = BAND[a]
        if hi - lo > 1e-9:
            axB.plot([j, j], [lo, hi], color="black", lw=1.2, zorder=5)
            axB.plot([j, j], [lo, lo], color="black", lw=1.2, marker="_", ms=7)
            axB.plot([j, j], [hi, hi], color="black", lw=1.2, marker="_", ms=7)
        else:
            axB.plot(j, sp[j], marker="D", color="black", ms=4, zorder=6)
        axB.text(j, sp[j] / 2, f"{sp[j]:+.2f}", ha="center", va="center",
                 rotation=90, fontsize=8, color="white", zorder=7)  # 竖排柱心标注，规避柱宽限制
    axB.axhline(0, color="grey", lw=0.8, ls="--")
    axB.set_xticks(x, SHORT)
    for lbl in axB.get_xticklabels():
        lbl.set(rotation=30, ha="right", fontsize=8)
    axB.set_ylim(-0.9, 0.85)
    axB.set_ylabel("Spearman (sfate vs CellRank)")
    axB.set_title("B  Per-arm fate concordance\n(band = across-assignment-rule range)",
                  fontsize=10)
    axB.text(0.02, 0.97, "red = discordant arm", transform=axB.transAxes,
             fontsize=8, color="#b2182b", va="top")  # 左上空白区，C1q 柱高 0.67 之上

    # ---- Panel C: self-absorption block ------------------------------------
    axC = fig.add_subplot(gs[0, 2])
    im2 = axC.imshow(block, cmap="viridis", vmin=0.04, vmax=0.24)
    axC.set_xticks(range(6), SHORT)
    plt.setp(axC.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    axC.set_yticks(range(6), SHORT)
    for i in range(6):
        for j in range(6):
            axC.text(j, i, f"{block[i, j]:.3f}", ha="center", va="center",
                     fontsize=7, color="white" if block[i, j] > 0.15 else "black")
    for i in range(6):  # diagonal boxes (self-absorption)
        axC.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                    edgecolor="red", lw=1.6))
    fig.colorbar(im2, ax=axC, shrink=0.85, label="mean absorption")
    axC.set_title("C  Self-absorption block\n(own state × target arm)", fontsize=10)
    fig.text(0.5, 0.005, "red box = self-absorption (diagonal); Prolif lowest "
                         "(0.147); all→Prolif ≈ 0.057", ha="center", fontsize=8)

    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_PDF)
    print(f"[written] {OUT_PNG}")
    print(f"[written] {OUT_PDF}")


if __name__ == "__main__":
    main()
