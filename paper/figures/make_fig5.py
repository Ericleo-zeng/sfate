#!/usr/bin/env python3
"""make_fig5.py — 75k fate panorama: sfate absorption probabilities for all six annotated states."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FATE_H5AD = ROOT / "output/fate_full_75k.h5ad"
SRC_H5AD = Path(
    "/mnt/t9/datasets/ad_microglia_fate_landscape_output/"
    "06_annotation/adata_annotated.h5ad"
)
C_LAYER_MD = ROOT / "output/c_layer_report.md"
OUT_PNG = Path(__file__).resolve().parent / "fig5.png"
OUT_PDF = Path(__file__).resolve().parent / "fig5.pdf"

C_CLIFF = "#a40000"


def parse_cr_uncomparable() -> list[str]:
    m = re.search(r"CR 侧无 fate 列的注释类（不可比）：\[([^\]]+)\]",
                  C_LAYER_MD.read_text(encoding="utf-8"))
    return [s.strip().strip("'") for s in m.group(1).split(",")]


def main() -> int:
    with h5py.File(FATE_H5AD, "r") as g:
        probs = g["obsm/sfate_fate"][:]
        bc_fate = g["obs/cell_barcode"][:].astype(str)
        classes = [c.decode() if isinstance(c, bytes) else str(c)
                   for c in g["uns/sfate_meta/classes"][:]]
        meta = {k: g["uns/sfate_meta"][k][()] for k in g["uns/sfate_meta"]}

    with h5py.File(SRC_H5AD, "r") as f:
        bc_src = f["obs/_index"][:].astype(str)
        umap = f["obsm/X_umap"][:]

    assert probs.shape == (74984, 6)
    assert np.array_equal(bc_fate, bc_src)
    assert umap.shape == (74984, 2)
    assert ((probs.sum(axis=1) >= 0.99) & (probs.sum(axis=1) <= 1.01)).all()

    uncomparable = parse_cr_uncomparable()
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.6), layout="constrained")
    fig.get_layout_engine().set(rect=(0.0, 0.045, 1.0, 0.94))
    sm = None
    order = classes
    panel_note = {
        "DAM_like": "no CR fate column (unsupported)",
        "PU1low_lymphoid": "no CR fate column (unsupported)",
        "Proliferating": "no CR fate column",
        "IFN_responsive": "CR: split into 3 substates",
        "C1q_inflammatory": "CR comparable arm",
        "Homeostatic": "CR comparable arm",
    }
    for ax, cls in zip(axes.flat, order):
        j = classes.index(cls)
        asc = np.argsort(probs[:, j], kind="stable")
        sm = ax.scatter(umap[asc, 0], umap[asc, 1], c=probs[asc, j],
                        s=0.75, cmap="viridis", vmin=0, vmax=1,
                        alpha=0.65, rasterized=True)
        note = panel_note.get(cls, "")
        comparable = cls not in uncomparable
        note_color = "#444444" if comparable else C_CLIFF
        ax.set_title(cls, fontsize=11, weight="bold")
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=8.3,
                color=note_color, weight="normal" if comparable else "bold",
                va="bottom",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
    cbar = fig.colorbar(sm, ax=list(axes.flat), location="right",
                        fraction=0.03, pad=0.02)
    cbar.set_label("absorption probability\n(linear 0–1, shared)", fontsize=9.5)
    cbar.ax.tick_params(labelsize=8.5)

    solve_mode = meta["solve_mode"]
    solve_mode = solve_mode.decode() if isinstance(solve_mode, bytes) else str(solve_mode)
    fig.suptitle(
        "75k fate panorama: sfate absorption probabilities for all six annotated states "
        "in the mouse 5xFAD/Cd28-cKO microglia atlas",
        fontsize=12, weight="bold")
    fig.text(0.01, 0.022,
             f"Absorption probabilities, sfate 6-state full annotation "
             f"(solve_mode={solve_mode}, k=30 latent kNN, r=30 absorbing representatives per state). "
             f"CR comparable arms: 3 (IFN/C1q/Homeostatic). "
             f"Dataset: Ayata et al. 2025; GEO: GSE296768.",
             fontsize=7.4, color="#666666")
    fig.savefig(OUT_PNG, dpi=300, facecolor="white")
    fig.savefig(OUT_PDF, facecolor="white")
    print(f"written: {OUT_PNG}\nwritten: {OUT_PDF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
