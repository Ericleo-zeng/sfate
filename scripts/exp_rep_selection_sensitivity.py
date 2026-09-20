#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_rep_selection_sensitivity.py — Representative-selection sensitivity on the 5k fixture.

Isolates the absorbing-state boundary condition by building the kNN graph once and
varying only the representative cells selected from each annotated class.

Reference : centroid-nearest r=30 cells per class in the 30-d scVI latent space.
Perturbation : 10 random 30-cell samples per class (without replacement; classes with
fewer than r cells use all cells).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata
import numpy as np
import scipy.stats as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfate.graph import build_transition_graph
from sfate.fate import absorption_probabilities

FIXTURE = ROOT / "output/fixture_5k.h5ad"
OUT_MD = ROOT / "output/rep_selection_sensitivity.md"
OUT_JSON = ROOT / "output/rep_selection_sensitivity.json"

K = 30
R = 30
KNN_SEED = 20260909
RANDOM_SEEDS = list(range(20260909, 20260909 + 10))


def centroid_nearest(latent, labels, classes, r):
    refs: dict[str, np.ndarray] = {}
    for cls in classes:
        mask = labels == cls
        idx = np.nonzero(mask)[0]
        center = latent[idx].mean(axis=0)
        dist2 = ((latent[idx] - center) ** 2).sum(axis=1)
        order = np.argsort(dist2, kind="stable")
        refs[cls] = idx[order[:min(r, len(idx))]]
    return refs


def random_selection(latent, labels, classes, r, rng):
    sel: dict[str, np.ndarray] = {}
    for cls in classes:
        idx = np.nonzero(labels == cls)[0]
        size = min(r, len(idx))
        sel[cls] = rng.choice(idx, size=size, replace=False)
    return sel


def make_terminal_labels(n, selection, classes):
    labels = np.full(n, None, dtype=object)
    for cls in classes:
        labels[selection[cls]] = cls
    return labels


def compute_fates(P, labels):
    return absorption_probabilities(
        P,
        labels,
        block_size=256,
        tol=1e-6,
        restart=50,
        maxiter=1000,
        solve_mode="columns",
        preconditioner="none",
        return_info=False,
    )


def pairwise_overlap_fraction(sets):
    """Mean |intersection| / |union| across unordered pairs."""
    m = len(sets)
    if m < 2:
        return 1.0
    vals = []
    for i in range(m):
        for j in range(i + 1, m):
            vals.append(len(np.intersect1d(sets[i], sets[j])) / len(np.union1d(sets[i], sets[j])))
    return float(np.mean(vals))


def main() -> int:
    ad = anndata.read_h5ad(FIXTURE)
    latent = np.asarray(ad.obsm["X_scVI"], dtype=np.float32)
    labels = np.asarray(ad.obs["microglia_state"].astype(str))
    classes = np.unique(labels)
    n = latent.shape[0]

    print("Building kNN graph once (k=30, seed=20260909)...", file=sys.stderr)
    P = build_transition_graph(
        latent,
        k=K,
        kernel="gaussian",
        symmetrize="max",
        knn_backend="pynndescent",
        random_state=KNN_SEED,
    )
    print(f"Graph nnz = {P.nnz}", file=sys.stderr)

    ref_sel = centroid_nearest(latent, labels, classes, R)
    ref_fates = compute_fates(P, make_terminal_labels(n, ref_sel, classes))

    per_class: dict[str, dict] = {cls: {"spearman": [], "mae": [], "linf": [], "overlap_vs_ref": []} for cls in classes}
    all_random_sets: dict[str, list[np.ndarray]] = {cls: [] for cls in classes}

    for seed in RANDOM_SEEDS:
        rng = np.random.default_rng(seed)
        sel = random_selection(latent, labels, classes, R, rng)
        for cls in classes:
            all_random_sets[cls].append(sel[cls])
        fates = compute_fates(P, make_terminal_labels(n, sel, classes))
        for ci, cls in enumerate(classes):
            ref_col = ref_fates[:, ci]
            rand_col = fates[:, ci]
            rho, _ = st.spearmanr(ref_col, rand_col)
            per_class[cls]["spearman"].append(float(rho))
            per_class[cls]["mae"].append(float(np.mean(np.abs(ref_col - rand_col))))
            per_class[cls]["linf"].append(float(np.max(np.abs(ref_col - rand_col))))
            per_class[cls]["overlap_vs_ref"].append(
                float(len(np.intersect1d(sel[cls], ref_sel[cls])) / len(np.union1d(sel[cls], ref_sel[cls])))
            )

    summary = {}
    for cls in classes:
        vals = per_class[cls]
        summary[cls] = {
            "n_cells": int(np.sum(labels == cls)),
            "ref_n": int(len(ref_sel[cls])),
            "median_spearman": float(np.median(vals["spearman"])),
            "spearman_5_95": [float(np.percentile(vals["spearman"], 5)), float(np.percentile(vals["spearman"], 95))],
            "mean_mae": float(np.mean(vals["mae"])),
            "mae_5_95": [float(np.percentile(vals["mae"], 5)), float(np.percentile(vals["mae"], 95))],
            "max_linf": float(np.max(vals["linf"])),
            "overlap_vs_ref_mean": float(np.mean(vals["overlap_vs_ref"])),
            "random_random_jaccard": float(pairwise_overlap_fraction(all_random_sets[cls])),
        }

    overall = {
        "median_spearman": float(np.median([summary[cls]["median_spearman"] for cls in classes])),
        "median_mae": float(np.median([summary[cls]["mean_mae"] for cls in classes])),
        "max_linf": float(np.max([summary[cls]["max_linf"] for cls in classes])),
    }

    result = {
        "n_cells": n,
        "r": R,
        "k": K,
        "knn_seed": KNN_SEED,
        "random_seeds": RANDOM_SEEDS,
        "graph_nnz": int(P.nnz),
        "summary": summary,
        "overall": overall,
    }

    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# Representative-selection sensitivity (5k fixture)",
        "",
        f"Graph built once: k={K}, seed={KNN_SEED}, nnz={P.nnz}.",
        f"Reference: centroid-nearest r={R} per class. Perturbation: {len(RANDOM_SEEDS)} random r={R} selections per class.",
        "",
        "## Per-class summary",
        "",
        "| state | n cells | median Spearman | 5–95% Spearman | mean MAE | 5–95% MAE | max L∞ | mean overlap vs ref | random–random Jaccard |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cls in classes:
        s = summary[cls]
        md_lines.append(
            f"| {cls} | {s['n_cells']} | {s['median_spearman']:.3f} | "
            f"[{s['spearman_5_95'][0]:.3f}, {s['spearman_5_95'][1]:.3f}] | "
            f"{s['mean_mae']:.3f} | [{s['mae_5_95'][0]:.3f}, {s['mae_5_95'][1]:.3f}] | "
            f"{s['max_linf']:.3f} | {s['overlap_vs_ref_mean']:.2f} | {s['random_random_jaccard']:.2f} |"
        )
    md_lines += [
        "",
        "## Overall",
        "",
        f"- Median per-state Spearman: {overall['median_spearman']:.3f}",
        f"- Median per-state MAE: {overall['median_mae']:.3f}",
        f"- Max L∞ across all states and draws: {overall['max_linf']:.3f}",
        "",
    ]
    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"written: {OUT_JSON}\nwritten: {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
