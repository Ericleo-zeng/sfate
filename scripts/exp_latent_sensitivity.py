#!/usr/bin/env python3
"""Latent-representation sensitivity on the 5k fixture.

Compares the production sfate pipeline when the latent embedding is changed from the
reference obsm['X_scVI'] (30D) to a PCA-30D embedding of the same cells (top 2,000 HVGs).
Optional scVI-10D/50D re-trainings are attempted if SCVI_DIMS is set and training finishes
within the budget (default 15 min per dim).

Outputs: output/latent_sensitivity.md, output/latent_sensitivity.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import anndata as ad
from sklearn.decomposition import PCA

from sfate import absorption_probabilities, build_transition_graph

SEED = 20260909
N_HVG = 2000
SRC = os.environ.get("SFATE_SOURCE_H5AD", None)  # path to adata_annotated.h5ad; source atlas not included in the repository
FIXTURE = ROOT / "output" / "fixture_5k.h5ad"
OUT_MD = ROOT / "output" / "latent_sensitivity.md"
OUT_JSON = ROOT / "output" / "latent_sensitivity.json"


def stratified_sample(state: pd.Series, n_target: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    counts = state.value_counts()
    exact = n_target * counts / counts.sum()
    base = exact.astype(int)
    for lvl in counts.index:
        if counts[lvl] >= 1 and base[lvl] == 0:
            base[lvl] = 1
    deficit = n_target - base.sum()
    if deficit > 0:
        remainders = (exact - base).sort_values(ascending=False)
        for lvl in remainders.index:
            if deficit == 0:
                break
            if base[lvl] < counts[lvl]:
                base[lvl] += 1
                deficit -= 1
    idx_all = np.arange(len(state))
    chosen = []
    for lvl, n_take in base.items():
        lvl_idx = idx_all[state.to_numpy() == lvl]
        chosen.append(rng.choice(lvl_idx, size=n_take, replace=False))
    return np.sort(np.concatenate(chosen))


def pca_embedding(full: ad.AnnData, sel: np.ndarray, n_hvg: int = N_HVG, n_components: int = 30) -> np.ndarray:
    """Return (n_sel, n_components) PCA float32 on log-normalized HVG subset."""
    var = full.var
    if "highly_variable" in var.columns and var["highly_variable"].sum() >= n_hvg:
        hvg_mask = var["highly_variable"].to_numpy()
        # if more than n_hvg, take by rank
        if hvg_mask.sum() > n_hvg and "highly_variable_rank" in var.columns:
            rank = var["highly_variable_rank"].to_numpy(dtype=float)
            rank[~hvg_mask] = np.inf
            threshold = np.partition(rank[rank < np.inf], n_hvg - 1)[n_hvg - 1]
            hvg_mask = rank <= threshold
    elif "highly_variable_rank" in var.columns:
        rank = var["highly_variable_rank"].to_numpy(dtype=float)
        threshold = np.partition(rank, n_hvg - 1)[n_hvg - 1]
        hvg_mask = rank <= threshold
    else:
        raise ValueError("source has no HVG annotation")
    hvg_idx = np.where(hvg_mask)[0]
    print(f"[PCA] HVG selected: {len(hvg_idx)}")
    X = full.X[sel][:, hvg_idx]
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    # center and scale per-gene
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    X = (X - mean) / std
    pca = PCA(n_components=n_components, random_state=SEED)
    Z = pca.fit_transform(X).astype(np.float32)
    print(f"[PCA] explained variance ratio sum: {pca.explained_variance_ratio_.sum():.4f}")
    return Z


def scvi_embedding(full: ad.AnnData, sel: np.ndarray, n_latent: int, budget_s: float = 900.0) -> np.ndarray | None:
    try:
        import scvi
    except Exception as e:
        print(f"[scVI-{n_latent}D] scvi-tools not importable: {e}")
        return None
    try:
        sub = full[sel].copy()
        # use counts layer if present
        if "counts" in sub.layers:
            sub.X = sub.layers["counts"]
        scvi.model.SCVI.setup_anndata(sub)
        model = scvi.model.SCVI(sub, n_latent=n_latent, n_layers=2, n_hidden=128)
        t0 = time.perf_counter()
        model.train(max_epochs=200, accelerator="cpu", devices=1, enable_progress_bar=False)
        elapsed = time.perf_counter() - t0
        print(f"[scVI-{n_latent}D] trained in {elapsed:.1f}s")
        if elapsed > budget_s:
            print(f"[scVI-{n_latent}D] over budget ({budget_s}s), discarding")
            return None
        return model.get_latent_representation().astype(np.float32)
    except Exception as e:
        print(f"[scVI-{n_latent}D] failed: {e}")
        return None


def select_representatives(Z: np.ndarray, labels: np.ndarray, r: int = 30) -> np.ndarray:
    """Return per-cell labels with None for non-representative cells."""
    n = Z.shape[0]
    out = np.array([None] * n, dtype=object)
    Z64 = Z.astype(np.float64)
    for c in np.unique(labels):
        idx = np.nonzero(labels == c)[0]
        centroid = Z64[idx].mean(axis=0)
        dist = np.linalg.norm(Z64[idx] - centroid, axis=1)
        reps = idx[np.argsort(dist, kind="stable")[:r]]
        for i in reps:
            out[i] = c
    return out


def run_pipeline(Z: np.ndarray, labels: np.ndarray, name: str) -> dict:
    print(f"[{name}] graph construction ...")
    P = build_transition_graph(Z, k=30, random_state=SEED)
    rep_labels = select_representatives(Z, labels, r=30)
    print(f"[{name}] solve ...")
    B, info = absorption_probabilities(P, rep_labels, tol=1e-6, return_info=True)
    return {"P": P, "B": B, "info": info}


def compare(ref_B: np.ndarray, test_B: np.ndarray, classes: np.ndarray) -> dict:
    per_state = {}
    for j, c in enumerate(classes):
        s = spearmanr(ref_B[:, j], test_B[:, j]).statistic
        mae = float(np.mean(np.abs(ref_B[:, j] - test_B[:, j])))
        linf = float(np.max(np.abs(ref_B[:, j] - test_B[:, j])))
        per_state[str(c)] = {"spearman": float(s), "mae": mae, "linf": linf}
    row_dev = float(np.max(np.abs(test_B.sum(axis=1) - 1.0)))
    return {
        "median_spearman": float(np.median([v["spearman"] for v in per_state.values()])),
        "mean_mae": float(np.mean([v["mae"] for v in per_state.values()])),
        "max_linf": float(np.max([v["linf"] for v in per_state.values()])),
        "row_sum_max_deviation": row_dev,
        "per_state": per_state,
    }


def main() -> int:
    t_start = time.perf_counter()
    warnings.filterwarnings("ignore", category=UserWarning)
    fixture = ad.read_h5ad(FIXTURE)
    ref_Z = np.asarray(fixture.obsm["X_scVI"], dtype=np.float32)
    labels = fixture.obs["microglia_state"].to_numpy()
    print(f"[ref] X_scVI shape {ref_Z.shape}")

    # source selection
    if SRC is None: raise SystemExit("Set SFATE_SOURCE_H5AD to the path of adata_annotated.h5ad (source atlas not included in the repository).")
    full = ad.read_h5ad(SRC, backed="r")
    sel = stratified_sample(full.obs["microglia_state"], 5000, SEED)
    print(f"[source] selected {len(sel)} cells matching fixture")

    results = {"seed": SEED, "n_cells": 5000, "representations": {}}

    ref_res = run_pipeline(ref_Z, labels, "scVI_30D")
    results["representations"]["scVI_30D"] = {"nnz": int(ref_res["P"].nnz)}

    # PCA
    t0 = time.perf_counter()
    pca_Z = pca_embedding(full, sel)
    pca_time = time.perf_counter() - t0
    pca_res = run_pipeline(pca_Z, labels, "PCA_30D")
    pca_cmp = compare(ref_res["B"], pca_res["B"], pca_res["info"]["classes"])
    results["representations"]["PCA_30D"] = {
        "embedding_time_s": round(pca_time, 2),
        "nnz": int(pca_res["P"].nnz),
        "metrics": pca_cmp,
    }

    # optional scVI dims
    scvi_dims = []
    if os.environ.get("SCVI_DIMS"):
        scvi_dims = [int(x) for x in os.environ.get("SCVI_DIMS", "").split(",") if x]
    for d in scvi_dims:
        Zd = scvi_embedding(full, sel, d)
        if Zd is None:
            continue
        r = run_pipeline(Zd, labels, f"scVI_{d}D")
        cmp = compare(ref_res["B"], r["B"], r["info"]["classes"])
        results["representations"][f"scVI_{d}D"] = {
            "nnz": int(r["P"].nnz),
            "metrics": cmp,
        }

    results["elapsed_total_s"] = round(time.perf_counter() - t_start, 2)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md = ["# Latent representation sensitivity (5k fixture)", ""]
    md.append(f"Reference: obsm['X_scVI'] (30D), n={results['n_cells']}, seed={results['seed']}.\n")
    for name, data in results["representations"].items():
        md.append(f"## {name}")
        if "metrics" in data:
            m = data["metrics"]
            md.append(f"- median Spearman vs reference: {m['median_spearman']:.4f}")
            md.append(f"- mean MAE vs reference: {m['mean_mae']:.4f}")
            md.append(f"- max L∞ vs reference: {m['max_linf']:.4e}")
            md.append(f"- row-sum max deviation from 1.0: {m['row_sum_max_deviation']:.2e}")
        md.append(f"- transition matrix nnz: {data['nnz']}")
        if "embedding_time_s" in data:
            md.append(f"- embedding computation time: {data['embedding_time_s']} s")
        md.append("")
    md.append(f"Total elapsed: {results['elapsed_total_s']} s.")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    print(f"written: {OUT_JSON} / {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
