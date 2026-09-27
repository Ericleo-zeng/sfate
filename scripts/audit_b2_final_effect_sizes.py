#!/usr/bin/env python
"""Final locked-run concordance: quantitative effect sizes + commitment.

The 'final locked CellRank run' is the Layer B2 audit run
(output/audit_b2_step3b_final_v2.npz): same production P, explicit six
annotation-matched terminals, cellrank 2.1.0 + krylov-Schur/PETSc-SLEPc;
fate profiles verified bit-stable across repeated runs.

This script produces the quantitative comparison for the AD paper:
  (a) paired genotype-delta effect sizes over the 27-cell grid
      (6 arms x 2 contrasts x 3 ages), mean-per-sample protocol
      (samples are the aggregation unit, never cells);
  (b) per-cell commitment (max arm probability) comparison, whole-atlas
      and per sample by genotype x age;
  (c) objective cross-check of sfate-side per-sample means against the
      archived c_layer analysis (mirrors test A's xref gate).

Run from ~/research_storage/工作文件/Cellrank重构
Output: output/audit_b2_final_effect_sizes_log.md
"""
from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
from scipy.stats import pearsonr, spearmanr

ANNOT_H5AD = Path("/mnt/t9/datasets/ad_microglia_fate_landscape_output/06_annotation/adata_annotated.h5ad")
FATE75K = Path("output/fate_full_75k.h5ad")
STEP3B_NPZ = Path("output/audit_b2_step3b_final_v2.npz")
CLAYER_RESULTS = Path("output/c_layer_results.json")
OUT_LOG = Path("output/audit_b2_final_effect_sizes_log.md")
M3_ORDER = ["C1q_inflammatory", "DAM_like", "Homeostatic",
            "IFN_responsive", "Proliferating", "PU1low_lymphoid"]
CONTRASTS = [("5xFAD", "control", "5xFAD-control"),
             ("5xFAD;Cd28-cKO", "5xFAD", "cKO-5xFAD")]


def read_obs_column(f: h5py.File, key: str) -> np.ndarray:
    g = f["obs"][key]
    if isinstance(g, h5py.Group):
        cats = [c.decode() if isinstance(c, bytes) else str(c)
                for c in g["categories"][:]]
        codes = np.asarray(g["codes"][:])
        out = np.empty(len(codes), dtype=object)
        m = codes >= 0
        out[m] = np.array(cats, dtype=object)[codes[m]]
        out[~m] = None
        return out
    arr = np.asarray(g[...])
    return np.array([x.decode() if isinstance(x, bytes) else str(x)
                     for x in arr], dtype=object)


def main() -> None:
    z = np.load(STEP3B_NPZ, allow_pickle=True)
    states = [str(s) for s in z["states"]]
    assert states == M3_ORDER, f"states {states}"
    F_cr = np.asarray(z["F_cellrank"], dtype=np.float64)
    F_sf = np.asarray(z["F_sfate"], dtype=np.float64)

    idx_prod = np.asarray(ad.read_h5ad(FATE75K).obs_names).astype(str)
    with h5py.File(ANNOT_H5AD, "r") as f:
        gobs = f["obs"]
        idx_ann = np.asarray(gobs[gobs.attrs["_index"]][...]).astype(str)
        s_sample = read_obs_column(f, "sample")
        s_genotype = read_obs_column(f, "genotype")
        s_age = read_obs_column(f, "age_months")
    lut = {b: i for i, b in enumerate(idx_ann)}
    sel = np.array([lut[b] for b in idx_prod], dtype=np.int64)
    sample = s_sample[sel]
    genotype = s_genotype[sel]
    age = np.array([str(a) for a in s_age[sel]], dtype=object)

    rows, samples_by = {}, {}
    for i, s in enumerate(sample):
        rows.setdefault(s, []).append(i)
    for s, idx in rows.items():
        gs = {genotype[i] for i in idx}
        as_ = {age[i] for i in idx}
        assert len(gs) == 1 and len(as_) == 1, f"sample {s} mixed meta"
        samples_by.setdefault((genotype[idx[0]], age[idx[0]]), []).append(s)
    assert len(rows) == 12, f"expected 12 samples, got {len(rows)}"
    cell_rows = {s: np.array(v, dtype=np.int64) for s, v in rows.items()}

    def delta(F, j, g1, g0, a):
        s1, s0 = samples_by.get((g1, a), []), samples_by.get((g0, a), [])
        if not s1 or not s0:
            return None
        return float(np.mean([F[cell_rows[s], j].mean() for s in s1]) -
                     np.mean([F[cell_rows[s], j].mean() for s in s0]))

    ages = sorted({a for (g, a) in samples_by})
    grid, d_cr, d_sf = [], [], []
    for j, arm in enumerate(M3_ORDER):
        for (g1, g0, tag) in CONTRASTS:
            for a in ages:
                x_cr, x_sf = delta(F_cr, j, g1, g0, a), delta(F_sf, j, g1, g0, a)
                grid.append({"arm": arm, "contrast": tag, "age": a,
                             "delta_cellrank": x_cr, "delta_sfate": x_sf})
                if x_cr is not None and x_sf is not None:
                    d_cr.append(x_cr)
                    d_sf.append(x_sf)
    d_cr, d_sf = np.array(d_cr), np.array(d_sf)

    overall = {
        "n_paired": int(len(d_cr)),
        "pearson_r": float(pearsonr(d_cr, d_sf).statistic),
        "spearman_rho": float(spearmanr(d_cr, d_sf).statistic),
        "mae": float(np.mean(np.abs(d_cr - d_sf))),
        "max_abs_diff": float(np.max(np.abs(d_cr - d_sf))),
    }

    # per-arm deltas for the cKO contrast at 3M/6M (manuscript spot values)
    spot = {}
    for r in grid:
        if r["contrast"] == "cKO-5xFAD" and r["age"] in ("3", "6"):
            spot.setdefault(r["age"], {})[r["arm"]] = {
                "cellrank": r["delta_cellrank"], "sfate": r["delta_sfate"]}

    # commitment = per-cell max arm probability
    com_cr, com_sf = F_cr.max(axis=1), F_sf.max(axis=1)
    commitment = {
        "per_cell_spearman": float(spearmanr(com_cr, com_sf).statistic),
        "per_cell_pearson": float(pearsonr(com_cr, com_sf).statistic),
        "by_sample": {},
    }
    for s in sorted(rows):
        idx = cell_rows[s]
        g, a = genotype[idx[0]], age[idx[0]]
        commitment["by_sample"][f"{s}|{g}|{a}"] = {
            "cellrank": float(com_cr[idx].mean()),
            "sfate": float(com_sf[idx].mean()),
        }

    # xref: sfate per-sample arm means vs archived c_layer analysis
    xref = "c_layer_results.json not found"
    if CLAYER_RESULTS.exists():
        arch = {r["sample"]: r for r in
                json.loads(CLAYER_RESULTS.read_text())["sample_level"]["sfate_sample_means"]}
        mism, checked = 0, 0
        for s in sorted(rows):
            if s not in arch:
                continue
            idx = cell_rows[s]
            for j, arm in enumerate(M3_ORDER):
                v_mine = float(F_sf[idx, j].mean())
                v_arch = float(arch[s][arm])
                checked += 1
                if abs(v_mine - v_arch) > 1e-4:
                    mism += 1
        xref = {"checked": checked, "mismatches_gt_1e-4": mism}

    OUT_LOG.write_text(json.dumps({
        "overall_paired_effect_sizes": overall,
        "cKO_contrast_spot_values_by_age": spot,
        "commitment": commitment,
        "xref_sfate_sample_means_vs_c_layer": xref,
        "full_grid": grid,
    }, ensure_ascii=False, indent=2))
    print(f"DONE n={overall['n_paired']} pearson={overall['pearson_r']:.4f} "
          f"spearman={overall['spearman_rho']:.4f} mae={overall['mae']:.5f} "
          f"xref={xref}", flush=True)


if __name__ == "__main__":
    main()
