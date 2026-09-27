#!/usr/bin/env python
"""Layer B2 audit - Step 2c: recover cell-order permutation, redo matching.

Step 2b: agreement=0/6 between rep-based (cache-order) and full-label
(annotated-h5ad-order) matching -> systematic row-order mismatch between
adata_annotated.h5ad obs and the production graph cache (counts matched,
order did not). This script:
  1. keys-match rows of obsm['latent'] (fate_full_75k.h5ad, production order)
     against obsm['X_scVI'] (annotated h5ad, h5py read) via rounded-row keys;
  2. applies the recovered permutation to the full annotation vector;
  3. recomputes macro<->annotation matching (expect ~6/6 agreement with the
     180-rep matching from Step 2);
  4. remaps the SAVED raw CellRank fate (audit_b2_step2b_cellrank_fate.npz)
     onto annotation columns and recomputes per-state Spearman. No CellRank
     recompute needed.

Outputs: output/audit_b2_step2c_order_fix.npz, output/audit_b2_step2c_log.md
"""
from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
from scipy.stats import spearmanr

ANNOT_H5AD = Path("/mnt/t9/datasets/ad_microglia_fate_landscape_output/06_annotation/adata_annotated.h5ad")
GRAPH_CACHE = Path("output/c_layer_graph.npz")
STEP1_NPZ = Path("output/audit_b2_step1_macrostates.npz")
STEP2B_NPZ = Path("output/audit_b2_step2b_cellrank_fate.npz")
FATE75K = Path("output/fate_full_75k.h5ad")
OUT_NPZ = Path("output/audit_b2_step2c_order_fix.npz")
OUT_LOG = Path("output/audit_b2_step2c_log.md")
M3_ORDER = ["C1q_inflammatory", "DAM_like", "Homeostatic",
            "IFN_responsive", "Proliferating", "PU1low_lymphoid"]


def read_obsm_dense(path: Path, key: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return np.asarray(f["obsm"][key][:], dtype=np.float64)


def row_keys(x: np.ndarray, ndigits: int = 5) -> np.ndarray:
    q = np.round(x, ndigits)
    return q.view(np.dtype((np.void, q.dtype.itemsize * q.shape[1]))).ravel()


def main() -> None:
    z = np.load(GRAPH_CACHE, allow_pickle=True)
    labels_rep = np.asarray(z["labels"])
    s1 = np.load(STEP1_NPZ, allow_pickle=True)
    memb = np.asarray(s1["memberships"], dtype=np.float32)

    labels_full = None
    with h5py.File(ANNOT_H5AD, "r") as f:
        g = f["obs"]["microglia_state"]
        cats = [c.decode() if isinstance(c, bytes) else str(c) for c in g["categories"][:]]
        codes = g["codes"][:]
    labels_h5ad = np.empty(len(codes), dtype=object)
    m = codes >= 0
    labels_h5ad[m] = np.array(cats, dtype=object)[codes[m]]
    labels_h5ad[~m] = None

    adata_sf = ad.read_h5ad(FATE75K)
    latent_keys = [k for k in adata_sf.obsm.keys()
                   if np.asarray(adata_sf.obsm[k]).shape == (74_984, 30)]
    assert len(latent_keys) == 1, f"ambiguous latent keys: {latent_keys}"
    print(f"[i] latent obsm key: {latent_keys[0]}", flush=True)
    prod = np.asarray(adata_sf.obsm[latent_keys[0]], dtype=np.float64)
    ref = read_obsm_dense(ANNOT_H5AD, "X_scVI")
    assert prod.shape == ref.shape == (74_984, 30), (prod.shape, ref.shape)

    kp, kr = row_keys(prod), row_keys(ref)
    exact = float(np.mean(kp == kr))
    # permutation via unique rounded-row keys
    lut = {}
    dup = 0
    for j, k in enumerate(kr):
        if k in lut:
            dup += 1
        else:
            lut[k] = j
    perm = np.array([lut.get(k, -1) for k in kp], dtype=np.int64)
    matched = float(np.mean(perm >= 0))

    labels_aligned = np.empty(74_984, dtype=object)
    ok = perm >= 0
    labels_aligned[ok] = labels_h5ad[perm[ok]]
    labels_aligned[~ok] = None

    def match(labels):
        hard = memb.argmax(axis=1)
        lab = [x for x in labels.tolist() if x is not None]
        anns = sorted(set(lab))
        C = np.zeros((6, len(anns)), dtype=np.int64)
        for mm in range(6):
            lm = labels[hard == mm]
            for j, a in enumerate(anns):
                C[mm, j] = int(np.sum(lm == a))
        W = C.copy()
        pairs = {}
        for _ in range(6):
            i, j = np.unravel_index(int(np.argmax(W)), W.shape)
            pairs[str(i)] = anns[j]
            W[i, :] = -1
            W[:, j] = -1
        return pairs, C.tolist(), anns

    pairs_aligned, C_aligned, anns = match(labels_aligned)
    pairs_rep, _, _ = match(labels_rep)
    agreement = sum(1 for k in pairs_aligned if pairs_rep.get(k) == pairs_aligned[k])

    s2b = np.load(STEP2B_NPZ, allow_pickle=True)
    F_cr_raw = np.asarray(s2b["F_cellrank_raw_by_macro"], dtype=np.float32)
    cr_cols = [str(c) for c in s2b["cr_macro_columns"]]
    F_sf = np.asarray(s2b["F_sfate"], dtype=np.float32)
    states = [str(s) for s in s2b["states"]]
    if set(states) != set(M3_ORDER):
        states = [s for s in M3_ORDER if s in set(states)]

    ann_of_macro = {name: pairs_aligned[str(i)] for i, name in enumerate([str(x) for x in s1["macro_names"]])}
    F_cr_ann = np.zeros((74_984, 6), dtype=np.float32)
    for j, sname in enumerate(states):
        src = [k for k, cname in enumerate(cr_cols) if ann_of_macro.get(cname) == sname]
        assert len(src) == 1, f"{sname}: {len(src)} macro columns"
        F_cr_ann[:, j] = F_cr_raw[:, src[0]]
    spearman = {s: float(spearmanr(F_cr_ann[:, j], F_sf[:, j]).statistic)
                for j, s in enumerate(states)}

    np.savez(OUT_NPZ, F_cellrank=F_cr_ann, F_sfate=F_sf,
             states=np.array(states),
             spearman=np.array([spearman[s] for s in states]),
             perm=perm, labels_aligned=labels_aligned)
    OUT_LOG.write_text(json.dumps({
        "row_exact_match_fraction": exact,
        "key_match_rate": matched,
        "key_duplicates_in_ref": dup,
        "matching_agreement_with_rep_6": agreement,
        "matching_aligned": ann_of_macro,
        "matching_rep_based": {k: pairs_rep[str(i)] for i, k in enumerate([str(x) for x in s1["macro_names"]])},
        "contingency_aligned": C_aligned,
        "contingency_annotations": anns,
        "per_state_spearman": spearman,
    }, ensure_ascii=False, indent=2))
    print(f"DONE exact={exact:.4f} matched={matched:.4f} agreement={agreement}/6 "
          f"spearman={spearman}", flush=True)


if __name__ == "__main__":
    main()
