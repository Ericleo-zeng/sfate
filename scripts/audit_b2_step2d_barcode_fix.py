#!/usr/bin/env python
"""Layer B2 audit - Step 2d: barcode-based order alignment + final concordance.

Step 2c abandoned: v2 fate h5ad carries no latent. Barcode route instead:
  - fate_full_75k.h5ad obs_names = production (graph-cache) order;
  - adata_annotated.h5ad obs index = its own (possibly reordered) order;
  - map annotations across via barcode dictionary.
HARD GATE: cache rep labels (180, cache order) must agree 180/180 with the
barcode-mapped annotations; anything less means the order assumption is broken
and nothing downstream is trustworthy. Then: recompute matching (expect ~6/6
agreement with the rep-based matching), remap the saved raw CellRank fate from
Step 2b, final per-state Spearman. No CellRank recompute.

Outputs: output/audit_b2_step2d_barcode_fix.npz, output/audit_b2_step2d_log.md
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
OUT_NPZ = Path("output/audit_b2_step2d_barcode_fix.npz")
OUT_LOG = Path("output/audit_b2_step2d_log.md")
M3_ORDER = ["C1q_inflammatory", "DAM_like", "Homeostatic",
            "IFN_responsive", "Proliferating", "PU1low_lymphoid"]


def read_ann_obs(path: Path):
    with h5py.File(path, "r") as f:
        gobs = f["obs"]
        idx = np.asarray(gobs[gobs.attrs["_index"]][...]).astype(str)
        g = gobs["microglia_state"]
        cats = [c.decode() if isinstance(c, bytes) else str(c) for c in g["categories"][:]]
        codes = np.asarray(g["codes"][:])
    lab = np.empty(len(codes), dtype=object)
    m = codes >= 0
    lab[m] = np.array(cats, dtype=object)[codes[m]]
    lab[~m] = None
    return idx, lab


def match(memb: np.ndarray, labels: np.ndarray):
    hard = memb.argmax(axis=1)
    lab = [x for x in labels.tolist() if x is not None]
    anns = sorted(set(lab))
    C = np.zeros((6, len(anns)), dtype=np.int64)
    for i in range(6):
        lm = labels[hard == i]
        for j, a in enumerate(anns):
            C[i, j] = int(np.sum(lm == a))
    W = C.copy()
    pairs = {}
    for _ in range(6):
        i, j = np.unravel_index(int(np.argmax(W)), W.shape)
        pairs[str(i)] = anns[j]
        W[i, :] = -1
        W[:, j] = -1
    return pairs, C.tolist(), anns


def main() -> None:
    z = np.load(GRAPH_CACHE, allow_pickle=True)
    labels_rep = np.asarray(z["labels"])
    s1 = np.load(STEP1_NPZ, allow_pickle=True)
    memb = np.asarray(s1["memberships"], dtype=np.float32)
    macro_names = [str(x) for x in s1["macro_names"]]

    adata_sf = ad.read_h5ad(FATE75K)
    idx_prod = np.asarray(adata_sf.obs_names).astype(str)
    assert idx_prod.shape == (74_984,), f"obs_names shape {idx_prod.shape}"
    assert len(set(idx_prod.tolist())) == 74_984, "obs_names not unique"

    idx_ann, labels_h5ad = read_ann_obs(ANNOT_H5AD)
    assert idx_ann.shape == (74_984,)
    lut = {b: i for i, b in enumerate(idx_ann)}
    missing = [b for b in idx_prod if b not in lut]
    assert not missing, f"{len(missing)} production barcodes absent from annotated obs"
    src = np.array([lut[b] for b in idx_prod], dtype=np.int64)
    labels_aligned = labels_h5ad[src]

    # HARD GATE 1: cache rep labels must agree 180/180
    rep_mask = np.array([x is not None for x in labels_rep], dtype=bool)
    n_rep = int(rep_mask.sum())
    agree_rep = int(np.sum(labels_aligned[rep_mask] == labels_rep[rep_mask]))
    assert agree_rep == n_rep, f"rep-label agreement {agree_rep}/{n_rep}"

    # HARD GATE 2: full matching must agree with rep matching
    pairs_full, C_full, anns = match(memb, labels_aligned)
    pairs_rep, _, _ = match(memb, labels_rep)
    agreement = sum(1 for k in pairs_full if pairs_rep.get(k) == pairs_full[k])
    assert agreement >= 5, f"matching agreement {agreement}/6"

    s2b = np.load(STEP2B_NPZ, allow_pickle=True)
    F_cr_raw = np.asarray(s2b["F_cellrank_raw_by_macro"], dtype=np.float32)
    cr_cols = [str(c) for c in s2b["cr_macro_columns"]]
    F_sf = np.asarray(s2b["F_sfate"], dtype=np.float32)
    states = [str(s) for s in s2b["states"]]
    if set(states) != set(M3_ORDER):
        states = [s for s in M3_ORDER if s in set(states)]

    ann_of_macro = {name: pairs_full[str(i)] for i, name in enumerate(macro_names)}
    F_cr_ann = np.zeros((74_984, 6), dtype=np.float32)
    for j, sname in enumerate(states):
        cands = [k for k, cname in enumerate(cr_cols) if ann_of_macro.get(cname) == sname]
        assert len(cands) == 1, f"{sname}: {len(cands)} macro columns"
        F_cr_ann[:, j] = F_cr_raw[:, cands[0]]

    spearman = {s: float(spearmanr(F_cr_ann[:, j], F_sf[:, j]).statistic)
                for j, s in enumerate(states)}

    np.savez(OUT_NPZ, F_cellrank=F_cr_ann, F_sfate=F_sf,
             states=np.array(states),
             spearman=np.array([spearman[s] for s in states]),
             labels_aligned=labels_aligned)
    OUT_LOG.write_text(json.dumps({
        "rep_label_agreement": f"{agree_rep}/{n_rep}",
        "matching_agreement_6": agreement,
        "matching_full": ann_of_macro,
        "matching_rep_based": {k: pairs_rep[str(i)] for i, k in enumerate(macro_names)},
        "contingency_full": C_full,
        "contingency_annotations": anns,
        "per_state_spearman": spearman,
    }, ensure_ascii=False, indent=2))
    print(f"DONE rep_agree={agree_rep}/{n_rep} agreement={agreement}/6 "
          f"spearman={spearman}", flush=True)


if __name__ == "__main__":
    main()
