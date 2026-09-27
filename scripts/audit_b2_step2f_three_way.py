#!/usr/bin/env python
"""Layer B2 audit - Step 2f: three-estimator face-off for the macro mapping.

Estimators for macro(i) -> annotation(a):
  E1 rep-argmax      (Step 2; 180 centroid-nearest reps)
  E2 soft-membership mass (Step 2e)
  E3 fate-profile    (new, independent): Spearman between CellRank fate column
                     toward macro i and sfate absorption column toward
                     annotation a - which arm pairs with which, directly.
Plus structure diagnostics for the DAM/Proliferating entanglement:
  full hard contingency over all cells, per-annotation mean membership,
  and rep-vs-bulk mean membership for DAM and Proliferating.

Outputs: output/audit_b2_step2f_log.md (no npz; judgment step)
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
OUT_LOG = Path("output/audit_b2_step2f_log.md")
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


def greedy(W: np.ndarray):
    W = W.astype(np.float64).copy()
    pairs = {}
    for _ in range(min(W.shape)):
        i, j = np.unravel_index(int(np.argmax(W)), W.shape)
        pairs[i] = j
        W[i, :] = -1
        W[:, j] = -1
    return pairs


def main() -> None:
    z = np.load(GRAPH_CACHE, allow_pickle=True)
    labels_rep = np.asarray(z["labels"])
    s1 = np.load(STEP1_NPZ, allow_pickle=True)
    memb = np.asarray(s1["memberships"], dtype=np.float64)
    s2b = np.load(STEP2B_NPZ, allow_pickle=True)
    F_cr_raw = np.asarray(s2b["F_cellrank_raw_by_macro"], dtype=np.float64)
    F_sf = np.asarray(s2b["F_sfate"], dtype=np.float64)

    idx_ann, labels_h5ad = read_ann_obs(ANNOT_H5AD)
    idx_prod = np.asarray(ad.read_h5ad(FATE75K).obs_names).astype(str)
    lut = {b: i for i, b in enumerate(idx_ann)}
    labels_full = labels_h5ad[np.array([lut[b] for b in idx_prod], dtype=np.int64)]

    hard = memb.argmax(axis=1)
    anns = M3_ORDER

    # full hard contingency (all cells)
    C_hard = np.zeros((6, 6))
    for i in range(6):
        lm = labels_full[hard == i]
        for j, a in enumerate(anns):
            C_hard[i, j] = np.sum(lm == a)

    # per-annotation mean membership (bulk)
    mean_memb_bulk = np.zeros((6, 6))
    mean_memb_rep = np.zeros((6, 6))
    for j, a in enumerate(anns):
        mean_memb_bulk[j] = memb[labels_full == a].mean(axis=0)
        mean_memb_rep[j] = memb[labels_rep == a].mean(axis=0)

    # E3: fate-profile similarity
    S_fate = np.zeros((6, 6))
    for i in range(6):
        for j, a in enumerate(anns):
            S_fate[i, j] = spearmanr(F_cr_raw[:, i], F_sf[:, j]).statistic

    e1 = greedy(np.zeros((6, 6)))  # placeholder, recomputed below properly
    # E1 from rep contingency
    C_rep = np.zeros((6, 6))
    for i in range(6):
        lm = labels_rep[hard == i]
        for j, a in enumerate(anns):
            C_rep[i, j] = np.sum(lm == a)
    e1 = greedy(C_rep)
    e2 = greedy(np.array([[memb[labels_full == a, i].sum() / memb[:, i].sum()
                           for a in anns] for i in range(6)]))
    e3 = greedy(S_fate)

    maps = {"E1_rep": {str(i): anns[j] for i, j in e1.items()},
            "E2_soft": {str(i): anns[j] for i, j in e2.items()},
            "E3_fate": {str(i): anns[j] for i, j in e3.items()}}
    consensus = {str(i): [maps[e][str(i)] for e in maps] for i in range(6)}

    OUT_LOG.write_text(json.dumps({
        "consensus_per_macro": consensus,
        "C_hard_all_cells_rows_macro": C_hard.tolist(),
        "mean_membership_bulk_rows_ann": np.round(mean_memb_bulk, 4).tolist(),
        "mean_membership_rep_rows_ann": np.round(mean_memb_rep, 4).tolist(),
        "S_fate_profile_rows_macro": np.round(S_fate, 4).tolist(),
        "annotations_order": anns,
    }, ensure_ascii=False, indent=2))
    print("DONE -> output/audit_b2_step2f_log.md", flush=True)


if __name__ == "__main__":
    main()
