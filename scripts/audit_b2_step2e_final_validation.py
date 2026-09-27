#!/usr/bin/env python
"""Layer B2 audit - Step 2e: final validation with on-disk ground truths.

Resolves the Step 2b/2d confounds:
  (a) sfate fate column order: authoritative production order = classes in
      output/c_layer_sfate_fate.npz (solver's own class order). Verified by
      correlating its fate B against fate_full_75k.h5ad's sfate_fate columns:
      the permutation must come out identity with corr ~ 1.0.
  (b) macro->annotation mapping: rep-based (centroid-nearest-30, Step 2)
      vs soft-membership mass matching; they must agree >=5/6. Also reports
      per-annotation rep concentration to judge whether the Proliferating
      pairing is meaningful (30 reps scattered -> weak pairing -> -0.70
      Spearman is an artifact of forced one-to-one matching).
  (c) final per-state Spearman with both sides on ground-truth labels.

Outputs: output/audit_b2_step2e_final.npz, output/audit_b2_step2e_log.md
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
CLAYER_NPZ = Path("output/c_layer_sfate_fate.npz")
FATE75K = Path("output/fate_full_75k.h5ad")
OUT_NPZ = Path("output/audit_b2_step2e_final.npz")
OUT_LOG = Path("output/audit_b2_step2e_log.md")
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


def greedy_one_to_one(W: np.ndarray):
    W = W.astype(np.float64).copy()
    pairs = {}
    for _ in range(min(W.shape)):
        i, j = np.unravel_index(int(np.argmax(W)), W.shape)
        pairs[(i, j)] = float(W[i, j])
        W[i, :] = -1
        W[:, j] = -1
    return pairs


def main() -> None:
    z = np.load(GRAPH_CACHE, allow_pickle=True)
    labels_rep = np.asarray(z["labels"])
    s1 = np.load(STEP1_NPZ, allow_pickle=True)
    memb = np.asarray(s1["memberships"], dtype=np.float32)

    cl = np.load(CLAYER_NPZ, allow_pickle=True)
    B = np.asarray(cl["fate"], dtype=np.float64)
    classes = [str(c) for c in cl["classes"]]

    s2b = np.load(STEP2B_NPZ, allow_pickle=True)
    F_sf = np.asarray(s2b["F_sfate"], dtype=np.float64)
    F_cr_raw = np.asarray(s2b["F_cellrank_raw_by_macro"], dtype=np.float64)
    cr_cols = [str(c) for c in s2b["cr_macro_columns"]]

    # (a) column-order ground truth: correlate independent production artifacts
    C = np.corrcoef(B.T, F_sf.T)[:6, 6:]
    perm = C.argmax(axis=1)
    matched = C.max(axis=1)
    identity = bool(np.array_equal(perm, np.arange(6)))
    assert matched.min() > 0.999, f"production fate matrices disagree: {matched}"
    assert len(set(perm.tolist())) == 6, f"permutation not bijective: {perm}"
    classes_eq_M3 = (classes == M3_ORDER)

    # full labels via barcodes (as validated 180/180 in Step 2d)
    idx_ann, labels_h5ad = read_ann_obs(ANNOT_H5AD)
    idx_prod = np.asarray(ad.read_h5ad(FATE75K).obs_names).astype(str)
    lut = {b: i for i, b in enumerate(idx_ann)}
    labels_full = labels_h5ad[np.array([lut[b] for b in idx_prod], dtype=np.int64)]

    anns = M3_ORDER
    hard = memb.argmax(axis=1)

    # (b1) rep-based matching (Step 2 estimator)
    Crep = np.zeros((6, 6))
    for i in range(6):
        lm = labels_rep[hard == i]
        for j, a in enumerate(anns):
            Crep[i, j] = np.sum(lm == a)
    pairs_rep = greedy_one_to_one(Crep)

    # (b2) soft-membership mass matching
    S = np.zeros((6, 6))
    for i in range(6):
        mass = memb[:, i].sum()
        for j, a in enumerate(anns):
            S[i, j] = memb[labels_full == a, i].sum() / mass
    pairs_soft = greedy_one_to_one(S)

    map_rep = {i: anns[j] for (i, j) in pairs_rep}
    map_soft = {i: anns[j] for (i, j) in pairs_soft}
    agree = sum(1 for i in range(6) if map_rep[i] == map_soft[i])

    # rep concentration per annotation (is each macro pairing meaningful?)
    conc = {}
    for a in anns:
        idx = np.nonzero(labels_rep == a)[0]
        conc[a] = np.bincount(hard[idx], minlength=6).tolist()

    if agree < 5:
        print(f"HALT: rep/soft agreement {agree}/6 - mapping unresolved, see log",
              flush=True)
        OUT_LOG.write_text(json.dumps({
            "halt_reason": f"rep/soft agreement {agree}/6",
            "map_rep": {str(k): v for k, v in map_rep.items()},
            "map_soft": {str(k): v for k, v in map_soft.items()},
            "soft_mass_matrix_rows_macro": np.round(S, 4).tolist(),
            "rep_contingency_rows_macro": Crep.tolist(),
            "annotations_order": anns,
            "rep_concentration": conc}, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    # (c) final spearman: mapping = rep-based (validated); sf columns in M3 order
    macro_names = [str(x) for x in s1["macro_names"]]
    ann_of_macro = {name: map_rep[i] for i, name in enumerate(macro_names)}
    F_cr_ann = np.zeros((74_984, 6))
    for j, s in enumerate(M3_ORDER):
        src = [k for k, c in enumerate(cr_cols) if ann_of_macro.get(c) == s]
        assert len(src) == 1
        F_cr_ann[:, j] = F_cr_raw[:, src[0]]
    # sf columns: file order == classes order == (checked) M3 order
    F_sf_m3 = F_sf if identity or classes_eq_M3 else F_sf[:, perm]
    if not (identity or classes_eq_M3):
        F_sf_m3 = F_sf[:, np.argsort(perm)]
    spearman = {s: float(spearmanr(F_cr_ann[:, j], F_sf_m3[:, j]).statistic)
                for j, s in enumerate(M3_ORDER)}

    np.savez(OUT_NPZ, F_cellrank=F_cr_ann, F_sfate=F_sf_m3,
             states=np.array(M3_ORDER),
             spearman=np.array([spearman[s] for s in M3_ORDER]),
             rep_concentration=np.array([conc[a] for a in M3_ORDER]))
    OUT_LOG.write_text(json.dumps({
        "classes_from_clayer": classes,
        "classes_eq_M3_order": bool(classes_eq_M3),
        "B_vs_Fsf_perm": perm.tolist(),
        "B_vs_Fsf_min_corr": float(matched.min()),
        "map_rep_based": {str(k): v for k, v in map_rep.items()},
        "map_soft_mass": {str(k): v for k, v in map_soft.items()},
        "rep_soft_agreement_6": agree,
        "rep_concentration_rows_M3_cols_macro01to5": conc,
        "per_state_spearman_FINAL": spearman,
    }, ensure_ascii=False, indent=2))
    print(f"DONE identity={identity} agree={agree}/6 spearman={spearman}", flush=True)


if __name__ == "__main__":
    main()	
