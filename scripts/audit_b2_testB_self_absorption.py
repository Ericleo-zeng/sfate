#!/usr/bin/env python
"""Layer B2 - Test B: per-state self-absorption ranking (transient-vs-terminal).

Question: is Proliferating a low-commitment (transient) compartment in sfate's
own production landscape? Diagnostic: mean per-cell absorption toward state a,
computed within cells OF state a (self-absorption diagonal block), ranked over
the six states; plus the full 6x6 block-mean matrix (rows = cell's own state,
cols = absorption target) for context.

Inputs : output/audit_b2_step3b_final_v2.npz (F_sfate, states, M3 order locked)
         adata_annotated.h5ad (microglia_state via barcode alignment)
Output : output/audit_b2_testB_log.md
"""
from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np

ANNOT_H5AD = Path("/mnt/t9/datasets/ad_microglia_fate_landscape_output/06_annotation/adata_annotated.h5ad")
FATE75K = Path("output/fate_full_75k.h5ad")
STEP3B_NPZ = Path("output/audit_b2_step3b_final_v2.npz")
OUT_LOG = Path("output/audit_b2_testB_log.md")
M3_ORDER = ["C1q_inflammatory", "DAM_like", "Homeostatic",
            "IFN_responsive", "Proliferating", "PU1low_lymphoid"]


def main() -> None:
    z = np.load(STEP3B_NPZ, allow_pickle=True)
    states = [str(s) for s in z["states"]]
    assert states == M3_ORDER, f"states {states}"
    F = np.asarray(z["F_sfate"], dtype=np.float64)          # (74984, 6)

    idx_prod = np.asarray(ad.read_h5ad(FATE75K).obs_names).astype(str)
    with h5py.File(ANNOT_H5AD, "r") as f:
        gobs = f["obs"]
        idx_ann = np.asarray(gobs[gobs.attrs["_index"]][...]).astype(str)
        g = gobs["microglia_state"]
        cats = [c.decode() if isinstance(c, bytes) else str(c)
                for c in g["categories"][:]]
        codes = np.asarray(g["codes"][:])
    lab = np.empty(len(codes), dtype=object)
    m = codes >= 0
    lab[m] = np.array(cats, dtype=object)[codes[m]]
    lab[~m] = None
    lut = {b: i for i, b in enumerate(idx_ann)}
    labels = lab[np.array([lut[b] for b in idx_prod], dtype=np.int64)]

    # 6x6 block means: row = cell's own state, col = absorption target
    block = np.zeros((6, 6))
    self_abs = {}
    for i, a in enumerate(M3_ORDER):
        rows_a = labels == a
        block[i] = F[rows_a].mean(axis=0)
        self_abs[a] = float(block[i, i])
    ranking = sorted(self_abs.items(), key=lambda kv: kv[1], reverse=True)

    # global mean absorption per arm (context)
    global_mean = F.mean(axis=0)

    OUT_LOG.write_text(json.dumps({
        "self_absorption_mean_F_within_state": self_abs,
        "self_absorption_ranking_desc": ranking,
        "block_means_rows_ownstate_cols_target": np.round(block, 4).tolist(),
        "global_mean_absorption_per_arm": {a: float(global_mean[j])
                                           for j, a in enumerate(M3_ORDER)},
        "proliferating_fraction_of_cells": float(np.mean(labels == "Proliferating")),
    }, ensure_ascii=False, indent=2))
    print(f"DONE self-absorption ranking (desc): {ranking}", flush=True)


if __name__ == "__main__":
    main()
