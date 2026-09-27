#!/usr/bin/env python
"""Layer B2 FINAL: natural-order memberships + Hungarian-optimal arm matching.

Fixes Step 3's two flaws:
  (a) memberships extracted IMMEDIATELY after compute_macrostates, BEFORE
      set_terminal_states can reorder lineage columns;
  (b) arm correspondence = scipy Hungarian MAXIMIZATION on S_fate (fate
      profiles are bit-identical across runs; sfate columns are locked) —
      no membership-ordering dependence at all.
Cross-check: e1_rep on natural-order memberships vs Hungarian assignment.
Outputs: output/audit_b2_step3b_final.npz, output/audit_b2_step3b_log.md
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import psutil
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

ANNOT_H5AD = Path("/mnt/t9/datasets/ad_microglia_fate_landscape_output/06_annotation/adata_annotated.h5ad")
GRAPH_CACHE = Path("output/c_layer_graph.npz")
FATE75K = Path("output/fate_full_75k.h5ad")
OUT_NPZ = Path("output/audit_b2_step3b_final.npz")
OUT_LOG = Path("output/audit_b2_step3b_log.md")
M3_ORDER = ["C1q_inflammatory", "DAM_like", "Homeostatic",
            "IFN_responsive", "Proliferating", "PU1low_lymphoid"]


def _rss(stop: threading.Event, bucket: dict) -> None:
    proc = psutil.Process()
    while not stop.is_set():
        bucket["peak_mb"] = max(bucket["peak_mb"], proc.memory_info().rss / 1e6)
        stop.wait(0.25)


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
    out = {}
    for _ in range(min(W.shape)):
        i, j = np.unravel_index(int(np.argmax(W)), W.shape)
        out[i] = j
        W[i, :] = -1
        W[:, j] = -1
    return out


def main() -> None:
    t0 = time.perf_counter()
    stop = threading.Event()
    rss = {"peak_mb": 0.0}
    th = threading.Thread(target=_rss, args=(stop, rss), daemon=True)
    th.start()

    z = np.load(GRAPH_CACHE, allow_pickle=True)
    labels_rep = np.asarray(z["labels"])
    P = sp.csr_matrix((z["P_data"], z["P_indices"], z["P_indptr"]),
                      shape=tuple(z["P_shape"])).astype(np.float32)
    idx_ann, labels_h5ad = read_ann_obs(ANNOT_H5AD)
    idx_prod = np.asarray(ad.read_h5ad(FATE75K).obs_names).astype(str)
    lut = {b: i for i, b in enumerate(idx_ann)}
    labels_full = labels_h5ad[np.array([lut[b] for b in idx_prod], dtype=np.int64)]

    a_sf = ad.read_h5ad(FATE75K)
    fk = [k for k in a_sf.obsm.keys() if np.asarray(a_sf.obsm[k]).shape == (74_984, 6)]
    F_sf = np.asarray(a_sf.obsm[fk[0]], dtype=np.float64)

    import cellrank as cr
    from cellrank.estimators import GPCCA
    pk = cr.kernels.PrecomputedKernel(P)
    g = GPCCA(pk)
    g.compute_schur(n_components=20, method="krylov")
    g.compute_macrostates(n_states=6, cluster_key=None)

    # (a) memberships in NATURAL order, BEFORE set_terminal_states
    memb = np.asarray(g.macrostates_memberships, dtype=np.float64)
    macro_names = [str(m) for m in g.macrostates_memberships.names]

    g.set_terminal_states(states=macro_names)
    g.compute_fate_probabilities(solver="gmres", use_petsc=False, preconditioner="ilu")

    fp = g.fate_probabilities
    F_cr = np.asarray(fp, dtype=np.float64)
    fate_names = [str(s) for s in fp.names]
    assert fate_names == macro_names, f"fate/misc name mismatch: {fate_names} vs {macro_names}"
    name_to_idx = {name: i for i, name in enumerate(macro_names)}

    hard = memb.argmax(axis=1)

    # e1 on natural-order memberships
    C_rep = np.zeros((6, 6))
    for i in range(6):
        lm = labels_rep[hard == i]
        for j, a in enumerate(M3_ORDER):
            C_rep[i, j] = np.sum(lm == a)
    e1 = {i: M3_ORDER[j] for i, j in greedy(C_rep).items()}

    # e2 on natural-order memberships
    S_mass = np.zeros((6, 6))
    for i in range(6):
        mass = memb[:, i].sum()
        for j, a in enumerate(M3_ORDER):
            S_mass[i, j] = memb[labels_full == a, i].sum() / mass
    e2 = {i: M3_ORDER[j] for i, j in greedy(S_mass).items()}

    # S_fate + HUNGARIAN (primary)
    S_fate = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            S_fate[i, j] = spearmanr(F_cr[:, i], F_sf[:, j]).statistic
    r_idx, c_idx = linear_sum_assignment(-S_fate)   # maximize
    hung = {int(r): M3_ORDER[int(c)] for r, c in zip(r_idx, c_idx)}

    agree_e1_hung = sum(1 for i in range(6) if e1[i] == hung[i])

    F_cr_ann = np.zeros((74_984, 6))
    for j, a in enumerate(M3_ORDER):
        src = [k for k, name in enumerate(fate_names) if hung[name_to_idx[name]] == a]
        assert len(src) == 1
        F_cr_ann[:, j] = F_cr[:, src[0]]
    spearman = {a: float(spearmanr(F_cr_ann[:, j], F_sf[:, j]).statistic)
                for j, a in enumerate(M3_ORDER)}
    F_sf_out = F_sf.copy(); F_sf_out[:, [4, 5]] = F_sf_out[:, [5, 4]]
    np.savez(OUT_NPZ, F_cellrank=F_cr_ann, F_sfate=F_sf_out, states=np.array(M3_ORDER),
             spearman=np.array([spearman[a] for a in M3_ORDER]),
             memberships_natural=memb, macro_names=np.array(macro_names),
             F_cr_raw=F_cr, fate_names=np.array(fate_names), S_fate=S_fate)
    wall = time.perf_counter() - t0
    stop.set(); th.join(timeout=2.0)
    OUT_LOG.write_text(json.dumps({
        "rep_concentration_natural": {a: np.bincount(
            hard[np.nonzero(labels_rep == a)[0]], minlength=6).tolist()
            for a in M3_ORDER},
        "e1_rep_natural": {str(k): v for k, v in e1.items()},
        "e2_soft_natural": {str(k): v for k, v in e2.items()},
        "hungarian_primary": {str(k): v for k, v in hung.items()},
        "e1_hungarian_agreement_6": agree_e1_hung,
        "S_fate": np.round(S_fate, 4).tolist(),
        "spearman_FINAL_HUNGARIAN": spearman,
        "peak_rss_gib": round(rss["peak_mb"] / 1024, 2), "wall_s": round(wall, 1),
    }, ensure_ascii=False, indent=2))
    print(f"DONE agree_e1_hung={agree_e1_hung}/6 spearman={spearman} "
          f"peak={rss['peak_mb']/1024:.2f}GiB wall={wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
