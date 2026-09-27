#!/usr/bin/env python
"""Layer B2 audit - Step 2b: macro<->annotation matching on FULL 74,984 labels.

Step 2's matching used only the 180 representative cells (cache labels are None
elsewhere). This script reloads the full microglia_state annotation from the
source annotated h5ad (h5py obs-only read: avoids the +6.9GB backed-open RSS
documented in the paper), rebuilds the matching on all cells, re-runs the
CellRank fate computation (15s), and re-computes per-state Spearman.
The 180-rep matching is kept as a cross-check: agreement confirms both the
matching and the cell-ordering assumption.

Inputs : output/audit_b2_step1_macrostates.npz, output/c_layer_graph.npz,
         output/fate_full_75k.h5ad, ANNOT_H5AD (below)
Outputs: output/audit_b2_step2b_cellrank_fate.npz, output/audit_b2_step2b_log.md
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
from scipy.stats import spearmanr

# TODO: 填实际路径（与 c_layer_validation.py / m3 使用的同一个 adata_annotated.h5ad）
ANNOT_H5AD = Path("/mnt/t9/datasets/ad_microglia_fate_landscape_output/06_annotation/adata_annotated.h5ad")
ANNOT_OBS_KEY = "microglia_state"

EXPECTED_COUNTS = {"Homeostatic": 27953, "IFN_responsive": 20303,
                   "C1q_inflammatory": 13575, "DAM_like": 6796,
                   "PU1low_lymphoid": 5560, "Proliferating": 797}

GRAPH_CACHE = Path("output/c_layer_graph.npz")
STEP1_NPZ = Path("output/audit_b2_step1_macrostates.npz")
FATE75K = Path("output/fate_full_75k.h5ad")
OUT_NPZ = Path("output/audit_b2_step2b_cellrank_fate.npz")
OUT_LOG = Path("output/audit_b2_step2b_log.md")


def _rss_sampler(stop: threading.Event, bucket: dict) -> None:
    proc = psutil.Process()
    while not stop.is_set():
        bucket["peak_mb"] = max(bucket["peak_mb"], proc.memory_info().rss / 1e6)
        stop.wait(0.25)


def read_obs_categorical(path: Path, key: str) -> np.ndarray:
    """h5py obs-only categorical read; returns object array with None for NA."""
    with h5py.File(path, "r") as f:
        if key not in f["obs"]:
            raise KeyError(f"{key} not in obs; available: {list(f['obs'].keys())}")
        g = f["obs"][key]
        cats = [c.decode() if isinstance(c, bytes) else str(c) for c in g["categories"][:]]
        codes = g["codes"][:]
    out = np.empty(len(codes), dtype=object)
    mask = codes >= 0
    out[mask] = np.array(cats, dtype=object)[codes[mask]]
    out[~mask] = None
    return out


def match(memb: np.ndarray, labels: np.ndarray):
    lab = [x for x in labels.tolist() if x is not None]
    hard = memb.argmax(axis=1)
    anns = sorted(set(lab))
    C = np.zeros((memb.shape[1], len(anns)), dtype=np.int64)
    for m in range(memb.shape[1]):
        lm = labels[hard == m]
        for j, a in enumerate(anns):
            C[m, j] = int(np.sum(lm == a))
    W = C.copy()
    pairs = {}
    for _ in range(min(memb.shape[1], len(anns))):
        m, j = np.unravel_index(int(np.argmax(W)), W.shape)
        pairs[str(m)] = anns[j]
        W[m, :] = -1
        W[:, j] = -1
    return pairs, C.tolist(), anns


def main() -> None:
    t0 = time.perf_counter()
    stop = threading.Event()
    rss = {"peak_mb": 0.0}
    th = threading.Thread(target=_rss_sampler, args=(stop, rss), daemon=True)
    th.start()

    z = np.load(GRAPH_CACHE, allow_pickle=True)
    labels_rep = np.asarray(z["labels"])                      # 180 reps, else None
    P = sp.csr_matrix((z["P_data"], z["P_indices"], z["P_indptr"]),
                      shape=tuple(z["P_shape"])).astype(np.float32)
    s1 = np.load(STEP1_NPZ, allow_pickle=True)
    memb = np.asarray(s1["memberships"], dtype=np.float32)
    macro_names = [str(m) for m in s1["macro_names"]]

    labels_full = read_obs_categorical(ANNOT_H5AD, ANNOT_OBS_KEY)
    assert labels_full.shape == (74_984,), f"full labels shape {labels_full.shape}"
    counts = {a: int(np.sum(labels_full == a)) for a in EXPECTED_COUNTS}
    assert counts == EXPECTED_COUNTS, f"annotation counts mismatch: {counts}"

    # both matchings + agreement cross-check
    pairs_full, C_full, anns_full = match(memb, labels_full)
    pairs_rep, _, _ = match(memb, labels_rep)
    agreement = sum(1 for m in pairs_full if pairs_rep.get(m) == pairs_full[m])

    adata_sf = ad.read_h5ad(FATE75K)
    fate_keys = [k for k in adata_sf.obsm.keys()
                 if np.asarray(adata_sf.obsm[k]).shape == (74_984, 6)]
    assert len(fate_keys) == 1
    ls = adata_sf.obsm[fate_keys[0]]
    F_sf = np.asarray(ls, dtype=np.float32)
    sf_names = ([str(s) for s in ls.names] if hasattr(ls, "names")
                else list(EXPECTED_COUNTS.keys()))
    if set(sf_names) != set(EXPECTED_COUNTS):  # fallback order wrong -> use intersection order
        sf_names = [s for s in ["C1q_inflammatory", "DAM_like", "Homeostatic",
                                "IFN_responsive", "Proliferating", "PU1low_lymphoid"]
                    if s in set(sf_names)]

    import cellrank as cr
    from cellrank.estimators import GPCCA
    pk = cr.kernels.PrecomputedKernel(P)
    g = GPCCA(pk)
    g.compute_schur(n_components=20, method="krylov")
    g.compute_macrostates(n_states=6, cluster_key=None)
    g.set_terminal_states(states=macro_names)
    g.compute_fate_probabilities(solver="gmres", use_petsc=False, preconditioner="ilu")

    fp = g.fate_probabilities
    cr_cols = [str(s) for s in fp.names]
    F_cr_raw = np.asarray(fp, dtype=np.float32)
    rs_dev = float(np.abs(F_cr_raw.sum(axis=1) - 1.0).max())
    assert rs_dev < 1e-3

    ann_of_macro = {mname: pairs_full[str(i)] for i, mname in enumerate(macro_names)}
    F_cr_ann = np.zeros((74_984, 6), dtype=np.float32)
    for j, s in enumerate(sf_names):
        src = [k for k, m in enumerate(cr_cols) if ann_of_macro.get(m) == s]
        assert len(src) == 1
        F_cr_ann[:, j] = F_cr_raw[:, src[0]]

    spearman = {s: float(spearmanr(F_cr_ann[:, j], F_sf[:, j]).statistic)
                for j, s in enumerate(sf_names)}

    np.savez(OUT_NPZ, F_cellrank=F_cr_ann, F_cellrank_raw_by_macro=F_cr_raw,
             cr_macro_columns=np.array(cr_cols),
             F_sfate=F_sf, states=np.array(sf_names),
             spearman=np.array([spearman[s] for s in sf_names]))
    wall = time.perf_counter() - t0
    stop.set(); th.join(timeout=2.0)
    OUT_LOG.write_text(json.dumps({
        "annotation_counts_check": counts,
        "matching_full": ann_of_macro,
        "matching_rep_based": {k: pairs_rep[str(i)] for i, k in enumerate(macro_names)},
        "matching_agreement_6": agreement,
        "contingency_full_hard_assignment": C_full,
        "contingency_annotations": anns_full,
        "fate_row_sum_dev_cellrank": rs_dev,
        "per_state_spearman": spearman,
        "peak_rss_gib": round(rss["peak_mb"] / 1024, 2),
        "wall_s": round(wall, 1),
    }, ensure_ascii=False, indent=2))
    print(f"DONE agreement={agreement}/6 spearman={spearman} "
          f"peak={rss['peak_mb']/1024:.2f}GiB wall={wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
