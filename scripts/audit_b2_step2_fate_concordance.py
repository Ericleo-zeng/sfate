#!/usr/bin/env python
"""Layer B2 audit - Step 2: annotation-matched terminals + fate concordance (75k).

Run from ~/research_storage/工作文件/Cellrank重构 (same CWD convention as Step 1).
Inputs : output/audit_b2_step1_macrostates.npz, output/c_layer_graph.npz,
         output/fate_full_75k.h5ad
Outputs: output/audit_b2_step2_cellrank_fate.npz, output/audit_b2_step2_log.md
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import anndata as ad
import numpy as np
import psutil
import scipy.sparse as sp
from scipy.stats import spearmanr

GRAPH_CACHE = Path("output/c_layer_graph.npz")
STEP1_NPZ = Path("output/audit_b2_step1_macrostates.npz")
FATE75K = Path("output/fate_full_75k.h5ad")
OUT_NPZ = Path("output/audit_b2_step2_cellrank_fate.npz")
OUT_LOG = Path("output/audit_b2_step2_log.md")


def _rss_sampler(stop: threading.Event, bucket: dict) -> None:
    proc = psutil.Process()
    while not stop.is_set():
        bucket["peak_mb"] = max(bucket["peak_mb"], proc.memory_info().rss / 1e6)
        stop.wait(0.25)


def match_macros_to_annotations(memb: np.ndarray, labels: np.ndarray):
    hard = memb.argmax(axis=1)
    n_macros = memb.shape[1]
    lab_list = [x for x in labels.tolist() if x is not None]
    n_none = int(labels.shape[0] - len(lab_list))
    print(f"[i] labels: {n_none} None excluded from matching", flush=True)
    anns = sorted(set(lab_list))
    C = np.zeros((n_macros, len(anns)), dtype=np.int64)
    for m in range(n_macros):
        lab_m = labels[hard == m]
        for j, a in enumerate(anns):
            C[m, j] = int(np.sum(lab_m == a))
    W = C.copy().astype(np.int64)
    pairs = {}
    for _ in range(min(n_macros, len(anns))):
        m, j = np.unravel_index(int(np.argmax(W)), W.shape)
        pairs[str(m)] = anns[j]
        W[m, :] = -1
        W[:, j] = -1
    purity = {str(m): float(C[m, anns.index(pairs[str(m)])]) / max(1, int((hard == m).sum()))
              for m in range(n_macros)}
    return pairs, C.tolist(), anns, purity


def main() -> None:
    t_start = time.perf_counter()
    stop = threading.Event()
    rss = {"peak_mb": 0.0}
    th = threading.Thread(target=_rss_sampler, args=(stop, rss), daemon=True)
    th.start()

    z = np.load(GRAPH_CACHE, allow_pickle=True)
    labels = np.asarray(z["labels"])
    P = sp.csr_matrix((z["P_data"], z["P_indices"], z["P_indptr"]),
                      shape=tuple(z["P_shape"])).astype(np.float32)

    s1 = np.load(STEP1_NPZ, allow_pickle=True)
    memb = np.asarray(s1["memberships"], dtype=np.float32)
    macro_names = [str(m) for m in s1["macro_names"]]
    assert memb.shape == (74_984, 6) and labels.shape == (74_984,)

    adata_sf = ad.read_h5ad(FATE75K)
    # auto-discover the (74984, 6) fate matrix: obsm key names differ across runs
    fate_keys = [k for k in adata_sf.obsm.keys()
                 if np.asarray(adata_sf.obsm[k]).shape == (74_984, 6)]
    assert len(fate_keys) == 1, f"ambiguous or missing fate keys: {fate_keys}"
    ls = adata_sf.obsm[fate_keys[0]]
    print(f"[i] sfate fate obsm key: {fate_keys[0]}", flush=True)
    F_sf = np.asarray(ls, dtype=np.float32)
    # column names: Lineage.names if preserved, else m3's ANNOTATED production order
    sf_names = ([str(s) for s in ls.names] if hasattr(ls, "names")
                else ["C1q_inflammatory", "DAM_like", "Homeostatic",
                      "IFN_responsive", "Proliferating", "PU1low_lymphoid"])
    assert F_sf.shape == (74_984, 6) and len(sf_names) == 6

    # 1) annotation matching + 2-3) explicit terminals + fate
    pairs, contingency, anns, purity = match_macros_to_annotations(memb, labels)
    assert len(pairs) == 6

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
    assert rs_dev < 1e-3, f"CellRank fate rows not normalized, dev={rs_dev:.2e}"

    # 4) align by annotation, per-state Spearman
    ann_of_macro = {mname: pairs[str(i)] for i, mname in enumerate(macro_names)}
    F_cr_ann = np.zeros((74_984, 6), dtype=np.float32)
    for j, s in enumerate(sf_names):
        src = [k for k, m in enumerate(cr_cols) if ann_of_macro.get(m) == s]
        assert len(src) == 1, f"annotation {s}: {len(src)} matching macro columns"
        F_cr_ann[:, j] = F_cr_raw[:, src[0]]

    rs_sf = float(np.abs(F_sf.sum(axis=1) - 1.0).max())
    spearman = {s: float(spearmanr(F_cr_ann[:, j], F_sf[:, j]).statistic)
                for j, s in enumerate(sf_names)}

    np.savez(OUT_NPZ, F_cellrank=F_cr_ann, F_sfate=F_sf,
             states=np.array(sf_names),
             spearman=np.array([spearman[s] for s in sf_names]))
    wall = time.perf_counter() - t_start
    stop.set(); th.join(timeout=2.0)
    OUT_LOG.write_text(json.dumps({
        "annotation_of_macro": ann_of_macro,
        "contingency": contingency, "contingency_annotations": anns,
        "matching_purity": purity,
        "fate_row_sum_dev_cellrank": rs_dev, "fate_row_sum_dev_sfate": rs_sf,
        "per_state_spearman": spearman,
        "peak_rss_gib": round(rss["peak_mb"] / 1024, 2), "wall_s": round(wall, 1),
    }, ensure_ascii=False, indent=2))
    print(f"DONE spearman={spearman} peak={rss['peak_mb']/1024:.2f}GiB wall={wall:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
