#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
4+5: GPCCA  +  ( cellrank + pygpcca)
- n in {2000, 5000, 10000}, k=30, method='krylov' (cellrank ; 'brandts'  Schur)
- GPCCA(n_states).fit() ->  RSS (tracemalloc + RSS )
-  5k fit  compute_absorption_probabilities() -> /
- : RSS  3000MB  ( 4GB)
"""
import csv, gc, os, sys, time, tracemalloc
import numpy as np
import psutil
import anndata as ad
import scanpy as sc
import cellrank as cr
from cellrank.kernels import ConnectivityKernel
from cellrank.estimators import GPCCA

PROC = psutil.Process()
RSS_LIMIT_MB = 3000.0

def rss_mb():
    return PROC.memory_info().rss / 1e6

def make_adata(n, d=32, n_clusters=8, seed=42, k=30):
    #  GMM:  + ,  ( GMM  GPCCA minChi )
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1.3, size=(n_clusters, d))  # ,  kNN
    weights = rng.uniform(0.5, 1.5, n_clusters); weights /= weights.sum()
    sigmas = rng.uniform(0.8, 1.5, n_clusters)
    labels = rng.choice(n_clusters, size=n, p=weights)
    X = (centers[labels] + rng.normal(0, 1.0, (n, d)) * sigmas[labels, None]).astype(np.float32)
    a = ad.AnnData(X=X); a.obsm["X_pca"] = X
    sc.pp.neighbors(a, n_neighbors=k, use_rep="X_pca")
    return a

def run_gpcca(n, n_states=5, n_schur=10):
    if rss_mb() > RSS_LIMIT_MB:
        print(f"[ABORT] RSS {rss_mb():.0f}MB ,  n={n}"); return None
    adata = make_adata(n)
    ck = ConnectivityKernel(adata).compute_transition_matrix()
    #  connectivity , brandts Schur  (cond(H)=inf);
    #  ( VelocityKernel  softmax ) ,  GPCCA
    import scipy.sparse as _sp
    from cellrank.kernels import PrecomputedKernel
    pt = adata.obsm["X_pca"][:, 0].astype(np.float64)  # PC1
    conn = adata.obsp["connectivities"].tocoo()
    dpt = pt[conn.col] - pt[conn.row]
    w = conn.data.astype(np.float64) * np.exp(dpt / max(np.std(dpt), 1e-9))
    T = _sp.csr_matrix((w, (conn.row, conn.col)), shape=conn.shape)
    inv = 1.0 / np.asarray(T.sum(1)).ravel()
    T = _sp.diags(inv) @ T
    pk = PrecomputedKernel(T, adata=adata)
    g = GPCCA(pk)
    gc.collect(); rss0 = rss_mb()
    tracemalloc.start()
    t0 = time.perf_counter()
    method_used = "krylov(request)"
    try:
        import slepc4py  # noqa
    except ImportError:
        method_used = "brandts(fallback,dense!)"  #  petsc4py/slepc4py  cellrank  brandts
    g.compute_schur(n_components=n_schur, method="krylov")
    t_schur = time.perf_counter() - t0
    rss_after_schur = rss_mb()
    t_macro = None
    for ns in [n_states, 4, 6, 3]:  #  optimize  n_states ,
        try:
            t0 = time.perf_counter()
            g.compute_macrostates(n_states=ns)
            t_macro = time.perf_counter() - t0
            n_states = ns
            break
        except ValueError as e:
            print(f"  [retry] n={n} n_states={ns} failed: {e}", flush=True)
    if t_macro is None:
        print(f"  [ABORT] n={n}  n_states ", flush=True)
        return None
    tm_peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    rss_total = rss_mb() - rss0
    sv = g.schur_vectors  # n x n_schur
    res = dict(exp=4, n=n, k=30, metric="gpcca_fit", method=method_used,
               n_states=n_states, n_schur=n_schur,
               schur_s=round(t_schur, 2), macro_s=round(t_macro, 3),
               tracemalloc_peak_mb=round(tm_peak, 1), rss_delta_mb=round(rss_total, 1),
               schur_vectors_shape=f"{sv.shape[0]}x{sv.shape[1]}",
               schur_vectors_mb=round(np.asarray(sv).nbytes / 1e6, 1))
    print(res, flush=True)
    out5 = None
    if n == 5000:
        out5 = run_absorption(g)
    del adata, ck, g
    gc.collect()
    return res, out5

def run_absorption(g):
    """5: compute_fate_probabilities (2.3.2 ;  compute_absorption_probabilities)"""
    g.predict_terminal_states()  #
    gc.collect(); rss0 = rss_mb()
    tracemalloc.start()
    t0 = time.perf_counter()
    g.compute_fate_probabilities()
    dt = time.perf_counter() - t0
    tm_peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    ap = g.fate_probabilities
    res = dict(exp=5, n=5000, k=30, metric="fate_probabilities",
               time_s=round(dt, 2), tracemalloc_peak_mb=round(tm_peak, 1),
               rss_delta_mb=round(rss_mb() - rss0, 1),
               ap_shape=str(ap.shape), n_terminal=len(g.terminal_states.cat.categories))
    print(res, flush=True)
    return res

CSV_PATH = "<out>/research/exp_t7/results_exp4.csv"

def append_rows(rows):
    allkeys = ["exp", "n", "k", "metric", "method", "n_states", "n_schur", "schur_s",
               "macro_s", "tracemalloc_peak_mb", "rss_delta_mb", "schur_vectors_shape",
               "schur_vectors_mb", "time_s", "ap_shape", "n_terminal"]
    exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=allkeys, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerows(rows)
    print(f"appended {len(rows)} rows -> {CSV_PATH}", flush=True)

def densify_probe(n=10000):
    """n=10000 brandts :  ( OOM kill );
    :  brandts  n=10000  4GB  OOM killer """
    adata = make_adata(n)
    ck = ConnectivityKernel(adata).compute_transition_matrix()
    T = ck.transition_matrix
    gc.collect(); rss0 = rss_mb()
    t0 = time.perf_counter()
    P_dense = np.asarray(T.toarray())  # brandts : densify
    dt = time.perf_counter() - t0
    rss_d = rss_mb() - rss0
    dense_mb = P_dense.nbytes / 1e6
    print(f"[densify probe] n={n} dense={dense_mb:.0f}MB densify={dt:.2f}s dRSS={rss_d:.0f}MB", flush=True)
    row = dict(exp=4, n=n, k=30, metric="brandts_densify_probe", method="brandts(fallback,dense!)",
               schur_s=round(dt, 2), tracemalloc_peak_mb=-1, rss_delta_mb=round(rss_d, 1),
               schur_vectors_shape=f"densified {n}x{n} float64 = {dense_mb:.0f}MB", schur_vectors_mb=round(dense_mb, 1))
    append_rows([row])
    del adata, ck, T, P_dense
    gc.collect()

def main():
    print("cellrank", cr.__version__, flush=True)
    for n in [2000, 5000]:
        r = run_gpcca(n)
        if r is None:
            continue
        rows = [r[0]] + ([r[1]] if r[1] else [])
        append_rows(rows)
    densify_probe(10000)
    # :  brandts GPCCA(n=10000)  ->  OOM killer  (4GB )
    print("done")

if __name__ == "__main__":
    main()
