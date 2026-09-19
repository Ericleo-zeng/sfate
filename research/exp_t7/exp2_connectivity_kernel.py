#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2+3: ConnectivityKernel  +  ( cellrank/scanpy)
- anndata  (X=32  latent), scanpy.pp.neighbors(n_neighbors=k)
- ConnectivityKernel(adata).compute_transition_matrix(density_normalize=True)
- n in {5000, 20000, 50000}, k in {15, 30, 50}
- :  RSS / tracemalloc /  nnz / obsp keys / float32->float64
- 3:  << n*n*8 bytes, " n×n "
"""
import csv, gc, time, tracemalloc
import numpy as np
import psutil
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
import cellrank as cr
from cellrank.kernels import ConnectivityKernel

PROC = psutil.Process()

def rss_mb():
    return PROC.memory_info().rss / 1e6

def make_adata(n, d=32, n_clusters=8, seed=42):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3.0, size=(n_clusters, d))
    X = (centers[rng.integers(0, n_clusters, n)] + rng.normal(0, 1.0, (n, d))).astype(np.float32)
    a = ad.AnnData(X=X)
    a.obsm["X_pca"] = X  #  latent ,  PCA
    return a

def run_one(n, k):
    adata = make_adata(n)
    gc.collect(); rss0 = rss_mb()
    tracemalloc.start()

    t0 = time.perf_counter()
    sc.pp.neighbors(adata, n_neighbors=k, use_rep="X_pca")
    t_neigh = time.perf_counter() - t0
    conn = adata.obsp["connectivities"]
    conn_dtype = str(conn.dtype); conn_nnz = conn.nnz
    obsp_after_neighbors = sorted(adata.obsp.keys())

    t0 = time.perf_counter()
    ck = ConnectivityKernel(adata)
    ck.compute_transition_matrix(density_normalize=True)
    t_kernel = time.perf_counter() - t0

    tm_peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    rss_peak_delta = rss_mb() - rss0
    T = ck.transition_matrix
    obsp_final = sorted(adata.obsp.keys())
    t_mem = (T.data.nbytes + T.indices.nbytes + T.indptr.nbytes) / 1e6 if sp.issparse(T) else T.nbytes / 1e6
    dense_n2_mb = n * n * 8 / 1e6  #  float64  n×n
    row_sums = np.asarray(T.sum(1)).ravel() if sp.issparse(T) else T.sum(1)
    res = dict(exp=2, n=n, k=k, metric="connectivity_kernel",
               conn_dtype=conn_dtype, conn_nnz=conn_nnz,
               T_format=type(T).__name__, T_nnz=int(T.nnz) if sp.issparse(T) else -1,
               T_dtype=str(T.dtype), T_mem_mb=round(t_mem, 1),
               obsp_after_neighbors="|".join(obsp_after_neighbors),
               obsp_final="|".join(obsp_final),
               neighbors_s=round(t_neigh, 2), kernel_s=round(t_kernel, 3),
               tracemalloc_peak_mb=round(tm_peak, 1), rss_delta_mb=round(rss_peak_delta, 1),
               dense_n2_mb=round(dense_n2_mb, 1),
               rowsum_ok=bool(np.allclose(row_sums, 1.0, atol=1e-8)))
    print(res)
    del adata, ck, T, conn
    gc.collect()
    return res

def main():
    print("cellrank", cr.__version__, "| scanpy", sc.__version__, "| anndata", ad.__version__)
    out = []
    for n in [5000, 20000, 50000]:
        for k in [15, 30, 50]:
            out.append(run_one(n, k))
    with open("<out>/research/exp_t7/results_exp2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print("saved results_exp2.csv")

if __name__ == "__main__":
    main()
