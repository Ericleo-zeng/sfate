#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1: kNN  nnz  (T7 ,  cellrank)
-  latent: 8 , 32  float32
- n in {5000, 10000, 20000, 50000}, k in {15, 30, 50}
- sklearn NearestNeighbors  kNN  -> CSR
-  nnz ~ n*k ;  /  RSS / tracemalloc
"""
import csv, gc, time, tracemalloc
import numpy as np
import psutil
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

PROC = psutil.Process()

def rss_mb():
    return PROC.memory_info().rss / 1e6

def make_latent(n, d=32, n_clusters=8, seed=42):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3.0, size=(n_clusters, d))
    labels = rng.integers(0, n_clusters, size=n)
    X = (centers[labels] + rng.normal(0, 1.0, size=(n, d))).astype(np.float32)
    return X, labels

def build_transition(X, k):
    """kNN  ->  ->  CSR ;  (T, nnz_dist_graph, )"""
    t0 = time.perf_counter()
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X)
    dist, idx = nn.kneighbors(X)
    #  (0)
    dist, idx = dist[:, 1:], idx[:, 1:]
    n = X.shape[0]
    # : sigma =  k  ( scanpy , )
    sigma = np.median(dist[:, -1])
    w = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    rows = np.repeat(np.arange(n), k)
    W = sp.csr_matrix((w.ravel(), (rows, idx.ravel())), shape=(n, n))
    #  (max) ,  scanpy/cellrank
    W = W.maximum(W.T).tocsr()
    rowsum = np.asarray(W.sum(1)).ravel()
    inv = np.divide(1.0, rowsum, out=np.zeros_like(rowsum), where=rowsum > 0)
    T = sp.diags(inv) @ W
    dt = time.perf_counter() - t0
    return T.tocsr(), n * k, dt

def main():
    rows_out = []
    for n in [5000, 10000, 20000, 50000]:
        X, _ = make_latent(n)
        for k in [15, 30, 50]:
            gc.collect()
            rss0 = rss_mb()
            tracemalloc.start()
            T, nnz_direct, dt = build_transition(X, k)
            tm_peak = tracemalloc.get_traced_memory()[1] / 1e6
            tracemalloc.stop()
            gc.collect()
            rss_peak_delta = rss_mb() - rss0
            nnz = T.nnz
            #
            t_mem = (T.data.nbytes + T.indices.nbytes + T.indptr.nbytes) / 1e6
            ratio = nnz / (n * k)
            print(f"n={n} k={k} nnz={nnz} nnz/(n*k)={ratio:.4f} "
                  f"build={dt:.2f}s tracemalloc={tm_peak:.1f}MB dRSS={rss_peak_delta:.1f}MB T_mem={t_mem:.1f}MB")
            rows_out.append(dict(exp=1, n=n, k=k, metric="knn_transition",
                                 nnz=nnz, nnz_per_nk=ratio, build_s=round(dt, 3),
                                 tracemalloc_peak_mb=round(tm_peak, 1),
                                 rss_delta_mb=round(rss_peak_delta, 1),
                                 T_mem_mb=round(t_mem, 1)))
            del T
        del X
        gc.collect()
    with open("<out>/research/exp_t7/results_exp1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)
    print("saved results_exp1.csv")

if __name__ == "__main__":
    main()
