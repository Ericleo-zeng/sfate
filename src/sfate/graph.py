"""graph.pylatent →  kNN → CSR scalable-fate M1

 validation-protocol I7  AGENTS.md
-  n×n  CSR/COO
-  scipy.sparse  toarray()/todense()tests/  monkeypatch  + grep
-  CSRfloat32

build_transition_graphkNN  →  →  →
absorb_statessplit_transient_absorbingM2
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as sp

MAX_LATENT_DIM = 256


def _validate_latent(latent) -> np.ndarray:
    """ (n, d)  latent n×n """
    X = np.asarray(latent)
    if X.ndim != 2:
        raise ValueError(f"latent  2  (n, d) ndim={X.ndim}")
    n, d = X.shape
    if d > MAX_LATENT_DIM:
        raise ValueError(
            f"latent  d={d}  {MAX_LATENT_DIM}"
        )
    if n == 0:
        raise ValueError("latent n=0")
    if sp.issparse(X):
        raise ValueError("latent  (n, d) ")
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError(f"latent dtype  {X.dtype}")
    return np.ascontiguousarray(X, dtype=np.float32)


def _query_knn(
    X: np.ndarray, k: int, backend: str, random_state: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """ (indices (n, k), distances (n, k))

     k+1  0  k
    """
    n = X.shape[0]
    if k < 1:
        raise ValueError(f"k  ≥1 {k}")
    if k >= n:
        raise ValueError(f"k={k}  < n={n}")

    if backend == "pynndescent":
        from pynndescent import NNDescent

        index = NNDescent(
            X, n_neighbors=k + 1, metric="euclidean", random_state=random_state
        )
        indices, distances = index.query(X, k=k + 1)
        # query  0  0
        return indices[:, 1:].copy(), distances[:, 1:].copy()
    if backend == "sklearn":
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
        nn.fit(X)
        distances, indices = nn.kneighbors(X)
        return indices[:, 1:].copy(), distances[:, 1:].copy()
    raise ValueError(f" knn_backend: {backend!r} pynndescent / sklearn")


def _edge_weights(
    indices: np.ndarray, distances: np.ndarray, kernel: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ COO  (row, col, data) kernel

    - connectivity:  1T7 exp2
    - gaussian: exp(-(d/d_k)^2)d_k =  k  T7 exp1
    """
    n, k = indices.shape
    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    cols = indices.ravel().astype(np.int64)
    dist = distances.astype(np.float64).ravel()

    if kernel == "connectivity":
        data = np.ones(rows.shape[0], dtype=np.float64)
    elif kernel == "gaussian":
        #  k  d_kd_k=0 1
        dk = distances[:, -1].astype(np.float64)
        dk_safe = np.where(dk > 0, dk, 1.0)
        scale = np.repeat(dk_safe, k)
        data = np.exp(-((dist / scale) ** 2))
        data = np.where(np.repeat(dk > 0, k), data, 1.0)
    else:
        raise ValueError(f" kernel: {kernel!r} gaussian / connectivity")

    return rows, cols, data


def _symmetrize_max(rows: np.ndarray, cols: np.ndarray, data: np.ndarray, n: int) -> sp.csr_matrix:
    """(max) (min(i,j), max(i,j))

    np.unique + reduceat n×n  CSR
    """
    key_rows = np.minimum(rows, cols).astype(np.int64)
    key_cols = np.maximum(rows, cols).astype(np.int64)
    key = key_rows * n + key_cols  # n≤1e6  n²≤1e12int64
    order = np.argsort(key, kind="stable")
    d_sorted = data[order]
    uniq, first = np.unique(key[order], return_index=True)
    seg_max = np.maximum.reduceat(d_sorted, first)
    kr = uniq // n
    kc = uniq % n
    out_rows = np.concatenate([kr, kc])
    out_cols = np.concatenate([kc, kr])
    out_data = np.concatenate([seg_max, seg_max])
    sym = sp.coo_matrix((out_data, (out_rows, out_cols)), shape=(n, n))
    return sym.tocsr()


def _row_normalize(csr: sp.csr_matrix, dtype) -> sp.csr_matrix:
    """float64  →  → cast dtype"""
    row_sums = np.asarray(csr.sum(axis=1)).ravel()  # float64
    if np.any(row_sums <= 0):
        raise ValueError(" 0 ")
    inv = 1.0 / row_sums
    out = sp.diags(inv).dot(csr).tocsr()
    out = out.astype(dtype)
    out.sort_indices()
    out.eliminate_zeros()
    return out


def build_transition_graph(
    latent,
    k: int = 30,
    kernel: Literal["gaussian", "connectivity"] = "gaussian",
    symmetrize: Literal["max", "mean", "none"] = "max",
    dtype=np.float32,
    knn_backend: Literal["pynndescent", "sklearn"] = "pynndescent",
    random_state: int | None = None,
) -> sp.csr_matrix:
    """ latent CSR


    ----
    latent : (n, d) array-like, d ≤ 256
    k :
    kernel : gaussian: exp(-(d/d_k)^2)connectivity:  1
    symmetrize : max / mean / none
    dtype :  dtype float32
    knn_backend : pynndescent/ sklearn
    random_state :  kNN


    ----
    scipy.sparse.csr_matrix, shape (n, n) CSR
    """
    X = _validate_latent(latent)
    n = X.shape[0]
    if symmetrize not in ("max", "mean", "none"):
        raise ValueError(f" symmetrize: {symmetrize!r}")

    indices, distances = _query_knn(X, k, knn_backend, random_state)
    rows, cols, data = _edge_weights(indices, distances, kernel)

    if symmetrize == "none":
        csr = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    elif symmetrize == "max":
        csr = _symmetrize_max(rows, cols, data, n)
    else:  # mean
        coo = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
        csr = ((coo + coo.T) * 0.5).tocsr()

    P = _row_normalize(csr, dtype)
    #  CSR
    return P


def absorb_states(P: sp.csr_matrix, absorbing_idx) -> sp.csr_matrix:
    """ 1

    validation-protocol I5 nnz=1data=1.0
     CSR
    """
    if not sp.isspmatrix_csr(P):
        raise ValueError("P  CSR")
    idx = np.unique(np.asarray(absorbing_idx, dtype=np.int64).ravel())
    if idx.size == 0:
        raise ValueError("absorbing_idx ")
    if idx.min() < 0 or idx.max() >= P.shape[0]:
        raise IndexError("absorbing_idx ")
    out = P.copy()
    n_rows = P.shape[0]
    #  indptr nnz=1
    row_nnz = np.diff(out.indptr)
    row_nnz[idx] = 1
    new_indptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(row_nnz, out=new_indptr[1:])
    new_indices = np.empty(new_indptr[-1], dtype=P.indices.dtype)
    new_data = np.empty(new_indptr[-1], dtype=P.data.dtype)
    is_abs = np.zeros(n_rows, dtype=bool)
    is_abs[idx] = True
    #  repeat +
    lengths = row_nnz[~is_abs]
    src_row_ids = np.nonzero(~is_abs)[0]
    total = int(lengths.sum())
    if total > 0:
        intra = np.arange(total, dtype=np.int64) - np.repeat(
            np.cumsum(lengths) - lengths, lengths
        )
        src_starts = np.repeat(out.indptr[src_row_ids], lengths)
        dest_starts = np.repeat(new_indptr[src_row_ids], lengths)
        src_pos = src_starts + intra
        dest_pos = dest_starts + intra
        new_indices[dest_pos] = out.indices[src_pos]
        new_data[dest_pos] = out.data[src_pos]
    #  1.0
    new_indices[new_indptr[idx]] = idx.astype(P.indices.dtype)
    new_data[new_indptr[idx]] = 1.0
    result = sp.csr_matrix((new_data, new_indices, new_indptr), shape=P.shape)
    result.sort_indices()
    return result


def split_transient_absorbing(P: sp.csr_matrix, absorbing_idx):
    """ Q=P_TT×R=P_TA× (I−Q)X=R

     (Q, R, trans_idx, abs_idx) CSR
    """
    if not sp.isspmatrix_csr(P):
        raise ValueError("P  CSR")
    n = P.shape[0]
    abs_idx = np.unique(np.asarray(absorbing_idx, dtype=np.int64).ravel())
    if abs_idx.size == 0 or abs_idx.size >= n:
        raise ValueError(" ≥1  < n")
    mask = np.zeros(n, dtype=bool)
    mask[abs_idx] = True
    trans_idx = np.nonzero(~mask)[0]
    Q = P[trans_idx][:, trans_idx].tocsr()
    R = P[trans_idx][:, abs_idx].tocsr()
    return Q, R, trans_idx, abs_idx
