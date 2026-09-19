"""fate.py (I−Q)X = Rscalable-fate M2

docs/validation-protocol-e1-addendum.md E1 P
fate  =  B = (I−Q)^{-1} R_agg
Q = P_TT×R_agg = P_TA  +  ≡
 lumpinga1

I7 graph.py
-  n×n S A = I−QILU
-  RHS  (n_t,) X (n_t, n_classes)  (n, n_classes)
  ——n_classes  O(10)
-  toarray()/todense()tests/test_fate.py  monkeypatch

C  75k scipy GMRES solve_mode="columns"
—— RHS 75k  5
 ~58 spilu  132–380s ILU
 float64  f32 ILU ——
spilu(f64) ~400s/ 77min
"""

from __future__ import annotations

import sys
import time
import warnings
from typing import Literal

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, gmres, spilu
import scipy.sparse.csgraph as _csgraph

from .graph import absorb_states, split_transient_absorbing

__all__ = ["absorption_probabilities"]


def _parse_labels(terminal_labels, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ → (,  id , )

    None / NaN / ""  np.unique
    """
    import pandas as pd

    if isinstance(terminal_labels, pd.Categorical):
        codes = np.asarray(terminal_labels.codes)
        if codes.shape[0] != n:
            raise ValueError(f"terminal_labels  {codes.shape[0]} != n={n}")
        cats = np.asarray(terminal_labels.categories)
        valid = codes >= 0
        lab = np.empty(n, dtype=object)
        lab[valid] = cats[codes[valid]].astype(str)
        lab[~valid] = None
    else:
        arr = np.asarray(terminal_labels)
        if arr.shape[0] != n:
            raise ValueError(f"terminal_labels  {arr.shape[0]} != n={n}")
        lab = np.empty(n, dtype=object)
        if arr.dtype.kind == "f":
            valid = ~np.isnan(arr)
            lab[valid] = arr[valid].astype(str)
            lab[~valid] = None
        else:
            for i, x in enumerate(arr):
                if x is None or (isinstance(x, float) and np.isnan(x)) or x == "":
                    lab[i] = None
                else:
                    lab[i] = str(x)
        valid = np.array([x is not None for x in lab], dtype=bool)

    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError("terminal_labels ")
    if n_valid == n:
        raise ValueError("terminal_labels (I−Q) ")
    classes = np.unique(lab[valid])
    cls_map = {c: j for j, c in enumerate(classes)}
    cls_id = np.full(n, -1, dtype=np.int64)
    cls_id[valid] = [cls_map[x] for x in lab[valid]]
    return valid, cls_id, classes


def _class_indicator(cls_id_abs: np.ndarray, n_classes: int) -> sp.csr_matrix:
    """ S (n_abs, n_classes)0/1 CSR"""
    return sp.csr_matrix(
        (
            np.ones(cls_id_abs.shape[0], dtype=np.float64),
            (np.arange(cls_id_abs.shape[0]), cls_id_abs),
        ),
        shape=(cls_id_abs.shape[0], n_classes),
    )


def _make_preconditioner(A: sp.csr_matrix, fill_factor: float, drop_tol: float):
    """spilu  fill_factor=10 actual  info """
    A_csc = A.tocsc()
    try:
        ilu = spilu(
            A_csc, fill_factor=fill_factor, drop_tol=drop_tol,
            options={"ILU_MILU": "SMILU_1"},
        )
        actual = fill_factor
    except Exception as e1:  # /
        warnings.warn(f"spilu(fill_factor={fill_factor}, SMILU_1) {e1} fill_factor=10")
        try:
            ilu = spilu(A_csc, fill_factor=10.0, drop_tol=drop_tol)
        except Exception:
            ilu = spilu(A_csc)  #
        actual = 10.0
    n = A.shape[0]
    M = LinearOperator((n, n), matvec=ilu.solve, dtype=A.dtype)
    return M, actual, ilu


def _gmres_column(A, M, b: np.ndarray, tol: float, restart: int, maxiter: int):
    """ GMRES pr_norm x0

    scipy gmres  ‖M(b−Ax)‖M
    M2 tol=1e-12  pr_norm  L∞  3e-8
    """
    n_iter = [0]
    last_rk = [np.inf]

    def _cb(rk):
        n_iter[0] += 1
        last_rk[0] = float(rk)

    b_norm = max(float(np.linalg.norm(b)), np.finfo(b.dtype).tiny)
    x = np.zeros_like(b)
    info = 1
    for _ in range(2):  # C f32  f64
        x, info = gmres(
            A, b, x0=x, rtol=tol, atol=0.0, restart=restart, maxiter=maxiter,
            M=M, callback=_cb, callback_type="pr_norm",
        )
        if info != 0:
            break
        true_rel = float(np.linalg.norm(b - A @ x)) / b_norm
        if true_rel <= max(tol, 10 * np.finfo(b.dtype).eps):
            info = 0
            break
        info = 1  #
    return x, info, n_iter[0], last_rk[0]


def _csc_column_dense(M_csc: sp.csc_matrix, j: int, dtype) -> np.ndarray:
    """ CSC  j  toarrayI7 """
    b = np.zeros(M_csc.shape[0], dtype=dtype)
    s, e = M_csc.indptr[j], M_csc.indptr[j + 1]
    if e > s:
        b[M_csc.indices[s:e]] = M_csc.data[s:e]
    return b


def absorption_probabilities(
    P: sp.csr_matrix,
    terminal_labels,
    block_size: int | None = 256,
    tol: float = 1e-6,
    *,
    restart: int = 50,
    maxiter: int = 1000,
    solve_mode: Literal["columns", "aggregated"] = "columns",
    preconditioner: Literal["none", "ilu"] = "none",
    ilu_fill_factor: float = 1.0,
    ilu_drop_tol: float = 0.0,
    return_info: bool = False,
    verbose: bool = False,
    col_callback=None,
):
    """ GMRES  (I−Q)X = R

    C  75k  #2——
    - columns (I−Q)x = R[:,j]
       aggregated a1
      75k  ~58  RHS 5
    - aggregated RHS CR2

    75k spilu  132–380s  GMRES ——
    ILU  noneilu


    ----
    P : (n, n) CSR dtype f32  / f64
    terminal_labels :  n str / int / Categorical
        None / NaN / ""  =
    block_size : RHS None =
    tol : GMRES rtolatol=0
    restart, maxiter : GMRES
    solve_mode : columns/ aggregated
    preconditioner : none/ ilu
    ilu_fill_factor, ilu_drop_tol : spilu  preconditioner="ilu"
    return_info : True  (B, info)
    verbose : / stderr


    ----
    B : (n, n_classes)dtype  P f32f64  f64
         cast ——M2  f64  cast f32  ~3e-8
    info : dictclasses /  GMRES columns
         maxconverged  / wall_time_s / n_f64_fallback / ILU
    """
    if not sp.isspmatrix_csr(P):
        raise ValueError("P  CSR")
    n = P.shape[0]
    if P.shape[0] != P.shape[1]:
        raise ValueError("P ")
    if block_size is None:
        block_size = 1 << 30
    if block_size < 1:
        raise ValueError(f"block_size  ≥1 {block_size}")
    if solve_mode not in ("columns", "aggregated"):
        raise ValueError(f" solve_mode: {solve_mode!r} columns / aggregated")
    if preconditioner not in ("none", "ilu"):
        raise ValueError(f" preconditioner: {preconditioner!r} none / ilu")

    valid, cls_id, classes = _parse_labels(terminal_labels, n)
    n_classes = classes.shape[0]
    abs_idx = np.nonzero(valid)[0]

    # M2 2k  190  → (I−Q)
    # / ⇒
    n_comp, comp = _csgraph.connected_components(P, directed=False)
    abs_comps = np.unique(comp[abs_idx])
    n_orphan = int((~np.isin(comp, abs_comps)).sum())
    if n_orphan:
        raise ValueError(
            f"{n_orphan}  {n_comp} "
            f"(I−Q)  k /  / "
        )

    dt = P.dtype if P.dtype in (np.float32, np.float64) else np.float64
    work = np.dtype(dt).type

    # Q / R  I5
    Pa = absorb_states(P, abs_idx)
    Q, R, trans_idx, abs_idx = split_transient_absorbing(Pa, abs_idx)
    Q = Q.astype(work)
    R = R.astype(work)
    n_t = Q.shape[0]

    #  Saggregated  RHScolumns
    S = _class_indicator(cls_id[abs_idx], n_classes).astype(work)
    R_agg = (R @ S).tocsc()

    # A = I − QCSR 1−q_ii
    A = (sp.eye(n_t, format="csr", dtype=work) - Q).tocsr()
    A.sort_indices()

    t0 = time.perf_counter()
    if preconditioner == "ilu":
        M, ilu_actual, ilu = _make_preconditioner(A, ilu_fill_factor, ilu_drop_tol)
        t_ilu = time.perf_counter() - t0
        if verbose:
            print(f"[fate] ILU(fill={ilu_actual})  {t_ilu:.1f}s", file=sys.stderr)
    else:
        M = ilu = None
        ilu_actual = None

    # f64  f32 ILU ILU
    #C  75k  spilu(f64, fill=10)  ~400s/ 77min
    A64 = None

    def _f64_fallback(b):
        nonlocal A64
        if A64 is None:
            A64 = A.astype(np.float64)
        if ilu is not None:
            M64 = LinearOperator(
                (n_t, n_t),
                matvec=lambda v: ilu.solve(np.asarray(v, dtype=np.float32)).astype(np.float64),
                dtype=np.float64,
            )
        else:
            M64 = None
        return _gmres_column(A64, M64, b.astype(np.float64), tol, restart, maxiter)

    # (rhs , )
    if solve_mode == "aggregated":
        rhs_mat = R_agg
        targets = [(j, j) for j in range(n_classes)]
    else:  # columns
        rhs_mat = R.tocsc()
        targets = [(j, int(c)) for j, c in enumerate(cls_id[abs_idx])]

    X = np.zeros((n_t, n_classes), dtype=work)
    iters = np.zeros(n_classes, dtype=np.int64)       #
    iters_max = np.zeros(n_classes, dtype=np.int64)   #
    pr_norm = np.full(n_classes, np.inf)
    n_converged = np.zeros(n_classes, dtype=np.int64)
    n_f64_fallback = 0

    for start in range(0, len(targets), block_size):
        for j, c in targets[start:start + block_size]:
            b = _csc_column_dense(rhs_mat, j, work)
            t_col0 = time.perf_counter()
            x, info, it, rk = _gmres_column(A, M, b, tol, restart, maxiter)
            if info != 0 and work is np.float32:
                n_f64_fallback += 1
                x64, info64, it64, rk64 = _f64_fallback(b)
                if info64 == 0:
                    x, info = x64.astype(work), 0
                    it, rk = it + it64, rk64
            if info != 0:
                raise RuntimeError(
                    f"GMRES  {classes[c]!r}  {j} info={info}"
                    f" f64  /  maxiter"
                )
            if solve_mode == "aggregated":
                X[:, c] = x
            else:
                X[:, c] += x
            t_col = time.perf_counter() - t_col0
            b_norm = max(float(np.linalg.norm(b)), np.finfo(b.dtype).tiny)
            true_rel = float(np.linalg.norm(b - A @ x)) / b_norm
            iters[c] += it
            iters_max[c] = max(iters_max[c], it)
            pr_norm[c] = min(pr_norm[c], rk)
            n_converged[c] += 1
            if col_callback is not None:
                col_callback(j, c, it, true_rel, t_col)
        if verbose:
            done = targets[min(start + block_size, len(targets)) - 1][1]
            print(f"[fate]  {start}–{min(start + block_size, len(targets)) - 1} "
                  f" {classes[done]}", file=sys.stderr)
    converged = n_converged > 0
    if verbose:
        print(f"[fate]  {iters.tolist()}f64  {n_f64_fallback} ",
              file=sys.stderr)

    wall_s = time.perf_counter() - t0

    # ‖AX − R_agg‖∞ @
    resid = np.abs(A @ X - _csc_all(R_agg, work))
    resid_max = resid.max(axis=0)

    #  (n, n_classes) =  = one-hotdtype
    B = np.zeros((n, n_classes), dtype=work)
    B[trans_idx, :] = X
    B[abs_idx, cls_id[abs_idx]] = 1.0

    if not return_info:
        return B
    info = {
        "classes": [str(c) for c in classes],
        "n_transient": int(n_t),
        "n_absorbing": int(abs_idx.shape[0]),
        "solve_mode": solve_mode,
        "preconditioner": preconditioner,
        "gmres_iters": iters,
        "gmres_iters_max": iters_max,
        "gmres_pr_norm": pr_norm,
        "residual_max": np.asarray(resid_max).ravel(),
        "converged": converged,
        "wall_time_s": wall_s,
        "n_f64_fallback": n_f64_fallback,
        "ilu_fill_factor_actual": ilu_actual,
        "ilu_drop_tol": ilu_drop_tol,
        "dtype": str(work.__name__),
    }
    return B, info


def _csc_all(M_csc: sp.csc_matrix, dtype) -> np.ndarray:
    """CSC →  (n_t, n_classes)  toarrayn_classes  O(10)"""
    out = np.zeros(M_csc.shape, dtype=dtype)
    for j in range(M_csc.shape[1]):
        s, e = M_csc.indptr[j], M_csc.indptr[j + 1]
        if e > s:
            out[M_csc.indices[s:e], j] = M_csc.data[s:e]
    return out
