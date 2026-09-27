"""fate.py：吸收概率求解器 (I−Q)X = R（scalable-fate M2）。

数学依据（docs/validation-protocol-e1-addendum.md E1）：给定同一行随机转移矩阵 P
与同一终末态集合，fate 概率 = 基本矩阵方程 B = (I−Q)^{-1} R_agg，其中
Q = P_TT（瞬态×瞬态）、R_agg = P_TA 按类聚合（逐细胞吸收 + 类内求和 ≡
伪态 lumping，a1 测试直接自证该等价性）。

内存纪律（I7，同 graph.py）：
- 禁止任何 n×n 稠密中间量；S 类指示阵、A = I−Q、ILU 因子全稀疏；
- 唯一稠密量：逐列 RHS 向量 (n_t,)、解缓冲 X (n_t, n_classes) 与输出 (n, n_classes)
  ——n_classes 为 O(10) 小常数；
- 不使用 toarray()/todense()（tests/test_fate.py 有 monkeypatch 守卫）。

求解路径（C 层 75k 实测定案）：scipy GMRES，默认 solve_mode="columns"（逐吸收细胞
求解后按类求和——聚合 RHS 在亚稳链上激发慢模态，75k 实测 5 万迭代不收敛；
逐列 ~58 迭代收敛），默认无预处理子（spilu 构造 132–380s 被判负资产，ILU 仅可选）；
单列不收敛时该列自动升 float64 重解（复用 f32 ILU 若存在，不重建——回退重建
spilu(f64) ~400s/列是 77min 卡死事件的根因）。
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
from .reachability import check_absorption_precondition

__all__ = ["absorption_probabilities"]


def _parse_labels(terminal_labels, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """解析逐细胞终末标签 → (有效掩码, 类 id 数组, 类名数组)。

    None / NaN / "" 视为瞬态（无标签）。类序按 np.unique 排序，确定性。
    """
    import pandas as pd

    if isinstance(terminal_labels, pd.Categorical):
        codes = np.asarray(terminal_labels.codes)
        if codes.shape[0] != n:
            raise ValueError(f"terminal_labels 长度 {codes.shape[0]} != n={n}")
        cats = np.asarray(terminal_labels.categories)
        valid = codes >= 0
        lab = np.empty(n, dtype=object)
        lab[valid] = cats[codes[valid]].astype(str)
        lab[~valid] = None
    else:
        arr = np.asarray(terminal_labels)
        if arr.shape[0] != n:
            raise ValueError(f"terminal_labels 长度 {arr.shape[0]} != n={n}")
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
        raise ValueError("terminal_labels 全部为空（无吸收类）")
    if n_valid == n:
        raise ValueError("terminal_labels 覆盖全部细胞（无瞬态，(I−Q) 为空）")
    classes = np.unique(lab[valid])
    cls_map = {c: j for j, c in enumerate(classes)}
    cls_id = np.full(n, -1, dtype=np.int64)
    cls_id[valid] = [cls_map[x] for x in lab[valid]]
    return valid, cls_id, classes


def _class_indicator(cls_id_abs: np.ndarray, n_classes: int) -> sp.csr_matrix:
    """吸收细胞的类指示阵 S (n_abs, n_classes)，0/1 CSR。"""
    return sp.csr_matrix(
        (
            np.ones(cls_id_abs.shape[0], dtype=np.float64),
            (np.arange(cls_id_abs.shape[0]), cls_id_abs),
        ),
        shape=(cls_id_abs.shape[0], n_classes),
    )


def _make_preconditioner(A: sp.csr_matrix, fill_factor: float, drop_tol: float):
    """spilu 预处理子；失败回退 fill_factor=10（记 actual 供 info 报告）。"""
    A_csc = A.tocsc()
    try:
        ilu = spilu(
            A_csc, fill_factor=fill_factor, drop_tol=drop_tol,
            options={"ILU_MILU": "SMILU_1"},
        )
        actual = fill_factor
    except Exception as e1:  # 奇异/选项不支持等，统一回退
        warnings.warn(f"spilu(fill_factor={fill_factor}, SMILU_1) 失败（{e1}），回退 fill_factor=10")
        try:
            ilu = spilu(A_csc, fill_factor=10.0, drop_tol=drop_tol)
        except Exception:
            ilu = spilu(A_csc)  # 全默认兜底
        actual = 10.0
    n = A.shape[0]
    M = LinearOperator((n, n), matvec=ilu.solve, dtype=A.dtype)
    return M, actual, ilu


def _gmres_column(A, M, b: np.ndarray, tol: float, restart: int, maxiter: int):
    """单列 GMRES；以真残差（非预处理 pr_norm）复检，不达标用 x0 续解。

    动机：scipy gmres 停机判据基于左预处理残差 ‖M(b−Ax)‖，M 病态时会低估
    真残差（M2 实测：tol=1e-12 下 pr_norm 达标但真解 L∞ 偏差 3e-8）。
    """
    n_iter = [0]
    last_rk = [np.inf]

    def _cb(rk):
        n_iter[0] += 1
        last_rk[0] = float(rk)

    b_norm = max(float(np.linalg.norm(b)), np.finfo(b.dtype).tiny)
    x = np.zeros_like(b)
    info = 1
    for _ in range(2):  # 续解轮数上限（C 层实测：f32 死循环轮数是无底洞，快速让位 f64 回退）
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
        info = 1  # 真残差不达标，续解
    return x, info, n_iter[0], last_rk[0]


def _csc_column_dense(M_csc: sp.csc_matrix, j: int, dtype) -> np.ndarray:
    """取 CSC 第 j 列为稠密向量（不经 toarray，I7 合规）。"""
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
    """分块稀疏 GMRES 求解吸收概率 (I−Q)X = R。

    求解模式（C 层 75k 实测定案，发现 #2——伪态聚合在亚稳链上激发慢模态）：
    - columns（默认）：按吸收细胞逐列解 (I−Q)x = R[:,j]，解完按类求和
      （线性性：聚合移到解后，与 aggregated 数学上严格相等，a1 已证）；
      75k 实测单列 ~58 迭代收敛，聚合 RHS 5 万迭代不收敛。
    - aggregated：先聚合成类 RHS 再解（CR2 伪态口径，仅留作对照）。

    预处理子（75k 实测：spilu 构造 132–380s 而无预处理 GMRES 秒级收敛——
    ILU 被判为负资产）：默认 none；ilu 保留为可选项。

    参数
    ----
    P : (n, n) CSR 行随机转移矩阵（dtype 决定求解精度：f32 生产 / f64 对照）
    terminal_labels : 长度 n 的逐细胞类标签（str / int / Categorical）；
        None / NaN / "" 视为瞬态。每个唯一非空标签 = 一个吸收类。
    block_size : RHS 列分块大小（内存口径；None = 全量）
    tol : GMRES rtol（atol=0，纯相对判据）
    restart, maxiter : GMRES 参数
    solve_mode : columns（默认）/ aggregated（对照）
    preconditioner : none（默认）/ ilu
    ilu_fill_factor, ilu_drop_tol : spilu 参数（仅 preconditioner="ilu" 时生效）
    return_info : True 时返回 (B, info)
    verbose : 逐类/逐阶段进度打 stderr

    返回
    ----
    B : (n, n_classes)，dtype 跟随 P（生产 f32；f64 对照路径保留 f64 以免
        下行 cast 量化掩盖求解误差——M2 实测 f64 解 cast f32 引入 ~3e-8 残差）
    info : dict（classes / 逐类 GMRES 迭代数（columns 模式为类内合计）与单列峰值、
        真残差 max、converged 标志 / wall_time_s / n_f64_fallback / ILU 实际参数）
    """
    if not sp.isspmatrix_csr(P):
        raise ValueError("P 必须为 CSR")
    n = P.shape[0]
    if P.shape[0] != P.shape[1]:
        raise ValueError("P 必须为方阵")
    if block_size is None:
        block_size = 1 << 30
    if block_size < 1:
        raise ValueError(f"block_size 必须 ≥1，得到 {block_size}")
    if solve_mode not in ("columns", "aggregated"):
        raise ValueError(f"未知 solve_mode: {solve_mode!r}（可选 columns / aggregated）")
    if preconditioner not in ("none", "ilu"):
        raise ValueError(f"未知 preconditioner: {preconditioner!r}（可选 none / ilu）")

    valid, cls_id, classes = _parse_labels(terminal_labels, n)
    n_classes = classes.shape[0]
    abs_idx = np.nonzero(valid)[0]

    # 可达性守卫（M2 实测：2k 合成链曾含 190 个无吸收态孤岛连通域 → (I−Q) 数值奇异，
    # 直接/迭代解在孤岛行上静默产出垃圾）。无向连通域不含吸收态 ⇒ 必不可达。
    n_comp, comp = _csgraph.connected_components(P, directed=False)
    abs_comps = np.unique(comp[abs_idx])
    n_orphan = int((~np.isin(comp, abs_comps)).sum())
    if n_orphan:
        raise ValueError(
            f"{n_orphan} 个细胞所在连通域（共 {n_comp} 个）不含任何吸收态，"
            f"(I−Q) 奇异、吸收概率无定义：请增大 k / 调整终末态 / 剔除孤岛后重试"
        )

    dt = P.dtype if P.dtype in (np.float32, np.float64) else np.float64
    work = np.dtype(dt).type

    # Q / R 切分（吸收态化保证 I5 结构，再切分）
    Pa = absorb_states(P, abs_idx)
    # 有向可达性预检：保证 rho(Q)<1（I−Q 为非奇异 M-矩阵）
    reach = check_absorption_precondition(Pa, abs_idx)
    if verbose:
        print(f"[fate] 预检 rho(Q)<1 通过：{reach['n_transient']} 个瞬态节点，"
              f"不可达 {reach['n_unreachable']} 个", file=sys.stderr)
    Q, R, trans_idx, abs_idx = split_transient_absorbing(Pa, abs_idx)
    Q = Q.astype(work)
    R = R.astype(work)
    n_t = Q.shape[0]

    # 类内聚合指示阵 S（aggregated 模式作 RHS；columns 模式留作残差校验）
    S = _class_indicator(cls_id[abs_idx], n_classes).astype(work)
    R_agg = (R @ S).tocsc()

    # A = I − Q（CSR，主对角 1−q_ii）
    A = (sp.eye(n_t, format="csr", dtype=work) - Q).tocsr()
    A.sort_indices()

    t0 = time.perf_counter()
    if preconditioner == "ilu":
        M, ilu_actual, ilu = _make_preconditioner(A, ilu_fill_factor, ilu_drop_tol)
        t_ilu = time.perf_counter() - t0
        if verbose:
            print(f"[fate] ILU(fill={ilu_actual}) 分解 {t_ilu:.1f}s", file=sys.stderr)
    else:
        M = ilu = None
        ilu_actual = None

    # f64 回退的预处理子：复用 f32 ILU（近似即可）；无 ILU 则无预处理
    #（C 层 75k 实测教训：回退重建 spilu(f64, fill=10) 需 ~400s/列，是 77min 卡死根因）
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

    # 求解目标：(rhs 列, 归属类) 列表
    if solve_mode == "aggregated":
        rhs_mat = R_agg
        targets = [(j, j) for j in range(n_classes)]
    else:  # columns：逐吸收细胞
        rhs_mat = R.tocsc()
        targets = [(j, int(c)) for j, c in enumerate(cls_id[abs_idx])]

    X = np.zeros((n_t, n_classes), dtype=work)
    iters = np.zeros(n_classes, dtype=np.int64)       # 类内迭代合计
    iters_max = np.zeros(n_classes, dtype=np.int64)   # 单列峰值
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
                    f"GMRES 未收敛：类 {classes[c]!r} 列 {j} info={info}"
                    f"（含 f64 回退），检查吸收态可达性 / 放大 maxiter"
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
            print(f"[fate] 块 {start}–{min(start + block_size, len(targets)) - 1} 完成"
                  f"（类 {classes[done]}）", file=sys.stderr)
    converged = n_converged > 0
    if verbose:
        print(f"[fate] 求解完成：逐类迭代合计 {iters.tolist()}，f64 回退 {n_f64_fallback} 列",
              file=sys.stderr)

    wall_s = time.perf_counter() - t0

    # 真残差（非预处理）：‖AX − R_agg‖∞ 逐类，稀疏@稠密
    resid = np.abs(A @ X - _csc_all(R_agg, work))
    resid_max = resid.max(axis=0)

    # 装配输出 (n, n_classes)：瞬态行 = 解，吸收行 = one-hot；dtype 跟随求解精度
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
    """CSC → 稠密 (n_t, n_classes) 小矩阵（不经 toarray，n_classes 为 O(10)）。"""
    out = np.zeros(M_csc.shape, dtype=dtype)
    for j in range(M_csc.shape[1]):
        s, e = M_csc.indptr[j], M_csc.indptr[j + 1]
        if e > s:
            out[M_csc.indices[s:e], j] = M_csc.data[s:e]
    return out
