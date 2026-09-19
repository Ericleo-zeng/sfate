"""fate.py E3 A a1/a2/a3/a5+ I  + no_densify

docs/validation-protocol-e1-addendum.md E3  + docs/validation-protocol.md §2
fixture_5k /
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.csgraph  # noqa: F401  sp.csgraph
from scipy.sparse.linalg import spsolve

from sfate.fate import _class_indicator, _csc_column_dense, absorption_probabilities
from sfate.graph import absorb_states, build_transition_graph, split_transient_absorbing
from conftest import make_branching_latent

SEED = 20260909
ATOL_ROW = 1e-4  # I1 float32


# ----------  helper ----------

def _dense_csc(M) -> np.ndarray:
    """ →  toarray (n_t, n_classes/300)"""
    M = sp.csc_matrix(M)
    out = np.zeros(M.shape, dtype=M.dtype)
    for j in range(M.shape[1]):
        s, e = M.indptr[j], M.indptr[j + 1]
        if e > s:
            out[M.indices[s:e], j] = M.data[s:e]
    return out


def _chain(n=2000, n_term_cells=100, dtype=np.float64):
    """n≈2k  +  n_term_cells 3

    M2 kNN  latent  ~190 I−Q
    2000→1600
    """
    X, parts = make_branching_latent(n=n, d=10, n_terminal=3, seed=7, return_parts=True)
    P = build_transition_graph(X, k=30, random_state=SEED, dtype=dtype)
    labels = np.array([None] * n, dtype=object)
    for j in range(3):
        idx = np.nonzero(parts["part_ids"] == 2 + j)[0]
        proj = (X[idx].astype(np.float64) - parts["branch_base"]) @ parts["branch_dirs"][j]
        top = idx[np.argsort(proj, kind="stable")[-n_term_cells:]]
        for i in top:
            labels[i] = f"T{j}"
    #
    valid = np.array([x is not None for x in labels], dtype=bool)
    n_comp, comp = sp.csgraph.connected_components(P, directed=False)
    keep = np.isin(comp, np.unique(comp[valid]))
    return P[keep][:, keep].tocsr(), labels[keep]


def _golden_f64(P, labels):
    """spsolve(f64)  (gold (n_t, n_cls), trans_idx, classes)"""
    valid = np.array([x is not None for x in labels], dtype=bool)
    abs_idx = np.nonzero(valid)[0]
    classes = np.unique(labels[valid])
    cls_map = {c: j for j, c in enumerate(classes)}
    cls_id = np.array([cls_map[labels[i]] for i in abs_idx], dtype=np.int64)
    P64 = P.astype(np.float64)
    Pa = absorb_states(P64, abs_idx)
    Q, R, trans_idx, abs_idx = split_transient_absorbing(Pa, abs_idx)
    A = (sp.eye(Q.shape[0], format="csr", dtype=np.float64) - Q).tocsr()
    S = _class_indicator(cls_id, classes.shape[0])
    R_agg = (R @ S).tocsc()
    gold = np.column_stack([
        spsolve(A, _csc_column_dense(R_agg, j, np.float64))
        for j in range(classes.shape[0])
    ])
    return gold, trans_idx, classes


# ---------- a1+ vs  lumpingf64  ----------

def test_a1_lumping_equivalence():
    P, labels = _chain(n=2000, n_term_cells=100)
    valid = np.array([x is not None for x in labels], dtype=bool)
    abs_idx = np.nonzero(valid)[0]
    classes = np.unique(labels[valid])
    cls_id = np.array(
        [{c: j for j, c in enumerate(classes)}[labels[i]] for i in abs_idx]
    )
    Pa = absorb_states(P, abs_idx)
    Q, R, trans_idx, _ = split_transient_absorbing(Pa, abs_idx)
    A = (sp.eye(Q.shape[0], format="csr", dtype=np.float64) - Q).tocsr()

    # variant ① CR1 (n_t, n_abs)
    B_pc = spsolve(A, R.tocsc())  # sparse RHS → sparse
    S = _class_indicator(cls_id, classes.shape[0])
    B1 = _dense_csc(B_pc @ S)

    # variant ②  lumpingCR2  RHS
    R_agg = (R @ S).tocsc()
    B2 = np.column_stack([
        spsolve(A, _csc_column_dense(R_agg, j, np.float64))
        for j in range(classes.shape[0])
    ])

    diff = np.abs(B1 - B2)
    assert diff.max() <= 1e-6, f"a1  variant L∞={diff.max():.2e} > 1e-6"


def _kappa_inf(A: sp.csr_matrix) -> float:
    """κ∞(A) M-A=I−Q, Q≥0  ⇒ A⁻¹=ΣQᵏ≥0
    ‖A⁻¹‖∞ = max(A⁻¹·1)‖A‖∞ =
    onenormest  ~400×M2 """
    from scipy.sparse.linalg import splu

    lu = splu(A.tocsc())
    n1 = lu.solve(np.ones(A.shape[0]))
    assert n1.min() >= -1e-9, "A⁻¹ M-"
    row_abs = np.abs(A).sum(axis=1)
    row_abs = np.asarray(row_abs).ravel()
    return float(row_abs.max() * n1.max())


# ---------- a2GMRES(f64, tol=1e-12) vs spsolve  ----------

@pytest.mark.parametrize("solve_mode", ["columns", "aggregated"])
def test_a2_direct_vs_gmres_f64(solve_mode):
    """M2  L∞≤1e-9  solve_mode

    ①  max ‖(I−Q)x−b‖∞ ≤ 1e-12columns
        =  ×  ×n_rep=100
    ② L∞(vs spsolve) ≤ 10·κ∞(I−Q)·backward_maxκ∞  M-

    GMRES  rtol=1e-12  tol
    ≈600·eps eps 10·κ·epsM2
      ≲ κ·——
     κ∞  1e-9 M2
    ① (I−Q)  _chain  fate.py
    ②  cast f32f64  ~3e-8
    —— dtype  P
    """
    P, labels = _chain(n=2000, n_term_cells=100)
    gold, trans_idx, classes = _golden_f64(P, labels)
    B, info = absorption_probabilities(
        P, labels, tol=1e-12, return_info=True,
        solve_mode=solve_mode, preconditioner="ilu",  #  ILU
    )
    assert info["converged"].all(), f"a2[{solve_mode}] GMRES "
    assert info["n_f64_fallback"] == 0
    # ①
    backward = float(info["residual_max"].max())
    gate = 1e-12 if solve_mode == "aggregated" else 1e-12 * 100  # ×n_rep
    assert backward <= gate, f"a2[{solve_mode}]  {backward:.2e} > {gate:.0e}"
    # ②  κ∞
    valid = np.array([x is not None for x in labels], dtype=bool)
    Pa = absorb_states(P, np.nonzero(valid)[0])
    Q, _, _, _ = split_transient_absorbing(Pa, np.nonzero(valid)[0])
    A = (sp.eye(Q.shape[0], format="csr", dtype=np.float64) - Q).tocsr()
    kappa_inf = _kappa_inf(A)
    bound = 10 * kappa_inf * backward
    diff = np.abs(B[trans_idx].astype(np.float64) - gold)
    assert diff.max() <= bound, (
        f"a2 L∞={diff.max():.2e} > 10·κ∞·backward={bound:.2e} "
        f"(κ∞={kappa_inf:.2e}, backward={backward:.2e})"
    )


# ---------- a3fixture_5k  f32 vs spsolve f64 ----------

_FIXTURE_CACHE: dict = {}


def _fixture_problem(fixture5k):
    """fixture_5k6  microglia_state  30 """
    if "prob" in _FIXTURE_CACHE:
        return _FIXTURE_CACHE["prob"]
    X = np.asarray(fixture5k.obsm["X_scVI"], dtype=np.float32)
    states = np.asarray(fixture5k.obs["microglia_state"].astype(str))
    n = X.shape[0]
    labels = np.array([None] * n, dtype=object)
    X64 = X.astype(np.float64)
    for c in np.unique(states):
        idx = np.nonzero(states == c)[0]
        centroid = X64[idx].mean(axis=0)
        dist = np.linalg.norm(X64[idx] - centroid, axis=1)
        reps = idx[np.argsort(dist, kind="stable")[:30]]
        for i in reps:
            labels[i] = c
    P = build_transition_graph(X, k=30, random_state=SEED)
    gold, trans_idx, classes = _golden_f64(P, labels)
    prob = {"P": P, "labels": labels, "gold": gold,
            "trans_idx": trans_idx, "classes": classes}
    _FIXTURE_CACHE["prob"] = prob
    return prob


@pytest.mark.parametrize("solve_mode,block_size", [
    ("aggregated", 1), ("aggregated", 16), ("aggregated", 256), ("aggregated", None),
    ("columns", 16), ("columns", None),
])
def test_a3_f32_vs_f64(fixture5k, solve_mode, block_size):
    """ solve_mode columns="""
    prob = _fixture_problem(fixture5k)
    B, info = absorption_probabilities(
        prob["P"], prob["labels"], block_size=block_size, tol=1e-6,
        return_info=True, solve_mode=solve_mode,
    )
    assert info["converged"].all(), (
        f"a3[{solve_mode}] bs={block_size} GMRES {info['classes']}"
        f" iters={info['gmres_iters']}"
    )
    diff = np.abs(B[prob["trans_idx"]].astype(np.float64) - prob["gold"])
    q = np.quantile(diff, [0.5, 0.9, 0.99, 1.0])
    assert diff.max() <= 1e-5, (
        f"a3[{solve_mode}] bs={block_size} L∞={diff.max():.3e} > 1e-5"
        f" p50/p90/p99/max = {q}"
    )


def test_a3_mode_equivalence(fixture5k):
    """columns vs aggregated  f32  1e-5  a3"""
    prob = _fixture_problem(fixture5k)
    Bs = {}
    for mode in ["columns", "aggregated"]:
        Bs[mode], _ = absorption_probabilities(
            prob["P"], prob["labels"], tol=1e-6, return_info=True, solve_mode=mode
        )
    d = np.abs(Bs["columns"].astype(np.float64) - Bs["aggregated"].astype(np.float64)).max()
    assert d <= 1e-5, f"columns vs aggregated L∞={d:.2e} > 1e-5"


@pytest.mark.parametrize("solve_mode", ["columns", "aggregated"])
def test_a3_block_consistency(fixture5k, solve_mode):
    """block_size  ≤1e-6"""
    prob = _fixture_problem(fixture5k)
    Bs = {}
    for bs in [16, None]:
        Bs[bs], _ = absorption_probabilities(
            prob["P"], prob["labels"], block_size=bs, tol=1e-6, return_info=True,
            solve_mode=solve_mode,
        )
    d = np.abs(Bs[16].astype(np.float64) - Bs[None].astype(np.float64)).max()
    assert d <= 1e-6, f"[{solve_mode}] block_size=16 vs   {d:.2e} > 1e-6"


# ---------- a4fixture  I1/I2/I3/I4/I6 +  one-hot ----------

@pytest.mark.parametrize("solve_mode", ["columns", "aggregated"])
def test_a4_invariants_on_fixture(fixture5k, solve_mode):
    prob = _fixture_problem(fixture5k)
    B, info = absorption_probabilities(
        prob["P"], prob["labels"], block_size=256, tol=1e-6, return_info=True,
        solve_mode=solve_mode,
    )
    assert B.dtype == np.float32 and B.shape == (prob["P"].shape[0], 6)
    # I1/I4 =1
    np.testing.assert_allclose(B.sum(axis=1), 1.0, rtol=0, atol=ATOL_ROW)
    # I2/I3
    assert B.min() >= -1e-6, f"I2  min={B.min():.2e}"
    assert B.max() <= 1 + 1e-6, f"I3  max={B.max():.2e}"
    # I6
    assert info["converged"].all()
    #  one-hot
    valid = np.array([x is not None for x in prob["labels"]], dtype=bool)
    abs_rows = B[valid]
    np.testing.assert_allclose(abs_rows.sum(axis=1), 1.0, rtol=0, atol=0)
    assert set(np.unique(abs_rows)).issubset({0.0, 1.0})


# ---------- a5 ----------

@pytest.mark.parametrize("solve_mode", ["columns", "aggregated"])
def test_a5_determinism(solve_mode):
    P1, labels = _chain(n=2000, n_term_cells=100, dtype=np.float32)
    B1, info1 = absorption_probabilities(P1, labels, tol=1e-6, return_info=True,
                                         solve_mode=solve_mode)
    P2, _ = _chain(n=2000, n_term_cells=100, dtype=np.float32)
    B2, info2 = absorption_probabilities(P2, labels, tol=1e-6, return_info=True,
                                         solve_mode=solve_mode)
    np.testing.assert_array_equal(P1.indptr, P2.indptr)
    np.testing.assert_array_equal(P1.indices, P2.indices)
    np.testing.assert_allclose(B1, B2, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(info1["gmres_iters"], info2["gmres_iters"])


# ---------- I7no_densify monkeypatch  ----------

def test_no_densify_fate(monkeypatch):
    def _boom(self=None, *a, **k):
        raise AssertionError(" toarray/todense")

    monkeypatch.setattr(sp.csr_matrix, "toarray", _boom, raising=True)
    monkeypatch.setattr(sp.csr_matrix, "todense", _boom, raising=True)
    monkeypatch.setattr(sp.coo_matrix, "toarray", _boom, raising=True)
    monkeypatch.setattr(sp.csc_matrix, "toarray", _boom, raising=True)
    P, labels = _chain(n=800, n_term_cells=30, dtype=np.float32)
    # columns + aggregated + ilu
    B, info = absorption_probabilities(P, labels, tol=1e-6, return_info=True)
    assert B.shape == (P.shape[0], 3) and info["converged"].all()
    B2, info2 = absorption_probabilities(P, labels, tol=1e-6, return_info=True,
                                         solve_mode="aggregated", preconditioner="ilu")
    assert info2["converged"].all()


# ----------  ----------

def test_input_validation():
    P, labels = _chain(n=800, n_term_cells=30, dtype=np.float32)
    n = P.shape[0]
    with pytest.raises(ValueError, match="CSR"):
        absorption_probabilities(P.tocoo(), labels)
    with pytest.raises(ValueError, match="terminal_labels"):
        absorption_probabilities(P, labels[:100])
    with pytest.raises(ValueError, match="terminal_labels"):
        absorption_probabilities(P, np.array([None] * n, dtype=object))
    with pytest.raises(ValueError, match="terminal_labels"):
        absorption_probabilities(P, np.array(["A"] * n, dtype=object))
    with pytest.raises(ValueError, match="block_size"):
        absorption_probabilities(P, labels, block_size=0)
    with pytest.raises(ValueError, match="solve_mode"):
        absorption_probabilities(P, labels, solve_mode="bogus")
    with pytest.raises(ValueError, match="preconditioner"):
        absorption_probabilities(P, labels, preconditioner="bogus")


def test_orphan_component_guard():
    """ → ValueErrorM2  (I−Q) """
    rng = np.random.default_rng(0)
    #
    X = np.vstack([rng.normal(0, 0.1, size=(200, 5)),
                   rng.normal(100, 0.1, size=(200, 5))]).astype(np.float32)
    P = build_transition_graph(X, k=10, random_state=SEED)
    labels = np.array([None] * 400, dtype=object)
    labels[:20] = "T0"  #  0  1
    with pytest.raises(ValueError, match="I−Q"):
        absorption_probabilities(P, labels)
