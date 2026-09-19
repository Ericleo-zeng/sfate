"""graph.py 12  M1  + fixture_5k

docs/review-summary.md §4  + docs/validation-protocol.md I1–I10
 n≤2000 mem_smoke_10k  PR CI
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest
import scipy.sparse as sp

from sfate.graph import (
    MAX_LATENT_DIM,
    absorb_states,
    build_transition_graph,
    split_transient_absorbing,
)

SEED = 20260909
ATOL_ROW = 1e-4  # float32 protocol I1 dtype


# ---------- I1  ----------

def test_row_stochastic(latent_2k):
    P = build_transition_graph(latent_2k, k=30, random_state=SEED)
    row_sums = np.asarray(P.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, 1.0, rtol=0, atol=ATOL_ROW)


# ---------- I2 clipping  ----------

def test_nonnegative_no_clip(latent_2k):
    P = build_transition_graph(latent_2k, k=30, random_state=SEED)
    assert P.data.min() >= 0.0, ""
    # gaussian  >0 clipping  0


# ---------- I7 monkeypatch  + tracemalloc  ----------

def test_no_densify(latent_2k, monkeypatch):
    def _boom(self=None, *a, **k):
        raise AssertionError(" toarray/todense")

    monkeypatch.setattr(sp.csr_matrix, "toarray", _boom, raising=True)
    monkeypatch.setattr(sp.csr_matrix, "todense", _boom, raising=True)
    monkeypatch.setattr(sp.coo_matrix, "toarray", _boom, raising=True)
    P = build_transition_graph(latent_2k, k=30, random_state=SEED)
    assert sp.isspmatrix_csr(P)


# ---------- I8 CSR  ----------

def test_csr_hygiene(latent_2k):
    P = build_transition_graph(latent_2k, k=30, random_state=SEED)
    assert sp.isspmatrix_csr(P)
    assert P.has_sorted_indices
    nnz_before = P.nnz
    P2 = P.copy()
    P2.sum_duplicates()
    assert P2.nnz == nnz_before, "I1 "
    assert P.dtype == np.float32
    assert P.indices.dtype in (np.int32, np.int64)


# ---------- I9 hub  [] ----------

def test_degree_bound(latent_2k):
    k = 30
    P = build_transition_graph(latent_2k, k=k, random_state=SEED)
    deg = np.diff(P.indptr)
    # min ≥ k query  k  bug
    assert deg.min() >= k, f" {deg.min()} < k={k}"
    #  ≤ 2k 2 T7 exp1  1.5~1.67
    #  hub  max  2kfixture  max=389@k30
    #  []——protocol I9  ≤k
    assert np.median(deg) <= 2 * k, f" {np.median(deg):.0f} > 2k={2*k}"


# ---------- I5  ----------

def test_absorbing_unit_rows(latent_2k):
    P = build_transition_graph(latent_2k, k=30, random_state=SEED)
    n = P.shape[0]
    rng = np.random.default_rng(SEED)
    abs_idx = np.sort(rng.choice(n, size=50, replace=False))
    Pa = absorb_states(P, abs_idx)
    assert sp.isspmatrix_csr(Pa)
    widths = Pa.indptr[abs_idx + 1] - Pa.indptr[abs_idx]
    assert (widths == 1).all(), " nnz != 1"
    assert (Pa.indices[Pa.indptr[abs_idx]] == abs_idx).all(), ""
    assert (Pa.data[Pa.indptr[abs_idx]] == 1.0).all(), " data != 1.0"
    #
    np.testing.assert_allclose(
        np.asarray(Pa.sum(axis=1)).ravel(), 1.0, rtol=0, atol=ATOL_ROW
    )
    #
    np.testing.assert_allclose(np.asarray(P.sum(axis=1)).ravel(), 1.0, atol=ATOL_ROW)


# ---------- T7 exp1 nnz/(n·k) ----------

def test_nnz_scaling(latent_small, latent_2k):
    k = 30
    for X, tag in [(latent_small, "800"), (latent_2k, "2000")]:
        P = build_transition_graph(X, k=k, kernel="gaussian", random_state=SEED)
        ratio = P.nnz / (X.shape[0] * k)
        # T7 exp1  1.5–1.678  1.39——
        #  [1.3, 1.8] benchmark  nnz
        assert 1.3 <= ratio <= 1.8, f"n={tag} nnz/(n·k)={ratio:.3f}  [1.3, 1.8]"
        Pc = build_transition_graph(X, k=k, kernel="connectivity", random_state=SEED)
        ratio_c = Pc.nnz / (X.shape[0] * k)
        assert 1.3 <= ratio_c <= 1.8, f"connectivity n={tag} nnz/(n·k)={ratio_c:.3f}"


# ---------- I10  ----------

def test_determinism(latent_2k):
    P1 = build_transition_graph(latent_2k, k=30, random_state=SEED)
    P2 = build_transition_graph(latent_2k, k=30, random_state=SEED)
    assert P1.nnz == P2.nnz
    np.testing.assert_array_equal(P1.indptr, P2.indptr)
    np.testing.assert_array_equal(P1.indices, P2.indices)
    np.testing.assert_allclose(P1.data, P2.data, atol=0)


# ----------  ----------

def test_reject_dense_input(latent_small):
    n = latent_small.shape[0]
    dense_square = np.ones((n, n), dtype=np.float32)
    with pytest.raises(ValueError):
        build_transition_graph(dense_square, k=30)  # d=n>256


def test_reject_highdim():
    X = np.zeros((100, MAX_LATENT_DIM + 1), dtype=np.float32)
    with pytest.raises(ValueError, match="latent"):
        build_transition_graph(X, k=10)


# ----------  CSR tracemalloc  ----------

def test_output_single_copy(latent_2k):
    tracemalloc.start()
    build_transition_graph(latent_2k, k=30, random_state=SEED)  #
    snapshot_before = tracemalloc.take_snapshot()
    P = build_transition_graph(latent_2k, k=30, random_state=SEED)
    snapshot_after = tracemalloc.take_snapshot()
    diff = snapshot_after.compare_to(snapshot_before, "filename")
    leaked = sum(s.size_diff for s in diff)
    tracemalloc.stop()
    #  = nnz*(4B data + 4B idx) + (n+1)*8B indptr
    n = P.shape[0]
    expected = P.nnz * 8 + (n + 1) * 8
    #  <1.5×  Python  2×
    assert leaked < 1.5 * expected, (
        f"tracemalloc  {leaked/1e6:.1f}MB vs  {expected/1e6:.1f}MB"
    )


# ---------- 1 <4GB ----------

def test_mem_smoke_10k(latent_10k):
    tracemalloc.start()
    P = build_transition_graph(latent_10k, k=30, random_state=SEED)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    n = P.shape[0]
    dense_n2_bytes = n * n * 8
    assert peak < 0.5 * dense_n2_bytes, " n×n I7 "
    assert peak < 4 * (1 << 30), f"tracemalloc  {peak/1e9:.2f}GB  4GB "


# ---------- fixture_5k  ----------

def test_on_fixture5k_invariants(fixture5k):
    X = np.asarray(fixture5k.obsm["X_scVI"])
    assert X.ndim == 2 and X.shape[1] <= MAX_LATENT_DIM
    k = 30
    P = build_transition_graph(X, k=k, kernel="gaussian", random_state=SEED)
    # I1 / I2 / I8 / I9 / I10
    np.testing.assert_allclose(np.asarray(P.sum(axis=1)).ravel(), 1.0, rtol=0, atol=ATOL_ROW)
    assert P.data.min() >= 0.0
    assert sp.isspmatrix_csr(P) and P.has_sorted_indices and P.dtype == np.float32
    P2 = P.copy()
    P2.sum_duplicates()
    assert P2.nnz == P.nnz
    deg = np.diff(P.indptr)
    assert deg.min() >= k and np.median(deg) <= 2 * k  # I9
    P1b = build_transition_graph(X, k=k, kernel="gaussian", random_state=SEED)
    np.testing.assert_array_equal(P.indptr, P1b.indptr)
    np.testing.assert_array_equal(P.indices, P1b.indices)
    # nnz  T7 exp1 [1.3, 1.8]fixture  1.67
    ratio = P.nnz / (X.shape[0] * k)
    assert 1.3 <= ratio <= 1.8, f"fixture nnz/(n·k)={ratio:.3f}"
    # pynndescent  A5
