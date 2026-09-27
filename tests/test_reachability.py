"""Tests for the directed-reachability precondition (v1.1.0)."""
import numpy as np
import pytest
import scipy.sparse as sp

from sfate.reachability import check_absorption_precondition


def _chain(rows):
    return sp.csr_matrix(rows, dtype=np.float64)


def test_passes_when_all_transients_reach_absorbing():
    # 0 -> absorb, 1 -> 0, 2 -> 1, so 2 reaches absorb via 1 -> 0
    P = _chain([
        [1.0, 0.0, 0.0],   # 0 absorbing
        [1.0, 0.0, 0.0],   # 1 -> 0
        [0.0, 1.0, 0.0],   # 2 -> 1
    ])
    out = check_absorption_precondition(P, absorbing_cells=[0])
    assert out["rho_q_lt_1"] is True
    assert out["n_unreachable"] == 0


def test_fails_on_closed_transient_class():
    # 0 absorbing; 1 <-> 2 closed transient 2-cycle, cannot reach 0
    P = _chain([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ])
    with pytest.raises(ValueError, match="Absorption precondition violated"):
        check_absorption_precondition(P, absorbing_cells=[0])


def test_returns_diagnostics_without_raising_when_requested():
    P = _chain([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ])
    out = check_absorption_precondition(P, absorbing_cells=[0], raise_on_fail=False)
    assert out["rho_q_lt_1"] is False
    assert out["n_unreachable"] == 2
    assert set(out["unreachable_indices"].tolist()) == {1, 2}
