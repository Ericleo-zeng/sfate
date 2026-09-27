"""Directed-reachability precondition for target-conditioned absorption.

Mathematical guarantee
----------------------
For the absorbing chain P_abs = [[Q, R], [0, I]], the system (I - Q) X = R
has the unique absorption-probability solution iff rho(Q) < 1, equivalently
iff (I - Q) is a nonsingular M-matrix. The necessary and sufficient graph
condition is: from EVERY transient state, at least one absorbing state is
reachable via a directed path of P_abs. This module checks exactly that
condition in O(n + nnz) by DFS on the reversed graph from all absorbing
states.

Why the existing guard is strictly weaker
-----------------------------------------
The pipeline currently guards only with
scipy.sparse.csgraph.connected_components(directed=False): each UNDIRECTED
component must contain an absorbing cell. A closed transient class (e.g. a
2-cycle among transient cells) can share an undirected component with
absorbing cells while being unreachable from them in the directed sense,
leaving (I - Q) singular. The check below subsumes that guard.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

__all__ = ["check_absorption_precondition"]


def check_absorption_precondition(P_abs, absorbing_cells, *, raise_on_fail=True):
    """Verify rho(Q) < 1 for the absorbing chain P_abs.

    Parameters
    ----------
    P_abs
        (n, n) CSR absorbing chain: absorbing rows are one-hot unit rows;
        transient rows are row-stochastic over their out-edges (some of
        which lead to absorbing states).
    absorbing_cells : array-like of int
        Indices of the absorbing states (representative cells).
    raise_on_fail : bool
        If True (default), raise ValueError when unreachable transient
        states exist. If False, return diagnostics only.

    Returns
    -------
    dict with keys n_transient, n_unreachable, unreachable_indices,
    rho_q_lt_1 (True iff n_unreachable == 0; guaranteed precondition).

    Raises
    ------
    ValueError
        If unreachable transient states exist and raise_on_fail=True, or if
        the absorbing rows are not one-hot unit rows.
    """
    P_abs = sp.csr_matrix(P_abs)
    if P_abs.shape[0] != P_abs.shape[1]:
        raise ValueError("P_abs must be square")
    n = P_abs.shape[0]
    absorbing = np.unique(np.asarray(absorbing_cells, dtype=np.int64).ravel())
    if absorbing.size == 0:
        raise ValueError("at least one absorbing state is required")

    rowsum = np.asarray(P_abs[absorbing].sum(axis=1)).ravel()
    if not np.allclose(rowsum, 1.0, atol=1e-6):
        raise ValueError("absorbing rows of P_abs must be one-hot unit rows")

    # DFS on the REVERSED graph from all absorbing states:
    # PT[v, u] > 0  <=>  original edge u -> v, so walking PT from an
    # absorbing state visits exactly the states that can reach it.
    PT = P_abs.T.tocsr()
    indptr, indices = PT.indptr, PT.indices
    visited = np.zeros(n, dtype=bool)
    visited[absorbing] = True
    stack = [int(a) for a in absorbing]
    while stack:
        v = stack.pop()
        for u in indices[indptr[v]:indptr[v + 1]]:
            if not visited[u]:
                visited[u] = True
                stack.append(int(u))

    transient_mask = np.ones(n, dtype=bool)
    transient_mask[absorbing] = False
    unreachable = np.nonzero(transient_mask & ~visited)[0]

    result = {
        "n_transient": int(transient_mask.sum()),
        "n_unreachable": int(unreachable.size),
        "unreachable_indices": unreachable,
        "rho_q_lt_1": bool(unreachable.size == 0),
    }
    if unreachable.size and raise_on_fail:
        raise ValueError(
            f"Absorption precondition violated: {unreachable.size} of "
            f"{result['n_transient']} transient states cannot reach any "
            f"absorbing state (rho(Q) = 1; I - Q singular). First affected "
            f"cell indices: {unreachable[:10].tolist()}. Add/adjust target "
            f"representatives or check the kernel so every transient state "
            f"has a directed path to an absorbing state."
        )
    return result
