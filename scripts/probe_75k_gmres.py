#!/usr/bin/env python
""" 75k GMRES+ILU ILU  vs GMRES """

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, gmres, spilu

from c_layer_validation import select_terminals, ANNOTATED, K, SEED, CACHE_GRAPH

t0 = time.perf_counter()
if CACHE_GRAPH.exists():
    z = np.load(CACHE_GRAPH, allow_pickle=True)
    P = sp.csr_matrix((z["P_data"], z["P_indices"], z["P_indptr"]), shape=tuple(z["P_shape"]))
    labels = z["labels"]
else:
    from sfate import build_transition_graph
    from sfate.io import load_latent, load_obs_categorical
    X = load_latent(ANNOTATED, "X_scVI")
    states = np.asarray(load_obs_categorical(ANNOTATED, "microglia_state").astype(str))
    P = build_transition_graph(X, k=K, random_state=SEED)
    labels = select_terminals(X, states)
    import scipy.sparse.csgraph as csgraph
    valid = np.array([x is not None for x in labels], dtype=bool)
    n_comp, comp = csgraph.connected_components(P, directed=False)
    keep = np.isin(comp, np.unique(comp[valid]))
    print(f" {n_comp} {(~keep).sum()}")
    P = P[keep][:, keep].tocsr()
    labels = labels[keep]
print(f" {time.perf_counter()-t0:.1f}s: n={P.shape[0]}, nnz={P.nnz}", flush=True)

from sfate.graph import absorb_states, split_transient_absorbing
abs_idx = np.nonzero(np.array([x is not None for x in labels], dtype=bool))[0]
Pa = absorb_states(P, abs_idx)
Q, R, t_idx, _ = split_transient_absorbing(Pa, abs_idx)
n_t = Q.shape[0]
A = (sp.eye(n_t, format="csr", dtype=np.float32) - Q).tocsr()
print(f"n_transient={n_t}, A nnz={A.nnz}", flush=True)

# ILU(fill=1)
t0 = time.perf_counter()
try:
    ilu = spilu(A.tocsc(), fill_factor=1.0, drop_tol=0.0, options={"ILU_MILU": "SMILU_1"})
    print(f"spilu(fill=1): {time.perf_counter()-t0:.1f}s, L+U nnz={ilu.L.nnz + ilu.U.nnz}", flush=True)
except Exception as e:
    print(f"spilu(fill=1) : {e}", flush=True)
    ilu = None

# ILU(fill=10)
t0 = time.perf_counter()
try:
    ilu10 = spilu(A.tocsc(), fill_factor=10.0)
    print(f"spilu(fill=10): {time.perf_counter()-t0:.1f}s, L+U nnz={ilu10.L.nnz + ilu10.U.nnz}", flush=True)
except Exception as e:
    print(f"spilu(fill=10) : {e}", flush=True)
    ilu10 = None

#  GMRES cap maxiter=200
b = np.asarray(R[:, 0].todense()).ravel().astype(np.float32)  #  src
for name, M_ in [("fill=1", ilu), ("fill=10", ilu10), ("", None)]:
    if M_ is not None:
        M = LinearOperator((n_t, n_t), matvec=M_.solve, dtype=np.float32)
    else:
        M = None
    it = [0]
    t0 = time.perf_counter()
    x, info = gmres(A, b, rtol=1e-6, atol=0.0, restart=50, maxiter=200, M=M,
                    callback=lambda rk: it.__setitem__(0, it[0] + 1), callback_type="pr_norm")
    dt = time.perf_counter() - t0
    resid = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
    print(f"GMRES[{name}]: info={info} iters={it[0]} {dt:.1f}s ({dt/max(it[0],1)*1000:.0f}ms/it) rel_resid={resid:.2e}", flush=True)
