#!/usr/bin/env python
"""Layer B2 concordance audit - Step 1: CellRank krylov-Schur on sfate's production P (75k).

Fills the missing quadrant of the 75k 2x2: {sfate kernel} x {GPCCA estimator}.
Input : output/c_layer_graph.npz (same cache used by m3_solve_75k_columns.py)
Output: output/audit_b2_step1_macrostates.npz (memberships + macro names, for Step 2)
        output/audit_b2_step1_log.md

Run in the SAME env as benchmark_slepc_krylov.py (must pass _is_petsc_slepc_available):
  source ~/sfate_env/bin/activate && OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 /usr/bin/time -v python scripts/audit_b2_cellrank_on_sfate_P.py \
    2> output/audit_b2_step1_time_v.txt
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import scipy.sparse as sp

GRAPH_CACHE = Path("output/c_layer_graph.npz")
OUT_NPZ = Path("output/audit_b2_step1_macrostates.npz")
OUT_LOG = Path("output/audit_b2_step1_log.md")

EXPECTED_NNZ = 3_807_850  # Table S3, real_75k_full, seed 20260909
N_COMPONENTS = 20          # same convention as benchmark_slepc_krylov.py
N_STATES = 6


def _rss_sampler(stop: threading.Event, bucket: dict) -> None:
    proc = psutil.Process()
    while not stop.is_set():
        bucket["peak_mb"] = max(bucket["peak_mb"], proc.memory_info().rss / 1e6)
        stop.wait(0.25)


def main() -> None:
    t_start = time.perf_counter()
    stop = threading.Event()
    rss = {"peak_mb": 0.0}
    th = threading.Thread(target=_rss_sampler, args=(stop, rss), daemon=True)
    th.start()

    # --- load sfate production P (identical object to m3's solve) ---
    z = np.load(GRAPH_CACHE, allow_pickle=True)
    P = sp.csr_matrix(
        (z["P_data"], z["P_indices"], z["P_indptr"]), shape=tuple(z["P_shape"])
    ).astype(np.float32)
    labels = z["labels"]
    keep = z["keep"].astype(bool)

    # --- structural gate: must be the Table S3 production matrix ---
    assert P.shape == (74_984, 74_984), f"P shape {P.shape}"
    assert P.dtype == np.float32, f"P dtype {P.dtype}"
    assert bool(keep.all()), "keep must be all-True for the 75k run"
    rs = np.asarray(P.sum(axis=1)).ravel()
    rs_dev = float(np.abs(rs - 1.0).max())
    assert rs_dev < 1e-5, f"P not row-stochastic, dev={rs_dev:.2e}"
    nnz_dev = abs(P.nnz - EXPECTED_NNZ) / EXPECTED_NNZ
    assert nnz_dev < 0.01, f"nnz {P.nnz} deviates {nnz_dev:.2%} from {EXPECTED_NNZ}"

    # --- CellRank GPCCA, krylov-Schur; fails loudly if PETSc absent ---
    import cellrank as cr
    import petsc4py
    import slepc4py
    from cellrank.estimators import GPCCA
    from cellrank.estimators.mixins.decomposition._schur import _is_petsc_slepc_available

    assert bool(_is_petsc_slepc_available()), "PETSc/SLEPc unavailable in this env"

    pk = cr.kernels.PrecomputedKernel(P)
    g = GPCCA(pk)
    g.compute_schur(n_components=N_COMPONENTS, method="krylov")
    g.compute_macrostates(n_states=N_STATES, cluster_key=None)

    wall = time.perf_counter() - t_start
    stop.set()
    th.join(timeout=2.0)

    memb = np.asarray(g.macrostates_memberships, dtype=np.float32)
    macro_names = [str(m) for m in g.macrostates_memberships.names]
    np.savez(
        OUT_NPZ,
        memberships=memb,                       # (74984, 6) macrostate memberships
        macro_names=np.array(macro_names),
        labels=np.asarray(labels),
    )

    log = {
        "cellrank": cr.__version__,
        "petsc4py": petsc4py.__version__,
        "slepc4py": slepc4py.__version__,
        "nnz": int(P.nnz), "nnz_expected": EXPECTED_NNZ, "nnz_dev": nnz_dev,
        "row_stochastic_dev": rs_dev,
        "schur_method": "krylov (explicit)",
        "n_components": N_COMPONENTS,
        "n_macrostates": len(macro_names),
        "macro_names": macro_names,
        "peak_rss_gib": round(rss["peak_mb"] / 1024, 2),
        "wall_s": round(wall, 1),
    }
    OUT_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"DONE macros={macro_names} peak={log['peak_rss_gib']}GiB wall={wall:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
