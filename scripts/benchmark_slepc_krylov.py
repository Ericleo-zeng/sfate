#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_slepc_krylov.py —  #1CellRank + SLEPc krylov-Schur 100k

 paper/PLAN.md  #1
- petsc4py/slepc4py  rfd3_envoverlay
- tests/conftest.py  latentmake_branching_latentd=30
  n_terminal=6seed=20260909n=100,000——GPCCA
  M1  5  _synthetic macrostates/fate
  k=30 kNN  nnz  ≈1.5–1.7·n·k
- scanpy neighbors → ConnectivityKernel → GPCCA.compute_schur(method='krylov')
  → macrostates → terminal_states → fate(gmres+ilu, use_petsc=False C )
-  RSS (250ms) + tracemalloc
  /usr/bin/time -v  output/slepc_krylov_100k_time_v.txt
- BLAS=1 output/
"""

from __future__ import annotations

import json
import os
import platform
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import psutil

SEED = 20260909
N = int(os.environ.get("SLEPC_N", "100000"))  #  SLEPC_N=5000
K = 30
D = 30
N_COMPONENTS = 20
N_STATES = 6
OUT_JSON = Path(f"output/slepc_krylov_{N // 1000}k.json")


def _branching(n: int, d: int = D) -> np.ndarray:
    """tests/conftest.py  latentn_terminal=6  6 """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
    from conftest import make_branching_latent

    return make_branching_latent(n=n, d=d, n_terminal=6, seed=SEED)


class StageMeter:
    """ wall time + RSS + tracemalloc """

    def __init__(self) -> None:
        self.records: list[dict] = []

    def run(self, stage: str, fn):
        import gc
        import tracemalloc

        gc.collect()
        proc = psutil.Process()
        rss_start = proc.memory_info().rss / 1e6
        stop = threading.Event()
        peak = {"mb": rss_start}

        def _sample():
            while not stop.is_set():
                peak["mb"] = max(peak["mb"], proc.memory_info().rss / 1e6)
                stop.wait(0.25)

        th = threading.Thread(target=_sample, daemon=True)
        th.start()
        tracemalloc.start()
        t0 = time.perf_counter()
        err = ""
        try:
            result = fn()
        except Exception:  # noqa: BLE001 —
            err = traceback.format_exc(limit=3).replace("\n", " | ")[-500:]
            result = None
        elapsed = time.perf_counter() - t0
        _, tm_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        stop.set()
        th.join(timeout=2.0)
        rec = {
            "stage": stage,
            "elapsed_s": round(elapsed, 3),
            "rss_peak_mb": round(peak["mb"], 1),
            "rss_delta_mb": round(peak["mb"] - rss_start, 1),
            "tracemalloc_mb": round(tm_peak / 1e6, 1),
            "error": err,
        }
        self.records.append(rec)
        print(
            f"[{stage:>16s}] {elapsed:9.2f}s  RSS_peak={peak['mb']:9.1f}MB "
            f"(+{rec['rss_delta_mb']:9.1f})  tm={rec['tracemalloc_mb']:8.1f}MB"
            + (f"  !! {err[:150]}" if err else ""),
            flush=True,
        )
        return result


def main() -> int:
    import anndata as ad
    import cellrank as cr
    import scanpy as sc
    import scipy
    from cellrank.estimators import GPCCA
    from cellrank.estimators.mixins.decomposition._schur import _is_petsc_slepc_available
    from cellrank.kernels import ConnectivityKernel

    import petsc4py
    import slepc4py
    from petsc4py import PETSc

    env = {
        "date": time.strftime("%Y-%m-%d"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "glibc": platform.libc_ver()[1],
        "cellrank": cr.__version__,
        "pygpcca": __import__("pygpcca").__version__,
        "scanpy": sc.__version__,
        "anndata": ad.__version__,
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "petsc4py": petsc4py.__version__,
        "slepc4py": slepc4py.__version__,
        "petsc_lib": ".".join(map(str, PETSc.Sys.getVersion())),
        "petsc4py_location": petsc4py.__file__,
        "slepc4py_location": slepc4py.__file__,
        "is_petsc_slepc_available": bool(_is_petsc_slepc_available()),
        "blas_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "seed": SEED,
    }
    print("env:", json.dumps(env, ensure_ascii=False), flush=True)
    assert env["is_petsc_slepc_available"], "SLEPc  krylov "

    meter = StageMeter()

    # 1)  latent
    X = meter.run("latent_gen", lambda: _branching(N))
    adata = ad.AnnData(X=np.zeros((N, 1), dtype=np.float32))
    adata.obsm["X_latent"] = X

    # 2) kNN
    conn = meter.run(
        "neighbors",
        lambda: sc.pp.neighbors(adata, n_neighbors=K, use_rep="X_latent", random_state=SEED)
        or adata.obsp["connectivities"],
    )
    nnz = int(conn.nnz)

    # 3) ConnectivityKernel
    def _kernel():
        ck = ConnectivityKernel(adata)
        ck.compute_transition_matrix(density_normalize=True)
        return ck

    ck = meter.run("kernel", _kernel)
    meter.records[-1]["nnz"] = int(ck.transition_matrix.nnz)

    # 4) krylov-Schur
    g = GPCCA(ck)

    def _schur():
        g.compute_schur(n_components=N_COMPONENTS, method="krylov")
        return g

    meter.run("schur_krylov", _schur, )
    schur_info = {
        "method_requested": "krylov",
        "n_components": N_COMPONENTS,
        "schur_vectors_shape": list(getattr(g, "schur_vectors", np.zeros(0)).shape)
        if getattr(g, "schur_vectors", None) is not None else None,
        "n_eigenvalues": int(len(g.eigenvalues)) if getattr(g, "eigenvalues", None) is not None else None,
    }
    print("schur_info:", json.dumps(schur_info), flush=True)

    # 5)  +  + fate
    meter.run("macrostates", lambda: g.compute_macrostates(n_states=N_STATES))
    macro = list(g.macrostates) if getattr(g, "macrostates", None) is not None else []

    def _terminal():
        g.predict_terminal_states(method="stability", stability_threshold=0.96)
        return g

    meter.run("terminal_states", _terminal)
    term = list(g.terminal_states) if getattr(g, "terminal_states", None) is not None else []

    fp = meter.run(
        "fate_probs",
        lambda: g.compute_fate_probabilities(
            solver="gmres", use_petsc=False, preconditioner="ilu"
        ),
    )

    out = {
        "env": env,
        "config": {
            "n": N, "k": K, "latent_dim": D,
            "n_components": N_COMPONENTS, "n_states": N_STATES,
            "fate_solver": "gmres+ilu, use_petsc=False C /step12 ",
        },
        "graph": {"nnz_connectivities": nnz, "nnz_transition": int(ck.transition_matrix.nnz)},
        "schur_info": schur_info,
        "macrostates": [str(m) for m in macro],
        "terminal_states": [str(t) for t in term],
        "fate_shape": list(fp.shape) if fp is not None else None,
        "stages": meter.records,
        "process_peak_rss_mb": round(psutil.Process().memory_info().rss / 1e6, 1),
        "note": " output/slepc_krylov_100k_time_v.txt  Maximum resident set size ",
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f" {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
