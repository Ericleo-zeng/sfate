#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
profile_cellrank.py — CellRank scalable-fate


----
 CellRank  /  /  nnz
 50~100+  OOM  scalable-fate
benchmark docs/validation-protocol.md

 conda/venv
----------------------------------------
    pip install "cellrank>=2.3" scanpy anndata numpy scipy psutil
    #
    pip install memory-profiler
    # PETSc  gmres
    pip install petsc petsc4py slepc slepc4py


--------
    # 5k  kernel
    python profile_cellrank.py --n-cells 5000 --stage kernel

    # kernel + GPCCA + fate probabilities CSV
    python profile_cellrank.py --n-cells 10000 30000 50000 \
        --stage all --n-states 6 --output cellrank_profile.csv

    #  RSS tracemalloc
    python profile_cellrank.py --n-cells 50000 --stage all --no-tracemalloc

 CSV
---------------
    stage            : latent_gen / knn_graph / kernel_tmat / gpcca_fit /
                       fate_probs / total
    n_cells          :
    k_neighbors      : kNN
    latent_dim       : latent
    nnz              :  NaN
    tmat_format      : csr/coo/dense/...
    elapsed_sec      :  wall time
    peak_rss_mb      :  RSS MBpsutil
    tracemalloc_mb   :  Python MB--no-tracemalloc  NaN
    notes            :


----
- peak_rss_mb " -  RSS"
   stage==total
- GPCCA  Schur  O(n^2) float64n x n  8*n^2
  n>=20000
- /CI  550 >=64GB /
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field

import numpy as np

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

# ---------------------------------------------------------------------------
# RSS  RSS
# ---------------------------------------------------------------------------


class RssSampler:
    """ RSSMBpeak """

    def __init__(self, interval: float = 0.05) -> None:
        self.interval = interval
        self._proc = psutil.Process(os.getpid()) if psutil else None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak: float = float("nan")

    def _rss_mb(self) -> float:
        if self._proc is None:
            return float("nan")
        return self._proc.memory_info().rss / 1e6

    def _run(self) -> None:
        peaks = []
        while not self._stop.is_set():
            peaks.append(self._rss_mb())
            self._stop.wait(self.interval)
        if peaks:
            self.peak = max(peaks)

    def __enter__(self) -> "RssSampler":
        if self._proc is not None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)


def rss_now_mb() -> float:
    if psutil is None:
        return float("nan")
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


# ---------------------------------------------------------------------------
# /
# ---------------------------------------------------------------------------


@dataclass
class StageRecord:
    stage: str
    n_cells: int
    k_neighbors: int
    latent_dim: int
    nnz: float = float("nan")
    tmat_format: str = ""
    elapsed_sec: float = float("nan")
    peak_rss_mb: float = float("nan")
    tracemalloc_mb: float = float("nan")
    notes: str = ""


@dataclass
class Profiler:
    """ wall time + RSS  + tracemalloc """

    n_cells: int
    k_neighbors: int
    latent_dim: int
    use_tracemalloc: bool = True
    records: list[StageRecord] = field(default_factory=list)

    def run_stage(self, stage: str, fn, notes: str = ""):
        """ fn() fn """
        import tracemalloc

        gc.collect()
        rss_start = rss_now_mb()
        if self.use_tracemalloc:
            tracemalloc.start()
        t0 = time.perf_counter()
        err = ""
        with RssSampler() as sampler:
            try:
                result = fn()
            except MemoryError:
                err = "MemoryError"
                result = None
            except Exception:  # noqa: BLE001 -
                err = traceback.format_exc(limit=3).replace("\n", " | ")[-400:]
                result = None
        elapsed = time.perf_counter() - t0
        if self.use_tracemalloc:
            _, tm_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            tm_mb = tm_peak / 1e6
        else:
            tm_mb = float("nan")

        rec = StageRecord(
            stage=stage,
            n_cells=self.n_cells,
            k_neighbors=self.k_neighbors,
            latent_dim=self.latent_dim,
            elapsed_sec=round(elapsed, 3),
            peak_rss_mb=round(sampler.peak - rss_start, 1)
            if not math.isnan(sampler.peak)
            else float("nan"),
            tracemalloc_mb=round(tm_mb, 1),
            notes="; ".join(x for x in [notes, err] if x),
        )
        self.records.append(rec)
        print(
            f"[{stage:>12s}] n={self.n_cells:<7d} k={self.k_neighbors:<3d} "
            f"{elapsed:8.2f}s  RSS+{rec.peak_rss_mb:9.1f}MB  tm={rec.tracemalloc_mb:9.1f}MB"
            + (f"  !! {err[:120]}" if err else ""),
            flush=True,
        )
        return result

    def finalize_last(self, nnz=None, tmat_format=None) -> None:
        if nnz is not None:
            self.records[-1].nnz = float(nnz)
        if tmat_format:
            self.records[-1].tmat_format = str(tmat_format)


# ---------------------------------------------------------------------------
#  latent
# ---------------------------------------------------------------------------


def make_synthetic_latent(
    n_cells: int, latent_dim: int, n_clusters: int, seed: int = 42
) -> np.ndarray:
    """ latent + n_clusters """
    rng = np.random.default_rng(seed)
    #
    t = rng.uniform(0, 1, n_cells)
    trunk = np.zeros((n_cells, latent_dim), dtype=np.float32)
    trunk[:, 0] = 3.0 * t
    trunk[:, 1:] = rng.normal(0, 0.4, (n_cells, latent_dim - 1))
    # t > 0.6
    branch_id = (t > 0.6).astype(int)
    if branch_id.any():
        labels = rng.integers(0, n_clusters, branch_id.sum())
        centers = rng.normal(0, 2.0, (n_clusters, latent_dim))
        centers[:, 0] = 3.0  #
        idx = np.where(branch_id)[0]
        trunk[idx] += (centers[labels] * (t[idx, None] - 0.6) / 0.4).astype(np.float32)
    return trunk


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------


def profile_one_size(
    n_cells: int,
    k: int,
    latent_dim: int,
    n_clusters: int,
    stage: str,
    n_states: int,
    solver: str,
    use_petsc: bool,
    use_tracemalloc: bool,
    seed: int,
) -> list[StageRecord]:
    import anndata as ad
    import cellrank as cr
    import scanpy as sc
    from cellrank.estimators import GPCCA
    from cellrank.kernels import ConnectivityKernel

    pf = Profiler(n_cells, k, latent_dim, use_tracemalloc=use_tracemalloc)

    # 1)  latent
    latent = pf.run_stage(
        "latent_gen",
        lambda: make_synthetic_latent(n_cells, latent_dim, n_clusters, seed),
    )
    if latent is None:
        return pf.records

    adata = ad.AnnData(X=latent)
    adata.obsm["X_latent"] = latent

    # 2) kNN scanpy latent
    def _knn():
        sc.pp.neighbors(adata, n_neighbors=k, use_rep="X_latent")
        conn = adata.obsp["connectivities"]
        return conn

    conn = pf.run_stage("knn_graph", _knn)
    if conn is not None:
        pf.finalize_last(nnz=conn.nnz, tmat_format=type(conn).__name__)
    else:
        return pf.records

    # 3) ConnectivityKernel
    if stage in ("kernel", "all"):
        vk = ConnectivityKernel(adata)

        def _tmat():
            vk.compute_transition_matrix(density_normalize=True)
            return vk.transition_matrix

        tmat = pf.run_stage("kernel_tmat", _tmat)
        if tmat is not None:
            import scipy.sparse as sp

            pf.finalize_last(
                nnz=tmat.nnz if sp.issparse(tmat) else tmat.size,
                tmat_format=tmat.format if sp.issparse(tmat) else "dense",
            )
            #  AnnData
            obsp_keys = [key for key in adata.obsp.keys() if "T_" in key or "transition" in key.lower()]
            pf.records[-1].notes += f"; obsp_keys={obsp_keys}"
        else:
            return pf.records

    # 4) GPCCA fitSchur  +  +
    if stage in ("gpcca", "all"):
        g = GPCCA(vk)

        def _fit():
            g.fit(n_states=n_states)
            return g

        g = pf.run_stage("gpcca_fit", _fit, notes=f"n_states={n_states}")
        if g is None:
            return pf.records

        # 5) fate
        def _fate():
            g.compute_fate_probabilities(solver=solver, use_petsc=use_petsc)
            return g.fate_probabilities

        fp = pf.run_stage(
            "fate_probs", _fate, notes=f"solver={solver},use_petsc={use_petsc}"
        )
        if fp is not None:
            pf.finalize_last(
                nnz=float(np.count_nonzero(np.asarray(fp))),
                tmat_format=f"dense Lineage {fp.shape}",
            )

    #
    total = StageRecord(
        stage="total",
        n_cells=n_cells,
        k_neighbors=k,
        latent_dim=latent_dim,
        peak_rss_mb=round(rss_now_mb(), 1),
        notes=" RSS",
    )
    pf.records.append(total)
    return pf.records


def main() -> None:
    ap = argparse.ArgumentParser(
        description="CellRank scalable-fate ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--n-cells", type=int, nargs="+", default=[5000, 20000, 50000],
                    help="/CI  5000050 >=64GB ")
    ap.add_argument("--k-neighbors", type=int, default=30, help="kNN ")
    ap.add_argument("--latent-dim", type=int, default=32, help=" latent ")
    ap.add_argument("--n-clusters", type=int, default=8, help="")
    ap.add_argument("--stage", choices=["knn", "kernel", "gpcca", "all"], default="all",
                    help="gpcca/all  Schur  fate ")
    ap.add_argument("--n-states", type=int, default=6, help="GPCCA ")
    ap.add_argument("--solver", default="gmres",
                    choices=["direct", "gmres", "lgmres", "bicgstab", "gcrotmk"],
                    help="fate probabilities ")
    ap.add_argument("--no-petsc", action="store_true", help=" PETSc ")
    ap.add_argument("--no-tracemalloc", action="store_true",
                    help=" tracemalloc RSS")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="cellrank_profile.csv", help=" CSV ")
    args = ap.parse_args()

    #
    import cellrank as cr
    import scanpy as sc
    import scipy
    import datetime

    env = {
        "date": datetime.date.today().isoformat(),
        "cellrank": cr.__version__,
        "scanpy": sc.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": sys.version.split()[0],
    }
    print(":", json.dumps(env, ensure_ascii=False), flush=True)

    all_records: list[StageRecord] = []
    for n in args.n_cells:
        if n > 50000:
            print(f"!! : n={n}  5", flush=True)
        all_records.extend(
            profile_one_size(
                n_cells=n,
                k=args.k_neighbors,
                latent_dim=args.latent_dim,
                n_clusters=args.n_clusters,
                stage=args.stage,
                n_states=args.n_states,
                solver=args.solver,
                use_petsc=not args.no_petsc,
                use_tracemalloc=not args.no_tracemalloc,
                seed=args.seed,
            )
        )
        gc.collect()

    #  CSV
    import csv

    fields = [
        "stage", "n_cells", "k_neighbors", "latent_dim", "nnz", "tmat_format",
        "elapsed_sec", "peak_rss_mb", "tracemalloc_mb", "notes",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_records:
            w.writerow({k_: getattr(r, k_) for k_ in fields})
    print(f" {args.output}{len(all_records)} ", flush=True)


if __name__ == "__main__":
    main()
