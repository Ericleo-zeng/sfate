#!/usr/bin/env python
"""M1 smoke benchmark5k  / 75k  / 1·50

tracemallocPython +  RSS /usr/bin/time -v
validation-protocol §4BLAS =1≥3  maxswap  gc

 /usr/bin/time -v  RSS
  source ~/sfate_env/bin/activate && OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /usr/bin/time -v python scripts/benchmark_m1_smoke.py 2> output/benchmark_time_v.txt
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import tracemalloc
from pathlib import Path

import numpy as np
import psutil

from sfate.graph import build_transition_graph
from sfate.io import load_latent

SEED = 20260909
REPS = 3
REPORT = Path("output/benchmark_m1_smoke.md")
TIME_V = Path("output/benchmark_time_v.txt")
ANNOTATED = (
    "<data>/"
    "06_annotation/adata_annotated.h5ad"
)


def _synthetic(n: int, d: int = 30) -> np.ndarray:
    """ 2 validation-protocol §3.1∝"""
    rng = np.random.default_rng(SEED)
    n_cl = 5
    per = n // n_cl
    centers = rng.normal(0, 3.0, size=(n_cl, d))
    parts = []
    for j in range(n_cl):
        nj = per + (1 if j < n % n_cl else 0)
        parts.append(centers[j] + rng.normal(0, 1.0, size=(nj, d)))
    return np.concatenate(parts, 0).astype(np.float32)


def _rss_sampler(stop: threading.Event, out: dict, interval: float = 0.25) -> None:
    """RSS 250ms  RSS protocol §4 """
    proc = psutil.Process()
    while not stop.is_set():
        out["peak_mb"] = max(out.get("peak_mb", 0.0), proc.memory_info().rss / 1e6)
        stop.wait(interval)


def bench_one(name: str, X: np.ndarray, k: int = 30, reps: int = REPS) -> dict:
    """ + tracemalloc  + RSS reps  max"""
    rows = {"name": name, "n": int(X.shape[0]), "d": int(X.shape[1]), "k": k}
    #  JIT/
    build_transition_graph(X, k=k, random_state=SEED, knn_backend="pynndescent")
    tracemalloc.start()
    tracemalloc.stop()
    best = {"build_s": -1.0, "peak_mb": -1.0, "rss_mb": -1.0}
    for _ in range(reps):
        rss: dict = {}
        stop = threading.Event()
        th = threading.Thread(target=_rss_sampler, args=(stop, rss))
        th.start()
        tracemalloc.start()
        t0 = time.perf_counter()
        P = build_transition_graph(X, k=k, random_state=SEED, knn_backend="pynndescent")
        build_s = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        stop.set()
        th.join()
        best["build_s"] = max(best["build_s"], build_s)
        best["peak_mb"] = max(best["peak_mb"], peak / 1e6)
        best["rss_mb"] = max(best["rss_mb"], rss.get("peak_mb", 0.0))
        del P
    rows["build_s_max"] = round(best["build_s"], 3)
    rows["tracemalloc_peak_mb_max"] = round(best["peak_mb"], 1)
    rows["rss_peak_mb_max"] = round(best["rss_mb"], 1)
    # nnz
    P = build_transition_graph(X, k=k, random_state=SEED, knn_backend="pynndescent")
    rows["nnz"] = int(P.nnz)
    rows["nnz_per_nk"] = round(P.nnz / (X.shape[0] * k), 4)
    rows["csr_theory_mb"] = round((P.nnz * 8 + (X.shape[0] + 1) * 8) / 1e6, 2)
    del P
    return rows


def backend_micro_compare(X: np.ndarray, k: int = 30) -> list[dict]:
    """kNN P1 pynndescent vs sklearn"""
    out = []
    for backend in ["pynndescent", "sklearn"]:
        best = -1.0
        for _ in range(REPS):
            t0 = time.perf_counter()
            P = build_transition_graph(X, k=k, random_state=SEED, knn_backend=backend)
            best = max(best, time.perf_counter() - t0)
            del P
        out.append({"backend": backend, "n": int(X.shape[0]), "build_s_max": round(best, 3)})
    return out


def main() -> int:
    import platform

    import anndata as ad
    import scipy

    results: list[dict] = []
    backend_rows: list[dict] = []

    # --- 5k fixture ---
    fx = ad.read_h5ad("output/fixture_5k.h5ad")
    results.append(bench_one("fixture_5k_real", np.asarray(fx.obsm["X_scVI"])))
    backend_rows += backend_micro_compare(np.asarray(fx.obsm["X_scVI"]))
    del fx

    # ---  1 ---
    X10k = _synthetic(10_000)
    results.append(bench_one("synthetic_10k", X10k))
    backend_rows += backend_micro_compare(X10k)
    del X10k

    # --- 75k h5py  X_scVI anndata backed  layers  6.9GB  ---
    X75k = load_latent(ANNOTATED)  # (74984, 30) float32 ≈ 9MB
    results.append(bench_one("real_75k_full", X75k))
    backend_rows += backend_micro_compare(X75k)
    del X75k

    # --- 50 SKIP_500K=1  [/overnight] ---
    skip500k = os.environ.get("SKIP_500K", "0") == "1"
    if not skip500k:
        X500k = _synthetic(500_000)
        results.append(bench_one("synthetic_500k", X500k))
        del X500k

    (Path("output")).mkdir(exist_ok=True)
    (Path("output") / "benchmark_m1_smoke.json").write_text(
        json.dumps({"results": results, "backend_compare": backend_rows}, indent=2)
    )

    # ----  markdown  ----
    env_fp = (
        f"python {platform.python_version()} / numpy {np.__version__} / "
        f"scipy {scipy.__version__} / anndata {ad.__version__}"
    )
    lines = [
        "# M1 smoke benchmark benchmark_m1_smoke.py ",
        "",
        "## ",
        f"- env: {env_fp}",
        "-  RSS `/usr/bin/time -v`  Maximum resident set size"
        f" {TIME_V}",
        "- tracemalloc Python numpy/scipy  pymalloc"
        " I7 ——validation-protocol §4",
        "- BLAS =1 ≥3  max",
        "- swap  `swapon --show` ",
        "",
        "## tracemalloc + ",
        "",
        "|  | n | k | build (s) | tracemalloc (MB) | RSS (MB) | nnz | nnz/(n·k) | CSR (MB) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    gate = {  # validation-protocol §3.3 + 75k  10
        "fixture_5k_real": "1 <2GB/<1min",
        "synthetic_10k": "1 <2GB/<1min",
        "real_75k_full": "10 <6GB/<10min",
        "synthetic_500k": "50 <16GB/<30min",
    }
    for r in results:
        lines.append(
            f"| {r['name']} | {r['n']:,} | {r['k']} | {r['build_s_max']} | "
            f"{r['tracemalloc_peak_mb_max']} | {r['rss_peak_mb_max']} | {r['nnz']:,} | "
            f"{r['nnz_per_nk']} | {r['csr_theory_mb']} |"
        )
    lines += ["", "### ", ""]
    for r in results:
        verdict = []
        if r["n"] <= 10_000:
            verdict.append(" PASS" if r["rss_peak_mb_max"] < 2048 else " FAIL(<2GB)")
            verdict.append(" PASS" if r["build_s_max"] < 60 else " FAIL(<1min)")
        elif r["n"] <= 100_000:
            verdict.append(" PASS" if r["rss_peak_mb_max"] < 6144 else " FAIL(<6GB)")
            verdict.append(" PASS" if r["build_s_max"] < 600 else " FAIL(<10min)")
        else:
            verdict.append(" PASS" if r["rss_peak_mb_max"] < 16384 else " FAIL(<16GB )")
            verdict.append(" PASS" if r["build_s_max"] < 1800 else " FAIL(<30min )")
        if abs(r["nnz_per_nk"] - 1.5) > 0.5:
            verdict.append("nnz >5% ")
        lines.append(f"- **{r['name']}**{gate.get(r['name'], '')}{''.join(verdict)}")
    lines += [
        "",
        "## kNN P1 ≥3  max",
        "",
        "| backend | n | build (s) |",
        "|---|---|---|",
    ]
    for b in backend_rows:
        lines.append(f"| {b['backend']} | {b['n']:,} | {b['build_s_max']} |")
    lines += [
        "",
        "## ",
        "",
        "-  nnz/(n·k)  CSR protocol §3.4",
        "  RSS  ≈ CSR  + kNN  scratch +  latent",
        "  RSS  tracemalloc  = numpy/scipy  pymallocprotocol §4",
        "   RSS  >30%",
        "- 100  overnight design 100 <6GB  [] ",
        "- 50[]/  SKIP_500K=1  [/overnight]",
        "",
        "## ",
        "",
        "- tracemalloc  numpy/scipy  time -v ",
        "- wall time ",
        "-  ~30GB50 latent  ~60MB",
    ]
    if skip500k:
        lines.append("- **50**SKIP_500K [/overnight]")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"results": results, "backend_compare": backend_rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
