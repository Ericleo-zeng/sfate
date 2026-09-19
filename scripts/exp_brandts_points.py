#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_brandts_points.py — P1  E3+E4 cellrank 2.1.0  brandts
n=2000 / 5000 / 8000

- E42k/5k  2.1.0  T7  2.3.2  288.2/1800.3 MB→  caveat
- E3n=8000  → 72·n²
- sfate_env petsc4py/slepc4py method='brandts'
  tests/conftest.py  latentd=10, n_terminal=3, seed=42
  tracemalloc  + RSS 250ms /usr/bin/time -v
  BLAS=1n=8000  ~72×8000²=4.6GB30GB
output/brandts_8k.md E4
"""

from __future__ import annotations

import json
import sys
import threading
import time
import tracemalloc
from pathlib import Path

import numpy as np
import psutil

SEED = 42
OUT_MD = Path("output/brandts_8k.md")
OUT_JSON = Path("output/brandts_8k.json")


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


def run_one(n: int) -> dict:
    import anndata as ad
    import scanpy as sc
    from cellrank import __version__ as cr_ver
    from cellrank.estimators import GPCCA
    from cellrank.kernels import ConnectivityKernel

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
    from conftest import make_branching_latent

    X = make_branching_latent(n=n, d=10, n_terminal=3, seed=SEED)
    adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
    adata.obsm["X_latent"] = X
    sc.pp.neighbors(adata, n_neighbors=30, use_rep="X_latent", random_state=SEED)
    ck = ConnectivityKernel(adata).compute_transition_matrix(density_normalize=True)

    stop = threading.Event()
    peak = {"mb": rss_mb()}

    def _sample():
        while not stop.is_set():
            peak["mb"] = max(peak["mb"], rss_mb())
            stop.wait(0.25)

    th = threading.Thread(target=_sample, daemon=True)
    th.start()
    tracemalloc.start()
    t0 = time.perf_counter()
    g = GPCCA(ck)
    g.compute_schur(n_components=10, method="brandts")
    t_schur = time.perf_counter() - t0
    t1 = time.perf_counter()
    g.compute_macrostates(n_states=3)
    t_macro = time.perf_counter() - t1
    _, tm_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    stop.set()
    th.join(timeout=2.0)
    return {
        "n": n,
        "cellrank": cr_ver,
        "schur_s": round(t_schur, 2),
        "macro_s": round(t_macro, 2),
        "tracemalloc_mb": round(tm_peak / 1e6, 1),
        "rss_peak_mb": round(peak["mb"], 1),
        "coef_tracemalloc": round(tm_peak / 1e6 * 1e6 / n**2, 1),  # B/n²
    }


def main() -> int:
    import cellrank as cr
    import scipy

    env = {
        "date": time.strftime("%Y-%m-%d"),
        "cellrank": cr.__version__,
        "pygpcca": __import__("pygpcca").__version__,
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "blas_threads": "1",
        "seed": SEED,
        "method_forced": "brandts",
    }
    print("env:", json.dumps(env, ensure_ascii=False), flush=True)

    results = []
    for n in (2000, 5000, 8000):
        r = run_one(n)
        results.append(r)
        coef = r["tracemalloc_mb"] * 1e6 / n**2
        print(f"n={n}: schur {r['schur_s']}s macro {r['macro_s']}s "
              f"tm={r['tracemalloc_mb']}MB rss_peak={r['rss_peak_mb']}MB "
              f"coef={coef:.1f} B/n²", flush=True)

    OUT_JSON.write_text(json.dumps({"env": env, "results": results}, indent=2))

    L = []
    L.append("# brandts  + 2.1.0 exp_brandts_points.py \n")
    L.append("## ")
    L.append(f"- cellrank **{env['cellrank']}**T7  2.3.2——"
             f" E4 / pygpcca {env['pygpcca']} / scipy {env['scipy']} / "
             f"numpy {env['numpy']} /  BLAS=1 / seed {SEED}")
    L.append("-  latenttests/conftest.pyd=10n_terminal=3"
             "scanpy neighbors(k=30) → ConnectivityKernel → "
             "`compute_schur(n_components=10, method='brandts')`"
             "→ compute_macrostates(n_states=3)")
    L.append("- tracemalloc  T7 + RSS 250ms"
             " output/brandts_8k_time_v.txt\n")
    L.append("## \n")
    L.append("| n | Schur (s) | macro (s) | tracemalloc (MB) | "
             "RSS (MB) | (B/n²) |")
    L.append("|---|---|---|---|---|---|")
    for r in results:
        coef = r["tracemalloc_mb"] * 1e6 / r["n"] ** 2
        L.append(f"| {r['n']} | {r['schur_s']} | {r['macro_s']} | "
                 f"{r['tracemalloc_mb']} | {r['rss_peak_mb']} | {coef:.1f} |")
    L.append("\n## E4 2.1.0  vs T7@2.3.2\n")
    L.append("| n | 2.3.2T7 | 2.1.0 |")
    L.append("|---|---|---|")
    t7 = {2000: 288.2, 5000: 1800.3}
    for r in results[:2]:
        L.append(f"| {r['n']} | {t7[r['n']]} MB | {r['tracemalloc_mb']} MB |")
    L.append("\n## E3 n=8000\n")
    r8 = results[2]
    coef8 = r8["tracemalloc_mb"] * 1e6 / r8["n"] ** 2
    L.append(f"- n=8000tracemalloc  {r8['tracemalloc_mb']} MB"
             f" {coef8:.1f} B/n²72·n²  {72 * r8['n']**2 / 1e6:.0f} MB")
    L.append("-  ∈ [64, 80] B/n² n² "
             "")
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"written: {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
