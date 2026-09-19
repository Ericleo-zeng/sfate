#!/usr/bin/env python
"""M2 benchmarkfixture_5k fate.py

 M1validation-protocol §4BLAS =1tracemalloc + RSS
250ms ≥3  max /usr/bin/time -v swap

/→+split→ILU→GMRES  GMRES
block_size ∈ {1,16,256,} hub  vs
/a3 vs spsolve f64

 /usr/bin/time -v
  source ~/sfate_env/bin/activate && OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /usr/bin/time -v python scripts/benchmark_m2.py 2> output/benchmark_m2_time_v.txt
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import anndata as ad
import numpy as np
import psutil
import scipy.sparse as sp
import scipy.stats
from scipy.sparse.linalg import spsolve

from sfate.fate import _class_indicator, _csc_column_dense, absorption_probabilities
from sfate.graph import absorb_states, build_transition_graph, split_transient_absorbing

SEED = 20260909
REPS = 3
K = 30
N_REPS_PER_CLASS = 30
BLOCK_SIZES = [1, 16, 256, None]
REPORT = Path("output/benchmark_m2.md")
RESULTS = Path("output/benchmark_m2.json")
TIME_V = Path("output/benchmark_m2_time_v.txt")
FIXTURE = Path("output/fixture_5k.h5ad")


def _rss_sampler(stop: threading.Event, out: dict, interval: float = 0.25) -> None:
    proc = psutil.Process()
    while not stop.is_set():
        out["peak_mb"] = max(out.get("peak_mb", 0.0), proc.memory_info().rss / 1e6)
        stop.wait(interval)


def _select_terminals(X: np.ndarray, states: np.ndarray):
    """ state  30  tests/test_fate.py """
    n = X.shape[0]
    labels = np.array([None] * n, dtype=object)
    X64 = X.astype(np.float64)
    for c in np.unique(states):
        idx = np.nonzero(states == c)[0]
        centroid = X64[idx].mean(axis=0)
        dist = np.linalg.norm(X64[idx] - centroid, axis=1)
        reps = idx[np.argsort(dist, kind="stable")[:N_REPS_PER_CLASS]]
        for i in reps:
            labels[i] = c
    return labels


def _golden_f64(P, labels):
    valid = np.array([x is not None for x in labels], dtype=bool)
    abs_idx = np.nonzero(valid)[0]
    classes = np.unique(labels[valid])
    cls_map = {c: j for j, c in enumerate(classes)}
    cls_id = np.array([cls_map[labels[i]] for i in abs_idx], dtype=np.int64)
    Pa = absorb_states(P.astype(np.float64), abs_idx)
    Q, R, trans_idx, abs_idx = split_transient_absorbing(Pa, abs_idx)
    A = (sp.eye(Q.shape[0], format="csc", dtype=np.float64) - Q.tocsc()).tocsc()
    S = _class_indicator(cls_id, classes.shape[0])
    R_agg = (R @ S).tocsc()
    gold = np.column_stack([
        spsolve(A, _csc_column_dense(R_agg, j, np.float64))
        for j in range(classes.shape[0])
    ])
    return gold, trans_idx


def main() -> None:
    t_start = time.perf_counter()
    import scipy

    env_fp = (f"python {sys.version.split()[0]} / numpy {np.__version__} / "
              f"scipy {scipy.__version__} / anndata {ad.__version__}")

    adata = ad.read_h5ad(FIXTURE)
    X = np.asarray(adata.obsm["X_scVI"], dtype=np.float32)
    states = np.asarray(adata.obs["microglia_state"].astype(str))
    labels = _select_terminals(X, states)
    n = X.shape[0]

    stages: dict = {}

    # ---  1 M1  numba JIT ---
    build_transition_graph(X, k=K, random_state=SEED)
    t0 = time.perf_counter()
    P = build_transition_graph(X, k=K, random_state=SEED)
    stages["build_graph_s"] = time.perf_counter() - t0
    deg = np.diff(P.indptr)

    # ---  2 spsolve(f64) ---
    t0 = time.perf_counter()
    gold, trans_idx = _golden_f64(P, labels)
    stages["golden_spsolve_s"] = time.perf_counter() - t0

    # ---  3 block_size  REPS  max ---
    absorption_probabilities(P, labels, block_size=256, tol=1e-6)  #
    rows = []
    for bs in BLOCK_SIZES:
        best = {"t": -1.0, "tracemalloc_mb": -1.0, "rss_mb": -1.0}
        B = info = None
        for _ in range(REPS):
            rss: dict = {}
            stop = threading.Event()
            th = threading.Thread(target=_rss_sampler, args=(stop, rss))
            th.start()
            tracemalloc.start()
            t0 = time.perf_counter()
            B, info = absorption_probabilities(
                P, labels, block_size=bs, tol=1e-6, return_info=True
            )
            dt = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            stop.set()
            th.join()
            best["t"] = max(best["t"], dt)
            best["tracemalloc_mb"] = max(best["tracemalloc_mb"], peak / 1e6)
            best["rss_mb"] = max(best["rss_mb"], rss.get("peak_mb", 0.0))
        diff = np.abs(B[trans_idx].astype(np.float64) - gold)
        rows.append({
            "block_size": "" if bs is None else bs,
            "wall_s": best["t"],
            "tracemalloc_mb": best["tracemalloc_mb"],
            "rss_mb": best["rss_mb"],
            "linf_vs_gold": float(diff.max()),
            "p99_vs_gold": float(np.quantile(diff, 0.99)),
            "gmres_iters": [int(x) for x in info["gmres_iters"]],
            "residual_max": [float(x) for x in info["residual_max"]],
            "n_f64_fallback": info["n_f64_fallback"],
            "ilu_fill_factor_actual": info["ilu_fill_factor_actual"],
        })

    # ---  4hub  ---
    #  vs
    B, info = absorption_probabilities(P, labels, block_size=None, tol=1e-6,
                                       return_info=True)
    valid = np.array([x is not None for x in labels], dtype=bool)
    abs_idx = np.nonzero(valid)[0]
    Pa = absorb_states(P, abs_idx)
    Q, R, t_idx, _ = split_transient_absorbing(Pa, abs_idx)
    A = (sp.eye(Q.shape[0], format="csr", dtype=np.float32) - Q).tocsr()
    S = _class_indicator(
        np.array([{c: j for j, c in enumerate(info["classes"])}[labels[i]]
                  for i in abs_idx], dtype=np.int64),
        len(info["classes"]),
    ).astype(np.float32)
    R_agg = (R.astype(np.float32) @ S).tocsc()
    R_dense = np.zeros(R_agg.shape, dtype=np.float32)
    for j in range(R_agg.shape[1]):
        s, e = R_agg.indptr[j], R_agg.indptr[j + 1]
        R_dense[R_agg.indices[s:e], j] = R_agg.data[s:e]
    resid = np.abs(A @ B[t_idx] - R_dense).max(axis=1)
    deg_t = np.diff(Pa.indptr)[t_idx]
    worst_rows = np.argsort(resid)[-5:][::-1]
    hub = {
        "deg_min": int(deg.min()), "deg_median": float(np.median(deg)),
        "deg_max": int(deg.max()), "nnz": int(P.nnz),
        "nnz_per_nk": float(P.nnz / (n * K)),
        "resid_deg_spearman": float(
            scipy.stats.spearmanr(deg_t, resid).statistic
        ),
        "worst_rows": [
            {"row": int(t_idx[r]), "deg": int(deg_t[r]), "resid": float(resid[r])}
            for r in worst_rows
        ],
    }

    wall_total = time.perf_counter() - t_start
    swap = subprocess.run(["swapon", "--show"], capture_output=True, text=True).stdout.strip()

    # ---  JSON + Markdown ---
    payload = {"env": env_fp, "seed": SEED, "k": K, "n": n,
               "n_reps_per_class": N_REPS_PER_CLASS, "stages": stages,
               "blocks": rows, "hub": hub, "wall_total_s": wall_total,
               "swap": swap}
    RESULTS.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    L = []
    L.append("# M2 benchmark benchmark_m2.py \n")
    L.append("## ")
    L.append(f"- env: {env_fp}")
    L.append("- BLAS =1 ≥3  maxtracemallocPython "
             "+ RSS 250ms time -v")
    L.append(f"- fixture_5kn={n}k={K}6  microglia_state "
             f" {N_REPS_PER_CLASS}  tests/test_fate.py\n")
    L.append("## ")
    L.append("|  | (s) |")
    L.append("|---|---|")
    L.append(f"| build_transition_graph | {stages['build_graph_s']:.2f} |")
    L.append(f"|  spsolve(f64)  | {stages['golden_spsolve_s']:.2f} |")
    L.append("")
    L.append("##  block_size f32 + GMRES + spilutol=1e-63 reps  max")
    L.append("| block_size | (s) | tracemalloc (MB) | RSS (MB) | "
             "L∞ vs spsolve(f64) | p99 | f64  | ILU fill_factor  |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['block_size']} | {r['wall_s']:.2f} | {r['tracemalloc_mb']:.1f} | "
                 f"{r['rss_mb']:.1f} | {r['linf_vs_gold']:.2e} | {r['p99_vs_gold']:.2e} | "
                 f"{r['n_f64_fallback']} | {r['ilu_fill_factor_actual']} |")
    a3_pass = all(r["linf_vs_gold"] <= 1e-5 for r in rows)
    L.append(f"\n**a3 L∞ ≤ 1e-5 → {'PASS' if a3_pass else 'FAIL'}"
             f" {max(r['linf_vs_gold'] for r in rows):.2e} "
             f"{1e-5 / max(r['linf_vs_gold'] for r in rows):.0f}×[]**\n")
    L.append("##  GMRES block_size= ")
    L.append("|  | GMRES  |  max ‖(I−Q)x−b‖∞ |")
    L.append("|---|---|---|")
    for j, c in enumerate(info["classes"]):
        L.append(f"| {c} | {info['gmres_iters'][j]} | {info['residual_max'][j]:.2e} |")
    L.append("")
    L.append("## hub M1  max  389 ")
    L.append(f"- P min={hub['deg_min']} / ={hub['deg_median']:.0f} / "
             f"max={hub['deg_max']}nnz={hub['nnz']}nnz/(n·k)={hub['nnz_per_nk']:.3f}")
    L.append(f"-  Spearman {hub['resid_deg_spearman']:.3f}")
    L.append("-  5 ")
    L.append("|  |  |  |")
    L.append("|---|---|---|")
    for w in hub["worst_rows"]:
        L.append(f"| {w['row']} | {w['deg']} | {w['resid']:.2e} |")
    L.append("")
    L.append("##  swap ")
    L.append("- time -v  output/benchmark_m2_time_v.txt ")
    swap_lines = [l for l in swap.splitlines() if l.strip()]
    L.append(f"- swap`{' ; '.join(swap_lines) if swap_lines else ''}`")
    L.append("")
    L.append("## ")
    L.append("- tracemalloc  numpy/scipy  time -v ")
    L.append("- a2n=2k GMRES f64 tol=1e-12 vs spsolve M2 ——"
             " ≤1e-12 + L∞ ≤ 10·κ∞·backward_maxκ∞  M- "
             "‖A‖∞·max(A⁻¹·1) L∞≤1e-9 "
             "(I−Q) +  cast f32  + eps "
             " docs/validation-protocol-e1-addendum.md E3-a2 ")
    REPORT.write_text("\n".join(L) + "\n")
    print(f"written: {REPORT} / {RESULTS} {wall_total:.1f}s")


if __name__ == "__main__":
    main()
