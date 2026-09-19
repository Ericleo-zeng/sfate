#!/usr/bin/env python
"""500k full-pipeline benchmark with incremental logging.

solve_mode=columnsrtol=1e-6restart=50
float32k=30r=30×6


  source ~/sfate_env/bin/activate && OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 COLUMN_SUBSET=18 \
    /usr/bin/time -v python scripts/exp_500k_full_pipeline.py --rep-offset 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sfate.graph import build_transition_graph
from sfate.fate import absorption_probabilities

SEED = 20260909
N = 500_000
D = 30
K = 30
R = 30
N_CL = 5

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")


def _rss_sampler(stop: threading.Event, out: dict, interval: float = 0.25) -> None:
    proc = psutil.Process()
    while not stop.is_set():
        out["peak_mb"] = max(out.get("peak_mb", 0.0), proc.memory_info().rss / 1e6)
        stop.wait(interval)


def _synthetic(n: int, d: int = D) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    per = n // N_CL
    parts, labels = [], []
    centers = rng.normal(0, 3.0, size=(N_CL, d))
    for j in range(N_CL):
        nj = per + (1 if j < n % N_CL else 0)
        parts.append(centers[j] + rng.normal(0, 1.0, size=(nj, d)))
        labels.append(np.full(nj, j, dtype=np.int64))
    return np.concatenate(parts, 0).astype(np.float32), np.concatenate(labels, 0)


def _select_absorbing(X: np.ndarray, labels: np.ndarray, r: int):
    n = X.shape[0]
    terminal_labels = np.full(n, None, dtype=object)
    abs_idx = []
    for c in np.unique(labels):
        mask = labels == c
        idx_c = np.nonzero(mask)[0]
        centroid = X[mask].mean(axis=0)
        dists = np.linalg.norm(X[mask] - centroid, axis=1)
        order = np.argsort(dists, kind="stable")
        chosen = idx_c[order[:min(r, len(idx_c))]]
        abs_idx.extend(chosen.tolist())
        terminal_labels[chosen] = str(c)
    return np.asarray(abs_idx, dtype=np.int64), terminal_labels


def _stage_log(logf, name: str, t0: float, rss: dict):
    logf.write(
        f"{datetime.now().isoformat()} STAGE {name} "
        f"elapsed={time.perf_counter()-t0:.3f}s "
        f"rss_peak_mb={rss.get('peak_mb', 0.0):.1f}\n"
    )
    logf.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep-offset", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="output")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    rep = args.rep_offset

    log_path = out_dir / "benchmark_500k_full.log"
    logf = open(log_path, "a", buffering=1, encoding="utf-8")
    logf.write(f"{datetime.now().isoformat()} START rep={rep} n={N} k={K} r={R}\n")
    logf.flush()

    rss: dict = {}
    stop = threading.Event()
    th = threading.Thread(target=_rss_sampler, args=(stop, rss))
    th.start()
    tracemalloc.start()

    def stage(name: str, t0: float):
        _stage_log(logf, name, t0, rss)

    col_subset = int(os.environ.get("COLUMN_SUBSET", "180"))
    logf.write(f"{datetime.now().isoformat()} CONFIG column_subset={col_subset}\n")
    logf.flush()

    # latent generation
    t0 = time.perf_counter()
    X, labels = _synthetic(N, D)
    stage("latent_done", t0)

    # graph construction
    t0 = time.perf_counter()
    P = build_transition_graph(X, k=K, random_state=SEED, knn_backend="pynndescent")
    stage("graph_done", t0)

    # absorbing states
    t0 = time.perf_counter()
    abs_idx, terminal_labels = _select_absorbing(X, labels, R)
    stage("absorbing_done", t0)

    # solve: column-wise logging
    def col_callback(j, c, it, true_rel, t_col):
        logf.write(
            f"{datetime.now().isoformat()} COL j={j} c={c} "
            f"iters={it} true_res={true_rel:.6e} sec={t_col:.3f}\n"
        )
        logf.flush()

    # subset selection: fixed seed, reproducible
    if col_subset < len(abs_idx):
        rng = np.random.default_rng(SEED)
        chosen_abs = rng.choice(len(abs_idx), size=col_subset, replace=False)
        chosen_abs.sort()
        terminal_labels_sub = np.full(N, None, dtype=object)
        terminal_labels_sub[abs_idx[chosen_abs]] = terminal_labels[abs_idx[chosen_abs]]
        terminal_labels = terminal_labels_sub
        logf.write(f"{datetime.now().isoformat()} SUBSET selected={col_subset}\n")
        logf.flush()

    t0 = time.perf_counter()
    stage("solve_start", t0)
    try:
        B, info = absorption_probabilities(
            P,
            terminal_labels,
            block_size=256,
            tol=1e-6,
            restart=50,
            maxiter=1000,
            solve_mode="columns",
            preconditioner="none",
            return_info=True,
            verbose=False,
            col_callback=col_callback,
        )
        solve_s = time.perf_counter() - t0
        converged = True
    except RuntimeError as e:
        solve_s = time.perf_counter() - t0
        info = {"error": str(e)}
        converged = False
        B = None

    stage("solve_done", t0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    stop.set()
    th.join()

    out = {
        "rep": rep,
        "n": N,
        "d": D,
        "k": K,
        "r": R,
        "column_subset": col_subset,
        "converged": converged,
        "rss_peak_mb": rss.get("peak_mb", 0.0),
        "tracemalloc_peak_mb": peak / 1e6,
        "nnz": int(P.nnz),
    }
    if converged:
        iters = info["gmres_iters"]
        out.update({
            "gmres_iters_sum": int(iters.sum()),
            "gmres_iters_min": int(iters.min()),
            "gmres_iters_median": float(np.median(iters)),
            "gmres_iters_max": int(iters.max()),
            "residual_max": float(np.max(info["residual_max"])),
            "n_f64_fallback": int(info["n_f64_fallback"]),
            "n_classes": int(len(info["classes"])),
            "solve_s": solve_s,
        })
    else:
        out["error"] = info.get("error", "unknown")
        out["solve_s"] = solve_s

    rep_json = out_dir / f"benchmark_500k_full_rep{rep}.json"
    rep_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    logf.write(f"{datetime.now().isoformat()} FINISH rep={rep} converged={converged}\n")
    logf.flush()
    logf.close()

    print(json.dumps(out, indent=2))
    return 0 if converged else 1


if __name__ == "__main__":
    sys.exit(main())
