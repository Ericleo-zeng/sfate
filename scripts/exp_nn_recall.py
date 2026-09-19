#!/usr/bin/env python3
"""A1: NN-Descent neighbor recall vs exact search on 5k real and 10k synthetic fixtures.

Outputs: output/nn_recall.md / output/nn_recall.json
Recall = |approx_knn ∩ exact_knn| / k per query cell, averaged over cells.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sfate.graph import _query_knn

SEEDS = [20260907, 20260909, 20260911]
K = 30
OUT_MD = ROOT / "output/nn_recall.md"
OUT_JSON = ROOT / "output/nn_recall.json"


def _synthetic(n: int, d: int = 30, seed: int = 20260909) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_cl = 5
    per = n // n_cl
    centers = rng.normal(0, 3.0, size=(n_cl, d))
    parts = []
    for j in range(n_cl):
        nj = per + (1 if j < n % n_cl else 0)
        parts.append(centers[j] + rng.normal(0, 1.0, size=(nj, d)))
    return np.concatenate(parts, 0).astype(np.float32)


def exact_neighbors(X: np.ndarray, k: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric="euclidean")
    nn.fit(X)
    _, idx = nn.kneighbors(X)
    return idx[:, 1:]  # drop self


def recall(approx: np.ndarray, exact: np.ndarray) -> float:
    k = approx.shape[1]
    hits = sum(len(set(a) & set(e)) for a, e in zip(approx, exact))
    return hits / (approx.shape[0] * k)


def evaluate(name: str, X: np.ndarray) -> dict:
    exact = exact_neighbors(X, K)
    rows = []
    for seed in SEEDS:
        t0 = time.perf_counter()
        approx, _ = _query_knn(X, K, backend="pynndescent", random_state=seed)
        t = time.perf_counter() - t0
        r = recall(approx, exact)
        rows.append({"seed": seed, "recall": round(r, 6), "time_s": round(t, 3)})
    recalls = [r["recall"] for r in rows]
    return {
        "name": name,
        "n": int(X.shape[0]),
        "k": K,
        "recalls": rows,
        "mean_recall": round(float(np.mean(recalls)), 6),
        "min_recall": round(float(np.min(recalls)), 6),
        "max_recall": round(float(np.max(recalls)), 6),
        "std_recall": round(float(np.std(recalls)), 6),
    }


def main() -> int:
    fx = ad.read_h5ad(ROOT / "output/fixture_5k.h5ad")
    X5k = np.asarray(fx.obsm["X_scVI"], dtype=np.float32)
    X10k = _synthetic(10_000)

    r5k = evaluate("fixture_5k_real", X5k)
    r10k = evaluate("synthetic_10k", X10k)
    report = {"date": time.strftime("%Y-%m-%d"), "results": [r5k, r10k]}

    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    lines = [
        "# NN-Descent recall and seed sensitivity (exp_nn_recall.py)",
        "",
        f"- date: {report['date']}",
        f"- exact comparator: sklearn NearestNeighbors(brute, euclidean), k={K}",
        f"- seeds: {SEEDS}",
        "",
        "| fixture | n | seed | recall | build+query time (s) |",
        "|---|---|---|---|---|",
    ]
    for res in report["results"]:
        for row in res["recalls"]:
            lines.append(
                f"| {res['name']} | {res['n']:,} | {row['seed']} | "
                f"{row['recall']:.4f} | {row['time_s']:.3f} |"
            )
        lines.append(
            f"| {res['name']} | {res['n']:,} | **summary** | "
            f"mean {res['mean_recall']:.4f} (min {res['min_recall']:.4f}, "
            f"max {res['max_recall']:.4f}, std {res['std_recall']:.4f}) | — |"
        )
    lines += [
        "",
        "Interpretation: recall > 0.99 with < 0.001 seed std ⇒ approximate neighbor graph",
        "is effectively deterministic and high-recall on these fixtures; 500k scalability",
        "is reported as completion, not recall, because exact search is intractable.",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"written: {OUT_JSON}\nwritten: {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
