#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_rep_selection_sensitivity.py — E65k fixture

k=30, seed 20260909 4  r=30
  centroid   r
  random1/2  r  20260910 / 20260911
  medoid     KMeans(30)
  boundary   r
 vs centroidper-cell Spearmanmin/medianclass-level ΔP_max
sample-level ΔP entropy/ +  Spearman
output/rep_selection_sensitivity.md / .json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SEED = 20260909
R = 30
K = 30
STRATEGIES = ["centroid", "random1", "random2", "medoid", "boundary"]
RANDOM_SEEDS = {"random1": 20260910, "random2": 20260911}
CONTRASTS = [("5xFAD", "control"), ("5xFAD;Cd28-cKO", "5xFAD")]
AGES = [3, 6, 8]


def main() -> int:
    t_run0 = time.perf_counter()
    sys.path.insert(0, "src")
    import anndata as ad
    from sklearn.cluster import KMeans

    from sfate import absorption_probabilities, build_transition_graph

    fx = ad.read_h5ad("output/fixture_5k.h5ad")
    X = np.asarray(fx.obsm["X_scVI"], dtype=np.float32)
    states = np.asarray(fx.obs["microglia_state"].astype(str))
    genotypes = np.asarray(fx.obs["genotype"].astype(str))
    samples = np.asarray(fx.obs["sample"].astype(str))
    ages = np.asarray(fx.obs["age_months"].astype(int))
    n = X.shape[0]
    X64 = X.astype(np.float64)
    classes = np.unique(states)

    t0 = time.perf_counter()
    P = build_transition_graph(X, k=K, random_state=SEED)
    t_graph = time.perf_counter() - t0
    print(f"graph: {t_graph:.1f}s", flush=True)

    # ---- 4  → labels ----
    def labels_centroid(ascending: bool):
        labels = np.array([None] * n, dtype=object)
        for c in classes:
            idx = np.nonzero(states == c)[0]
            cen = X64[idx].mean(axis=0)
            dist = np.linalg.norm(X64[idx] - cen, axis=1)
            order = np.argsort(dist, kind="stable")
            if not ascending:
                order = order[::-1]
            for i in idx[order[:R]]:
                labels[i] = c
        return labels

    def labels_random(seed: int):
        rng = np.random.default_rng(seed)
        labels = np.array([None] * n, dtype=object)
        for c in classes:
            idx = np.nonzero(states == c)[0]
            for i in idx[rng.permutation(len(idx))[:R]]:
                labels[i] = c
        return labels

    def labels_medoid():
        labels = np.array([None] * n, dtype=object)
        for c in classes:
            idx = np.nonzero(states == c)[0]
            km = KMeans(n_clusters=R, random_state=SEED, n_init=10).fit(X64[idx])
            d = np.linalg.norm(
                X64[idx][:, None, :] - km.cluster_centers_[None, :, :], axis=2)
            chosen: list[int] = []
            used: set[int] = set()
            #
            for j in range(R):
                for cand in np.argsort(d[:, j], kind="stable"):
                    if int(cand) not in used:
                        used.add(int(cand))
                        chosen.append(int(idx[cand]))
                        break
            assert len(chosen) == R
            for i in chosen:
                labels[i] = c
        return labels

    label_fns = {
        "centroid": lambda: labels_centroid(True),
        "random1": lambda: labels_random(RANDOM_SEEDS["random1"]),
        "random2": lambda: labels_random(RANDOM_SEEDS["random2"]),
        "medoid": labels_medoid,
        "boundary": lambda: labels_centroid(False),
    }

    sols: dict[str, np.ndarray] = {}
    t_solve: dict[str, float] = {}
    for s in STRATEGIES:
        labels = label_fns[s]()
        assert sum(1 for v in labels if v is not None) == R * len(classes)
        t0 = time.perf_counter()
        B, info = absorption_probabilities(P, labels, tol=1e-6,
                                           return_info=True)
        t_solve[s] = time.perf_counter() - t0
        sols[s] = B.astype(np.float64)
        print(f"{s}: solve {t_solve[s]:.1f}s "
              f"={int(info['n_f64_fallback'])}", flush=True)

    base = sols["centroid"]
    n_class = len(classes)

    # ----  1per-cell Spearman----
    def per_cell_spearman(B: np.ndarray) -> list[float]:
        return [float(spearmanr(B[:, j], base[:, j]).statistic)
                for j in range(n_class)]

    # ----  2class-level ----
    class_mean = {s: sols[s].mean(axis=0) for s in STRATEGIES}

    # ----  3sample-level ΔP  ----
    #  sample  (genotype, age)  sample
    uniq_samples = np.unique(samples)
    sgeno = {sm: genotypes[samples == sm][0] for sm in uniq_samples}
    sage = {sm: int(ages[samples == sm][0]) for sm in uniq_samples}

    def dp_signs(B: np.ndarray) -> dict[str, float | None]:
        sm_mean = {sm: B[samples == sm].mean(axis=0) for sm in uniq_samples}
        out: dict[str, float | None] = {}
        for ga, gb in CONTRASTS:
            for age in AGES:
                for j, c in enumerate(classes):
                    key = f"{ga}-{gb}|{age}m|{c}"
                    sa = [sm for sm in uniq_samples
                          if sgeno[sm] == ga and sage[sm] == age]
                    sb = [sm for sm in uniq_samples
                          if sgeno[sm] == gb and sage[sm] == age]
                    if not sa or not sb:
                        out[key] = None
                        continue
                    out[key] = float(
                        np.mean([sm_mean[sm][j] for sm in sa])
                        - np.mean([sm_mean[sm][j] for sm in sb]))
        return out

    dp = {s: dp_signs(sols[s]) for s in STRATEGIES}

    # ----  4entropy ----
    def entropy(B: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            h = np.where(B > 0, B * np.log(B), 0.0)
        return -h.sum(axis=1)

    ent = {s: entropy(sols[s]) for s in STRATEGIES}

    # ----  ----
    results: dict[str, dict] = {}
    for s in STRATEGIES:
        if s == "centroid":
            results[s] = {
                "entropy_mean": float(ent[s].mean()),
                "entropy_median": float(np.median(ent[s])),
                "class_mean_prob": {c: float(class_mean[s][j])
                                    for j, c in enumerate(classes)},
            }
            continue
        rho = per_cell_spearman(sols[s])
        dmax = float(np.abs(class_mean[s] - class_mean["centroid"]).max())
        keys = [k for k, v in dp["centroid"].items() if v is not None]
        agree = sum(1 for k in keys
                    if np.sign(dp[s][k]) == np.sign(dp["centroid"][k]))
        results[s] = {
            "spearman_per_class": {c: rho[j] for j, c in enumerate(classes)},
            "spearman_min": float(min(rho)),
            "spearman_median": float(np.median(rho)),
            "dP_class_max": dmax,
            "class_mean_prob": {c: float(class_mean[s][j])
                                for j, c in enumerate(classes)},
            "dp_sign_agree": f"{agree}/{len(keys)}",
            "dp_sign_agree_rate": agree / len(keys),
            "entropy_mean": float(ent[s].mean()),
            "entropy_median": float(np.median(ent[s])),
            "entropy_spearman_vs_centroid": float(
                spearmanr(ent[s], ent["centroid"]).statistic),
        }
        print(f"{s}: Spearman min={min(rho):.4f} median={np.median(rho):.4f} "
              f"ΔPmax={dmax:.4f} sign={agree}/{len(keys)} "
              f"H-ρ={results[s]['entropy_spearman_vs_centroid']:.4f}",
              flush=True)

    comp = [s for s in STRATEGIES if s != "centroid"]
    robust = (all(results[s]["spearman_min"] >= 0.9 for s in comp)
              and all(results[s]["dp_sign_agree_rate"] >= 0.9 for s in comp))
    total_s = time.perf_counter() - t_run0

    Path("output/rep_selection_sensitivity.json").write_text(json.dumps({
        "env": {"seed": SEED, "k": K, "r": R, "graph_s": round(t_graph, 1),
                "solve_s": {s: round(t_solve[s], 1) for s in STRATEGIES},
                "total_s": round(total_s, 1),
                "random_seeds": RANDOM_SEEDS},
        "baseline": "centroid",
        "classes": list(classes),
        "dp_sign_keys": [k for k, v in dp["centroid"].items() if v is not None],
        "dp_signs": dp,
        "results": results,
        "verdict_robust": bool(robust)}, ensure_ascii=False, indent=2))

    # ---- md  ----
    L = ["# E65k fixture"
         "exp_rep_selection_sensitivity.py\n",
         "## ",
         f"fixture_5k.h5adn={n}6  microglia_statescVI latent (30d, "
         f"f32) → kNN  max  Pk={K}seed {SEED}"
         f" r={R}  GMRES "
         "(tol=1e-6)  B (n×6)4 **centroid**"
         " r **random1/random2**"
         f" {RANDOM_SEEDS['random1']}/{RANDOM_SEEDS['random2']}**medoid**"
         f" KMeans({R}, seed {SEED}) "
         "**boundary** r  centroid "
         "\n",
         "## 1. per-cell Spearman vs centroid",
         "|  | Spearman min | Spearman median |", "|---|---|---|"]
    for s in comp:
        L.append(f"| {s} | {results[s]['spearman_min']:.4f} | "
                 f"{results[s]['spearman_median']:.4f} |")
    L += ["\n## 2. class-level ",
          "|  | ΔP_class_max vs centroid |", "|---|---|"]
    for s in comp:
        L.append(f"| {s} | {results[s]['dP_class_max']:.4f} |")
    L += ["\n## 3. sample-level ΔP ",
          " sample  → (genotype, age)  sample  "
          "→ 5xFAD−control5xFAD;Cd28−cKO−5xFAD× 3  × 6  "
          " ΔP  NA "
          f" = {len([k for k, v in dp['centroid'].items() if v is not None])}",
          "|  |  |", "|---|---|"]
    for s in comp:
        L.append(f"| {s} | {results[s]['dp_sign_agree']} "
                 f"({results[s]['dp_sign_agree_rate']*100:.1f}%) |")
    L += ["\n## 4. entropy 6  Shannon ",
          "|  |  |  |  Spearman vs centroid |",
          "|---|---|---|---|"]
    L.append(f"| centroid () | {results['centroid']['entropy_mean']:.4f} "
             f"| {results['centroid']['entropy_median']:.4f} | — |")
    for s in comp:
        r_ = results[s]
        L.append(f"| {s} | {r_['entropy_mean']:.4f} | "
                 f"{r_['entropy_median']:.4f} | "
                 f"{r_['entropy_spearman_vs_centroid']:.4f} |")

    branch_robust = ("\n## \nrandom1/random2/medoid/boundary  centroid "
                     " per-cell Spearman min  ≥ 0.9 sample-level "
                     "ΔP  ≥ 90%**target representation "
                     "**centroid-nearest  "
                     "modeling choice")
    branch_first = ("\n## \n centroid  per-cell Spearman "
                    "min < 0.9  sample-level ΔP  < 90%**target "
                    "representation **"
                    "")
    L.append(branch_robust if robust else branch_first)
    L.append(f"\n---\n {total_s:.1f}s {t_graph:.1f}s +  "
             f"{sum(t_solve.values()):.1f}s")
    Path("output/rep_selection_sensitivity.md").write_text(
        "\n".join(L), encoding="utf-8")
    print(f"written: output/rep_selection_sensitivity.md (total {total_s:.1f}s)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
