#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_sign_robustness_reps.py — E6b75k

 c_layer output/c_layer_graph.npz74984²k=30seed 20260909
n_drop=0 random / medoid
Fig5 scripts/export_sample_tsv.py  genotype_delta_sign
centroid  9/15 IFN 4/4C1q 4/4Homeo 0/2
 target representation

- random 30  20260910 E6 random1
- medoid KMeans(30, random_state=20260909, n_init=10)
   E6 medoid
- centroidoutput/fate_full_75k.h5ad

CR fate_prob_by_sample_WIDE.tsv


  output/sign_robustness_across_reps.md / .json
  output/fate_probabilities/fate_by_sample_sign_robustness.tsv
  output/sign_robustness_reps_cache.npzrandom/medoid


  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    <local>/sfate_env/bin/python scripts/exp_sign_robustness_reps.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

SEED_KMEANS = 20260909
SEED_RANDOM = 20260910
R = 30
ANNOTATED = ("<data>/"
             "06_annotation/adata_annotated.h5ad")
CR_WIDE = ("<data>/"
           "11_cellrank/fate_prob_by_sample_WIDE.tsv")
GRAPH_CACHE = Path("output/c_layer_graph.npz")
FATE_H5AD = Path("output/fate_full_75k.h5ad")
SOLVE_CACHE = Path("output/sign_robustness_reps_cache.npz")
OUT_MD = Path("output/sign_robustness_across_reps.md")
OUT_JSON = Path("output/sign_robustness_across_reps.json")
OUT_TSV = Path("output/fate_probabilities/fate_by_sample_sign_robustness.tsv")
CLASSES = ["IFN_responsive", "C1q_inflammatory", "Homeostatic"]  # CR  3
CONTRASTS = [("5xFAD", "control", "5xFAD-control"), ("5xFAD;Cd28-cKO", "5xFAD", "cKO-5xFAD")]
STRATEGIES = ["centroid", "random", "medoid"]


def labels_random(X: np.ndarray, states: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """ 30  20260910 E6 random1"""
    rng = np.random.default_rng(SEED_RANDOM)
    labels = np.array([None] * len(states), dtype=object)
    for c in classes:
        idx = np.nonzero(states == c)[0]
        for i in idx[rng.permutation(len(idx))[:R]]:
            labels[i] = c
    return labels


def labels_medoid(X: np.ndarray, states: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """ KMeans(30)  E6 medoid"""
    from sklearn.cluster import KMeans

    X64 = X.astype(np.float64)
    labels = np.array([None] * len(states), dtype=object)
    for c in classes:
        idx = np.nonzero(states == c)[0]
        km = KMeans(n_clusters=R, random_state=SEED_KMEANS, n_init=10).fit(X64[idx])
        d = np.linalg.norm(
            X64[idx][:, None, :] - km.cluster_centers_[None, :, :], axis=2)
        chosen: list[int] = []
        used: set[int] = set()
        for j in range(R):
            for cand in np.argsort(d[:, j], kind="stable"):
                if int(cand) not in used:
                    used.add(int(cand))
                    chosen.append(int(idx[cand]))
                    break
        assert len(chosen) == R, f"{c}: medoid  {len(chosen)} != {R}"
        for i in chosen:
            labels[i] = c
    return labels


def solve_strategy(name: str, P_sub, labels_sub) -> dict:
    """ {fate, classes, gmres_iters, residual_max, n_f64_fallback, t_solve}"""
    key = lambda k: f"{name}__{k}"
    if SOLVE_CACHE.exists():
        z = np.load(SOLVE_CACHE, allow_pickle=True)
        if key("fate") in z.files:
            print(f"[cache]  {SOLVE_CACHE}  {name} ", flush=True)
            return {k: z[key(k)] for k in
                    ["fate", "classes", "gmres_iters", "residual_max",
                     "n_f64_fallback", "t_solve"]}

    from sfate import absorption_probabilities

    t0 = time.perf_counter()
    B, info = absorption_probabilities(P_sub, labels_sub, tol=1e-6,
                                       return_info=True, verbose=True)
    t_solve = time.perf_counter() - t0
    out = {"fate": B, "classes": np.array(info["classes"]),
           "gmres_iters": np.asarray(info["gmres_iters"]),
           "residual_max": np.asarray(info["residual_max"]),
           "n_f64_fallback": np.int64(info["n_f64_fallback"]),
           "t_solve": np.float64(t_solve)}
    #
    store = {key(k): v for k, v in out.items()}
    if SOLVE_CACHE.exists():
        z = np.load(SOLVE_CACHE, allow_pickle=True)
        store = {**{k2: z[k2] for k2 in z.files}, **store}
    np.savez_compressed(SOLVE_CACHE, **store)
    print(f"[cache]  {SOLVE_CACHE}{name}", flush=True)
    return out


def main() -> int:
    import scipy.sparse as sp
    import pandas as pd
    import anndata as ad

    from sfate.io import load_latent, load_obs_categorical, load_obs_columns

    t_run0 = time.perf_counter()

    # ---- 1.  + latent +  + obs ----
    z = np.load(GRAPH_CACHE, allow_pickle=True)
    P_full = sp.csr_matrix((z["P_data"], z["P_indices"], z["P_indptr"]),
                           shape=tuple(z["P_shape"]))
    keep = z["keep"].astype(bool)
    n_drop, n_comp = int(z["n_drop"]), int(z["n_comp"])
    n = P_full.shape[0]
    print(f"graph: {n}², nnz={P_full.nnz}, n_drop={n_drop}, n_comp={n_comp}", flush=True)

    X = load_latent(ANNOTATED, "X_scVI")
    states = np.asarray(load_obs_categorical(ANNOTATED, "microglia_state").astype(str))
    obs = load_obs_columns(ANNOTATED, ["sample", "genotype", "age_months"])
    assert X.shape[0] == n == len(states) == len(obs), ""
    classes = np.unique(states)
    assert len(classes) == 6

    # ---- 2.  labels → keep  centroid ----
    t0 = time.perf_counter()
    lab_random = labels_random(X, states, classes)
    lab_medoid = labels_medoid(X, states, classes)
    for nm, lab in [("random", lab_random), ("medoid", lab_medoid)]:
        assert sum(1 for v in lab if v is not None) == R * len(classes), nm
        for c in classes:
            assert sum(1 for v in lab if v == c) == R, f"{nm}/{c}  != {R}"
    t_labels = time.perf_counter() - t0
    print(f"labels {t_labels:.1f}s", flush=True)

    P_sub = P_full[keep][:, keep].tocsr()

    # ---- 3. ----
    sols: dict[str, dict] = {}
    for nm, lab in [("random", lab_random), ("medoid", lab_medoid)]:
        sols[nm] = solve_strategy(nm, P_sub, lab[keep])
        print(f"{nm}: solve {float(sols[nm]['t_solve']):.1f}s "
              f"iters={sols[nm]['gmres_iters'].tolist()} "
              f"resmax={float(sols[nm]['residual_max'].max()):.2e} "
              f"f64={int(sols[nm]['n_f64_fallback'])}", flush=True)

    # centroid fate_full_75k.h5ad ANNOTATED n_drop=0
    a = ad.read_h5ad(FATE_H5AD)
    sols["centroid"] = {
        "fate": np.asarray(a.obsm["sfate_fate"]),
        "classes": np.array(a.uns["sfate_meta"]["classes"]),
        "gmres_iters": np.asarray(a.uns["sfate_meta"]["gmres_iters_max_column"]),
        "residual_max": np.asarray(a.uns["sfate_meta"]["residual_max"]),
        "n_f64_fallback": np.int64(a.uns["sfate_meta"]["n_f64_fallback"]),
        "t_solve": np.float64(a.uns["sfate_meta"]["wall_time_s"]),
    }
    assert sols["centroid"]["fate"].shape[0] == int(keep.sum())
    print("centroid:  fate_full_75k.h5adcolumns ", flush=True)

    # ---- 4.  × 15  export_sample_tsv.py----
    cr = pd.read_csv(CR_WIDE, sep="\t")
    cr["age"] = cr["age"].astype(str)
    cr = cr[["sample", "genotype", "age"] + [f"fate_{c}" for c in CLASSES]]

    sample_means: dict[str, pd.DataFrame] = {}
    for s in STRATEGIES:
        classes_all = [str(c) for c in sols[s]["classes"]]
        fate = pd.DataFrame(sols[s]["fate"], columns=classes_all,
                            index=obs.index[keep])
        o = obs.copy()
        o.index = fate.index
        df = o.join(fate)
        sm = (df.groupby(["sample", "genotype", "age_months"], observed=True)[CLASSES]
                .mean().reset_index().rename(columns={"age_months": "age"}))
        sm["age"] = sm["age"].astype(str)
        sample_means[s] = sm

    merged: dict[str, pd.DataFrame] = {}
    for s in STRATEGIES:
        m = sample_means[s].merge(cr, on=["sample", "genotype", "age"], how="inner")
        assert m.shape[0] == 12, f"{s}:  join  {m.shape[0]} != 12"
        merged[s] = m

    ages = sorted(cr["age"].unique())
    cells = []  #
    for c in CLASSES:
        for g1, g0, tag in CONTRASTS:
            for age in ages:
                sub = merged["centroid"][merged["centroid"]["age"] == age]
                v1 = sub.loc[sub["genotype"] == g1]
                v0 = sub.loc[sub["genotype"] == g0]
                if v1.empty or v0.empty:
                    continue  # 8  cKO
                d_cr = float(v1[f"fate_{c}"].mean() - v0[f"fate_{c}"].mean())
                row = {"fate": c, "contrast": tag, "age": age, "delta_cr": d_cr,
                       "sign_cr": int(np.sign(d_cr))}
                for s in STRATEGIES:
                    m = merged[s]
                    sub_s = m[m["age"] == age]
                    d_sf = float(sub_s.loc[sub_s["genotype"] == g1, c].mean()
                                 - sub_s.loc[sub_s["genotype"] == g0, c].mean())
                    row[f"delta_sfate_{s}"] = d_sf
                    row[f"sign_sfate_{s}"] = int(np.sign(d_sf))
                    row[f"sign_agree_{s}"] = bool(np.sign(d_sf) == np.sign(d_cr))
                cells.append(row)
    assert len(cells) == 15, f" {len(cells)} != 15"

    # ---- 5.  ----
    def arm_summary(s: str) -> dict:
        out = {}
        for c in CLASSES:
            sub = [r for r in cells if r["fate"] == c]
            out[c] = {"agree": sum(1 for r in sub if r[f"sign_agree_{s}"]),
                      "total": len(sub)}
        out["total"] = {"agree": sum(v["agree"] for k, v in out.items() if k != "total"),
                        "total": len(cells)}
        return out

    arm = {s: arm_summary(s) for s in STRATEGIES}

    # sfate  delta  CR
    for r in cells:
        signs = {r[f"sign_sfate_{s}"] for s in STRATEGIES}
        r["sfate_internal_sign_unanimous"] = bool(len(signs) == 1)
        r["sfate_internal_flips"] = {
            s: bool(r[f"sign_sfate_{s}"] != r["sign_sfate_centroid"])
            for s in STRATEGIES if s != "centroid"}
    n_internal_unanimous = sum(1 for r in cells if r["sfate_internal_sign_unanimous"])

    #  A = IFN/C1q  8  4  4/4Homeo
    # cKO−5xFAD 2  0/2 random  medoid
    # sign_agree  centroid
    def _subset_pattern_holds(s: str) -> bool:
        for r in cells:
            key_ok = True
            if r["fate"] in ("IFN_responsive", "C1q_inflammatory") and r["age"] != "8":
                key_ok = bool(r[f"sign_agree_{s}"])  #  4/4
            elif r["fate"] == "Homeostatic" and r["contrast"] == "cKO-5xFAD":
                key_ok = not r[f"sign_agree_{s}"]  #  0/2
            if not key_ok:
                return False
        return True

    branch_a = all(_subset_pattern_holds(s) for s in ("random", "medoid"))
    subset_status = {s: _subset_pattern_holds(s) for s in STRATEGIES}
    flipped_cells = [
        {"fate": r["fate"], "contrast": r["contrast"], "age": r["age"],
         "strategies_disagreeing_with_centroid_pattern": {
             s: {"delta_sfate": r[f"delta_sfate_{s}"],
                 "sign_agree": r[f"sign_agree_{s}"]}
             for s in ("random", "medoid")
             if r[f"sign_agree_{s}"] != r[f"sign_agree_centroid"]},
         "delta_cr": r["delta_cr"], "sign_cr": r["sign_cr"],
         "delta_sfate_centroid": r["delta_sfate_centroid"]}
        for r in cells
        if any(r[f"sign_agree_{s}"] != r[f"sign_agree_centroid"]
               for s in ("random", "medoid"))]

    wall = time.perf_counter() - t_run0

    # ---- 6.  ----
    js = {
        "experiment": "E6b: 75k  × ",
        "env": {"graph_cache": str(GRAPH_CACHE), "n_cells": n,
                "n_drop": n_drop, "n_comp": n_comp,
                "kmeans_seed": SEED_KMEANS, "random_seed": SEED_RANDOM,
                "n_reps_per_class": R, "tol": 1e-6},
        "solve": {s: {"t_solve_s": round(float(sols[s]["t_solve"]), 2),
                      "gmres_iters": [int(x) for x in sols[s]["gmres_iters"]],
                      "residual_max": [float(x) for x in sols[s]["residual_max"]],
                      "n_f64_fallback": int(sols[s]["n_f64_fallback"]),
                      "source": ("fate_full_75k.h5ad " if s == "centroid"
                                 else " sign_robustness_reps_cache.npz")}
                  for s in STRATEGIES},
        "cells": cells,
        "arm_summary": arm,
        "sfate_internal": {"n_sign_unanimous_across_strategies": n_internal_unanimous,
                           "n_cells": len(cells)},
        "branch_a_holds": branch_a,
        "subset_pattern_holds": subset_status,
        "flipped_cells": flipped_cells,
        "wall_s": round(wall, 1),
    }
    OUT_JSON.write_text(json.dumps(js, ensure_ascii=False, indent=2), encoding="utf-8")

    # tsv
    tsv_rows = []
    for r in cells:
        for s in STRATEGIES:
            tsv_rows.append({
                "strategy": s, "fate": r["fate"], "contrast": r["contrast"],
                "age": r["age"],
                "delta_sfate": f"{r[f'delta_sfate_{s}']:+.6f}",
                "delta_cr": f"{r['delta_cr']:+.6f}",
                "sign_agree": r[f"sign_agree_{s}"],
                "sign_agree_same_as_centroid": r[f"sign_agree_{s}"] == r["sign_agree_centroid"],
                "sfate_sign_same_as_centroid": r[f"sign_sfate_{s}"] == r["sign_sfate_centroid"],
            })
    tsv = pd.DataFrame(tsv_rows)
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    header = ("# E6b sfate centroid= / "
              "random seed 20260910 / medoid KMeans30 seed 20260909× 15  "
              "delta CR  step12 \n")
    OUT_TSV.write_text(header + tsv.to_csv(sep="\t", index=False), encoding="utf-8")

    # md
    L = ["# E6b75k \n",
         "## ",
         f"output/c_layer_graph.npz {n}²k=30 latent kNNseed "
         f"20260909 {n_comp}  {n_drop} "
         f" r={R}**centroid** scVI latent  30 "
         " fate_full_75k.h5adcolumns **random**"
         f" 30  {SEED_RANDOM}**medoid** "
         f"KMeans(30, random_state={SEED_KMEANS}, n_init=10)"
         " E6  exp_rep_selection_sensitivity.py "
         "random/medoid  74984  labels  keep "
         "n_drop=0absorption_probabilities(tol=1e-6, columns) "
         " scripts/export_sample_tsv.py  "
         "genotype_delta_sign 3 IFN_responsive / C1q_inflammatory / "
         "Homeostatic× 2 5xFAD−controlcKO−5xFAD× 3 3/6/8 "
         "cKO 8  → 15 CR  step12 "
         "fate_prob_by_sample_WIDE.tsvcentroid "
         "9/15IFN 4/5C1q 4/5Homeo 1/5—— "
         "fate_by_sample_sfate_vs_cellrank.tsv "
         "IFN/C1q  8  4 4/4Homeo  cKO−5xFAD 2 0/2\n",
         "## ",
         "|  | (s) | GMRES  |  max | f64  |",
         "|---|---|---|---|---|"]
    for s in STRATEGIES:
        L.append(f"| {s} | {float(sols[s]['t_solve']):.1f} | "
                 f"{[int(x) for x in sols[s]['gmres_iters']]} | "
                 f"{max(float(x) for x in sols[s]['residual_max']):.2e} | "
                 f"{int(sols[s]['n_f64_fallback'])} |")
    L += ["\n## 15 delta  =  CR ",
          "| fate |  |  | ΔCR | Δsfate centroid | Δsfate random | Δsfate medoid | "
          " centroid |  random |  medoid | sfate  |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in cells:
        L.append(f"| {r['fate']} | {r['contrast']} | {r['age']} | {r['delta_cr']:+.4f} | "
                 f"{r['delta_sfate_centroid']:+.4f} | {r['delta_sfate_random']:+.4f} | "
                 f"{r['delta_sfate_medoid']:+.4f} | "
                 f"{'✓' if r['sign_agree_centroid'] else '✗'} | "
                 f"{'✓' if r['sign_agree_random'] else '✗'} | "
                 f"{'✓' if r['sign_agree_medoid'] else '✗'} | "
                 f"{'✓' if r['sfate_internal_sign_unanimous'] else '✗'} |")
    L += ["\n##  CR ",
          "| fate | centroid | random | medoid |",
          "|---|---|---|---|"]
    for c in CLASSES:
        L.append(f"| {c} | {arm['centroid'][c]['agree']}/{arm['centroid'][c]['total']} | "
                 f"{arm['random'][c]['agree']}/{arm['random'][c]['total']} | "
                 f"{arm['medoid'][c]['agree']}/{arm['medoid'][c]['total']} |")
    L.append(f"| **** | **{arm['centroid']['total']['agree']}/15** | "
             f"**{arm['random']['total']['agree']}/15** | "
             f"**{arm['medoid']['total']['agree']}/15** |")
    L += ["\n## sfate  CR ",
          f"- 15  sfate delta unanimous"
          f"**{n_internal_unanimous}/15**",
          f"- random vs centroid {sum(1 for r in cells if r['sfate_internal_flips']['random'])}",
          f"- medoid vs centroid {sum(1 for r in cells if r['sfate_internal_flips']['medoid'])}"]

    #
    L.append("\n## ")
    subset_desc = (" = IFN_responsive  C1q_inflammatory  8  "
                   "4 4/4+ Homeostatic cKO−5xFAD 2 0/2")
    if branch_a:
        L.append(f"** A **{subset_desc} random  medoid "
                 " centroid "
                 "**75k  delta  "
                 "target representation**centroid-nearest "
                 " 9/15 ")
    else:
        broke = [s for s in ("random", "medoid") if not subset_status[s]]
        L.append(f"** B**{subset_desc} random / medoid "
                 f"{', '.join(broke)}centroid "
                 f"={subset_status['centroid']} target "
                 "representation ****"
                 " |Δ|  1e-4–1e-3 CR "
                 " delta "
                 " centroid  sign_agree ")
        if flipped_cells:
            L.append("| fate |  |  | ΔCR | Δsfate centroid | Δsfate /  |")
            L.append("|---|---|---|---|---|---|")
            for f in flipped_cells:
                det = "; ".join(
                    f"{s}: {v['delta_sfate']:+.4f} / {'✓' if v['sign_agree'] else '✗'}"
                    for s, v in f["strategies_disagreeing_with_centroid_pattern"].items())
                L.append(f"| {f['fate']} | {f['contrast']} | {f['age']} | "
                         f"{f['delta_cr']:+.4f} | {f['delta_sfate_centroid']:+.4f} | {det} |")
        else:
            L.append("——")
    t_solve_total = sum(float(sols[s]["t_solve"]) for s in STRATEGIES)
    L.append(f"\n---\n {t_solve_total:.1f}scentroid 29.0s "
             f"random 856.7smedoid 23.6s ——random  GMRES "
             f" 5  f64 "
             f"{wall:.1f}slabels  {t_labels:.1f}s")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f": {json.dumps(arm, ensure_ascii=False)}")
    print(f"sfate  unanimous: {n_internal_unanimous}/15")
    print(f": {'A' if branch_a else 'B'}: {len(flipped_cells)}")
    print(f"written: {OUT_MD} / {OUT_JSON} / {OUT_TSV} {wall:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
