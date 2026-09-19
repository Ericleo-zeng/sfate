#!/usr/bin/env python
"""C 75k sfate vs step12  CellRank

docs/step12_postmortem.md75k + output/b_layer_report.md
B /

<mnt>  + h5py  anndata backedAGENTS.md
 output/S2  npz +


  source ~/sfate_env/bin/activate && OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /usr/bin/time -v python scripts/c_layer_validation.py 2> output/c_layer_time_v.txt
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

import numpy as np

SEED = 20260909
K = 30
N_REPS_PER_CLASS = 30
ANNOTATED = ("<data>/"
             "06_annotation/adata_annotated.h5ad")
CR_H5AD = ("<data>/"
           "11_cellrank/adata_cellrank_INTEGRATED_FINAL.h5ad")
CR_WIDE = ("<data>/"
           "11_cellrank/fate_prob_by_sample_WIDE.tsv")
CR_REPORT = ("<data>/"
             "11_cellrank/cellrank_integrated_final_report.json")
CACHE = Path("output/c_layer_sfate_fate.npz")
CACHE_GRAPH = Path("output/c_layer_graph.npz")  # S1 P + labels + keep
REPORT = Path("output/c_layer_report.md")
RESULTS = Path("output/c_layer_results.json")
CR_CLASSES = ["IFN_responsive", "C1q_inflammatory", "Homeostatic"]  # CR  fate  3

_RSS = {"peak_mb": 0.0}


def _rss_sampler(stop: threading.Event) -> None:
    import psutil
    proc = psutil.Process()
    while not stop.is_set():
        _RSS["peak_mb"] = max(_RSS["peak_mb"], proc.memory_info().rss / 1e6)
        stop.wait(0.25)


def select_terminals(X: np.ndarray, states: np.ndarray) -> np.ndarray:
    """ state  30  tests/test_fate.py / b_layer"""
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


def s1_s2_solve() -> dict:
    """ +  npz  / fate fate  dict"""
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        print(f"[cache]  {CACHE}+")
        return {k: z[k] for k in z.files}

    from sfate import absorption_probabilities
    import scipy.sparse as sp

    if CACHE_GRAPH.exists():
        z = np.load(CACHE_GRAPH, allow_pickle=True)
        print(f"[cache]  {CACHE_GRAPH}")
        P = sp.csr_matrix((z["P_data"], z["P_indices"], z["P_indptr"]),
                          shape=tuple(z["P_shape"]))
        labels = z["labels"]
        keep = z["keep"]
        n_drop, n_comp = int(z["n_drop"]), int(z["n_comp"])
        t_load = float(z["t_load"])
        t_graph = float(z["t_graph"])
    else:
        from sfate import build_transition_graph
        from sfate.io import load_latent, load_obs_categorical

        t0 = time.perf_counter()
        X = load_latent(ANNOTATED, "X_scVI")
        states = np.asarray(load_obs_categorical(ANNOTATED, "microglia_state").astype(str))
        t_load = time.perf_counter() - t0

        t0 = time.perf_counter()
        P = build_transition_graph(X, k=K, random_state=SEED)
        t_graph = time.perf_counter() - t0

        labels = select_terminals(X, states)

        # M2  raise
        import scipy.sparse.csgraph as csgraph
        valid = np.array([x is not None for x in labels], dtype=bool)
        n_comp, comp = csgraph.connected_components(P, directed=False)
        keep = np.isin(comp, np.unique(comp[valid]))
        n_drop = int((~keep).sum())
        if n_drop:
            print(f"[guard]  {n_drop}  {n_comp} ")
            P = P[keep][:, keep].tocsr()
            labels = labels[keep]
        else:
            keep = np.ones(P.shape[0], dtype=bool)
        np.savez_compressed(CACHE_GRAPH, P_data=P.data, P_indices=P.indices,
                            P_indptr=P.indptr, P_shape=np.array(P.shape),
                            labels=labels, keep=keep,
                            n_drop=np.int64(n_drop), n_comp=np.int64(n_comp),
                            t_load=t_load, t_graph=t_graph)
        print(f"[cache]  {CACHE_GRAPH}")

    t0 = time.perf_counter()
    B, info = absorption_probabilities(P, labels, tol=1e-6, return_info=True,
                                       verbose=True)
    t_solve = time.perf_counter() - t0

    out = {
        "fate": B, "classes": np.array(info["classes"]),
        "keep": keep, "n_drop": np.int64(n_drop), "n_comp": np.int64(n_comp),
        "gmres_iters": info["gmres_iters"],
        "residual_max": info["residual_max"],
        "n_f64_fallback": np.int64(info["n_f64_fallback"]),
        "times": np.array([t_load, t_graph, t_solve]),  # load/graph/solve
    }
    np.savez_compressed(CACHE, **out)
    print(f"[cache]  {CACHE}")
    return out


def s3_spearman(sf: dict) -> dict:
    """ Spearmansfate 6  vs CR 3  fate_*  barcode """
    print("[stage] S3 ", flush=True)
    import h5py
    from scipy.stats import pearsonr, spearmanr

    with h5py.File(ANNOTATED, "r") as f:
        g = f["obs"]
        idx_ann = np.asarray(g[g.attrs["_index"]][...]).astype(str)
    with h5py.File(CR_H5AD, "r") as f:
        g = f["obs"]
        idx_cr = np.asarray(g[g.attrs["_index"]][...]).astype(str)
        cr_fate = {c: np.asarray(g[f"fate_{c}"][...]) for c in CR_CLASSES}
        commit = np.asarray(g["commitment_score"][...])

    keep = sf["keep"].astype(bool)
    idx_sf = idx_ann[keep]  # sfate
    # dict 75k
    common = np.intersect1d(idx_sf, idx_cr)
    map_sf = {b: i for i, b in enumerate(idx_sf)}
    map_cr = {b: i for i, b in enumerate(idx_cr)}
    rows_sf = np.array([map_sf[b] for b in common])
    rows_cr = np.array([map_cr[b] for b in common])

    F = sf["fate"]
    classes = [str(c) for c in sf["classes"]]
    per_pair = []
    for c in CR_CLASSES:
        j = classes.index(c)
        v_sf = F[rows_sf, j].astype(np.float64)
        v_cr = cr_fate[c][rows_cr]
        per_pair.append({
            "class": c,
            "spearman": float(spearmanr(v_sf, v_cr).statistic),
            "pearson": float(pearsonr(v_sf, v_cr).statistic),
        })
    return {"n_common": int(len(common)), "n_sf": int(len(idx_sf)),
            "n_cr": int(len(idx_cr)), "per_pair": per_pair,
            "unmatched_classes": [c for c in classes if c not in CR_CLASSES],
            "commitment_score_cr": {"mean": float(commit.mean()),
                                    "frac_ge_08": float((commit >= 0.8).mean())}}


def s4_sample_level(sf: dict) -> dict:
    """sfate  vs WIDE.tsv delta """
    import h5py
    import pandas as pd

    with h5py.File(ANNOTATED, "r") as f:
        g = f["obs"]
        idx_ann = np.asarray(g[g.attrs["_index"]][...]).astype(str)
    from sfate.io import load_obs_columns
    obs = load_obs_columns(ANNOTATED, ["sample", "genotype", "age_months"])
    obs["__idx"] = idx_ann

    keep = sf["keep"].astype(bool)
    classes = [str(c) for c in sf["classes"]]
    df = pd.DataFrame(sf["fate"], columns=classes)
    df["sample"] = obs["sample"].values[keep]
    df["genotype"] = obs["genotype"].values[keep]
    df["age"] = obs["age_months"].values[keep].astype(str)
    sf_by_sample = df.groupby(["sample", "genotype", "age"], observed=True)[classes].mean()

    wide = pd.read_csv(CR_WIDE, sep="\t")
    wide["age"] = wide["age"].astype(str)
    wide = wide.set_index(["sample", "genotype", "age"])

    #  delta 5xFAD − controlcKO − 5xFAD
    # 8  cKO  KeyError—— None
    def _delta_signs(frame: pd.DataFrame, col: str, ages, g1, g0) -> list:
        signs = []
        for age in ages:
            sub = frame.xs(age, level="age")
            try:
                v1 = sub.xs(g1, level="genotype")[col].mean()
                v0 = sub.xs(g0, level="genotype")[col].mean()
            except KeyError:
                signs.append(None)  #
                continue
            signs.append(int(np.sign(v1 - v0)))
        return signs

    ages = sorted(wide.reset_index()["age"].unique())
    rows = []
    for c in CR_CLASSES:
        for (g1, g0, tag) in [("5xFAD", "control", "5xFAD−control"),
                              ("5xFAD;Cd28-cKO", "5xFAD", "cKO−5xFAD")]:
            s_sf = _delta_signs(sf_by_sample, c, ages, g1, g0)
            s_cr = _delta_signs(wide, f"fate_{c}", ages, g1, g0)
            paired = [(a, b) for a, b in zip(s_sf, s_cr) if a is not None and b is not None]
            agree = sum(1 for a, b in paired if a == b)
            rows.append({"class": c, "contrast": tag, "ages": list(ages),
                         "sfate_signs": s_sf, "cr_signs": s_cr,
                         "agree": agree, "total": len(paired)})
    return {"ages": list(ages), "rows": rows,
            "sfate_sample_means": sf_by_sample.reset_index().to_dict("records"),
            "n_samples": int(wide.shape[0])}


def s5_pathology() -> dict:
    """GPCCA  75k macrostates_fwd × microglia_state  + """
    import h5py
    import pandas as pd

    with h5py.File(CR_H5AD, "r") as f:
        g = f["obs"]["macrostates_fwd"]
        codes = np.asarray(g["codes"][...])
        cats = np.asarray(g["categories"][...]).astype(str)
    macro = pd.Categorical.from_codes(codes, categories=cats)
    from sfate.io import load_obs_categorical
    states = load_obs_categorical(ANNOTATED, "microglia_state").astype(str)
    # ==macrostates_fwd  30  NaNCR n_cells=30
    ct = pd.crosstab(pd.Series(np.asarray(macro), name="macrostate"),
                     pd.Series(np.asarray(states), name="annotation"))
    rep = json.load(open(CR_REPORT))
    return {"crosstab": ct.to_dict(), "n_unassigned": int((codes < 0).sum()),
            "cr_terminal_states": rep["terminal_states"],
            "cr_macrostates": rep["macrostates"],
            "cr_unsupported": rep.get("unsupported_states", []),
            "cr_fate_columns": rep["fate_columns"]}


def main() -> None:
    t_start = time.perf_counter()
    tracemalloc.start()
    stop = threading.Event()
    th = threading.Thread(target=_rss_sampler, args=(stop,))
    th.start()

    import scipy
    import platform

    sf = s1_s2_solve()
    print("[stage] S1/S2 ", flush=True)
    r3 = s3_spearman(sf)
    print("[stage] S3 ", flush=True)
    r4 = s4_sample_level(sf)
    print("[stage] S4 ", flush=True)
    r5 = s5_pathology()
    print("[stage] S5 ", flush=True)

    _, peak_tm = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    stop.set()
    th.join()
    wall = time.perf_counter() - t_start
    swap = subprocess.run(["swapon", "--show"], capture_output=True, text=True).stdout.strip()

    t_load, t_graph, t_solve = (float(x) for x in sf["times"])
    classes = [str(c) for c in sf["classes"]]
    sp_min = min(p["spearman"] for p in r3["per_pair"])
    sp_med = float(np.median([p["spearman"] for p in r3["per_pair"]]))
    agree_total = sum(r["agree"] for r in r4["rows"])
    agree_n = sum(r["total"] for r in r4["rows"])

    results = {
        "env": {"python": platform.python_version(), "scipy": scipy.__version__,
                "numpy": np.__version__, "seed": SEED,
                "cellrank_note": "C  CellRank 2.x "},
        "sfate": {"classes": classes, "gmres_iters": [int(x) for x in sf["gmres_iters"]],
                  "residual_max": [float(x) for x in sf["residual_max"]],
                  "n_f64_fallback": int(sf["n_f64_fallback"]),
                  "n_orphan_dropped": int(sf["n_drop"]), "n_comp": int(sf["n_comp"]),
                  "times": {"load": t_load, "graph": t_graph, "solve": t_solve}},
        "cell_level": r3, "sample_level": r4, "pathology": r5,
        "sign_agree": {"agree": agree_total, "total": agree_n},
        "wall_s": wall, "tracemalloc_peak_mb": peak_tm / 1e6,
        "rss_sample_peak_mb": _RSS["peak_mb"], "swap": swap,
    }
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str))

    # ---  ---
    L = []
    L.append("# C 75k c_layer_validation.py \n")
    L.append("## ")
    L.append(f"- scipy {scipy.__version__} / numpy {np.__version__} /  {SEED} /  BLAS=1")
    L.append("- CellRank  step12 postmortem  75k  ~405GB "
             " cellrank 2.1.0 vs  2.3.2 ")
    L.append(f"- <mnt> h5py  anndata backed output/")
    L.append(f"- S3–S5{wall:.1f}sS1/S2 "
             f" RSS ≈1.89GB [ time -v]"
             f" {max(peak_tm/1e6, _RSS['peak_mb']):.0f}MBtime -v  output/c_layer_time_v.txt")
    L.append(f"- swap`{' ; '.join(l for l in swap.splitlines() if l.strip())}`\n")
    L.append("## sfate 75k S1/S2")
    L.append("|  | (s) |")
    L.append("|---|---|")
    L.append(f"| load latent+obsh5py  | {t_load:.1f} |")
    L.append(f"| build_transition_graph k=30 | {t_graph:.1f} |")
    L.append(f"| absorption_probabilities 6  | {t_solve:.1f} |")
    L.append(f"\n- {int(sf['n_comp'])}  {int(sf['n_drop'])} ")
    L.append(f"- **75k GMRES design P0 ** "
             f"{[int(x) for x in sf['gmres_iters']]} max "
             f"{max(float(x) for x in sf['residual_max']):.2e}f64  "
             f"{int(sf['n_f64_fallback'])}\n")
    L.append("## S3sfate vs step12 fate_* barcode ")
    L.append(f"- sfate {r3['n_sf']} ∩ CR {r3['n_cr']} →  {r3['n_common']} ")
    L.append("|  | Spearman | Pearson |")
    L.append("|---|---|---|")
    for p in r3["per_pair"]:
        L.append(f"| {p['class']} | {p['spearman']:.4f} | {p['pearson']:.4f} |")
    L.append(f"\n-  {sp_min:.4f} /  {sp_med:.4f}CR  fate "
             f"{r3['unmatched_classes']}")
    L.append(f"- CR commitment_score {r3['commitment_score_cr']['mean']:.3f}"
             f"≥0.8  {r3['commitment_score_cr']['frac_ge_08']:.3f}")
    L.append("- C ——B "
             "GPCCA  DAM/PU1low/Proliferating\n")
    L.append("## S4")
    L.append(f"- 12  delta **{agree_total}/{agree_n}**")
    L.append("|  |  |  | sfate  | step12  |  |")
    L.append("|---|---|---|---|---|---|")
    for r in r4["rows"]:
        L.append(f"| {r['class']} | {r['contrast']} | {','.join(r['ages'])} | "
                 f"{r['sfate_signs']} | {r['cr_signs']} | {r['agree']}/{r['total']} |")
    L.append("")
    L.append("## GPCCA  75k S5")
    L.append(f"- step12 {r5['cr_terminal_states']}"
             f"unsupported{r5['cr_unsupported']}fate {r5['cr_fate_columns']}")
    L.append(f"- sfate6 {classes}——")
    L.append("\n###  × macrostate 75kmacrostates_fwd  30 "
             f" {r5['n_unassigned']} ——CR n_cells=30 ")
    ct = r5["crosstab"]
    ann_cols = sorted({a for v in ct.values() for a in v})
    L.append("|  \\  | " + " | ".join(ann_cols) + " |")
    L.append("|" + "---|" * (len(ann_cols) + 1))
    for m, row in ct.items():
        L.append(f"| {m} | " + " | ".join(str(row.get(a, 0)) for a in ann_cols) + " |")
    #
    b_min = 0.136  # B  3 output/b_layer_report.md
    L.append("\n### ")
    L.append(f"- B  5kGPCCA  1  Spearman min = {b_min}")
    L.append(f"- C  75kstep12  5 IFN  3 DAM_like/PU1low_lymphoid "
             f"unsupportedProliferating  fate "
             f" 75k  = **** 5k "
             f"DAM/PU1low IFN ")
    L.append("-  postmortem §6.3 GPCCA  Schur "
             "——6  2–3  "
             "transition/unsupportedsfate  6 ")
    REPORT.write_text("\n".join(L) + "\n")
    print(f"Spearman min/med: {sp_min:.4f}/{sp_med:.4f} |  {agree_total}/{agree_n}")
    print(f"written: {REPORT} / {RESULTS} {wall:.1f}s")


if __name__ == "__main__":
    main()
