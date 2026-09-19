#!/usr/bin/env python
"""E7CellRank step12 obsp['T_fwd']0.8*VelocityKernel+0.2*ConnectivityKernel
+  target → sfate 75k

2×2  75k
  a. kernel  targetB_cr_ann vs sfate kNN fate_full_75k.h5ad
     obsm['sfate_fate']→ 6  Spearman
  b. target  kernelB_cr_ann vs step12 GPCCA  fate T_fwd
     adata_cellrank.h5ad obs  fate_* → 3  Spearman
     IFN_responsive / C1q_inflammatory / Homeostatic c_layer

<mnt>  + h5py  anndata backedAGENTS.md
 output/ c_layer_validation.select_terminals 30
f64 tests/test_fate.py


  source ~/sfate_env/bin/activate && OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /usr/bin/time -v python scripts/exp_cr_kernel_annotation_targets.py \
    2> output/cr_kernel_annotation_targets_time_v.txt
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import threading
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))          # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from c_layer_validation import ANNOTATED, select_terminals, _rss_sampler, _RSS

CR_H5AD = ("<data>/"
           "11_cellrank/adata_cellrank.h5ad")
SFATE_FATE = Path("output/fate_full_75k.h5ad")
REPORT = Path("output/cr_kernel_annotation_targets.md")
RESULTS = Path("output/cr_kernel_annotation_targets.json")
CR_CLASSES = ["IFN_responsive", "C1q_inflammatory", "Homeostatic"]  #  c_layer


def load_h5ad_obs_index(path: str, key: str | None = None) -> np.ndarray:
    """h5py  obs  obs.attrs['_index'] """
    import h5py
    with h5py.File(path, "r") as f:
        g = f["obs"]
        col = key or g.attrs["_index"]
        return np.asarray(g[col][...]).astype(str)


def load_cr_transition(path: str):
    """h5py  obsp['T_fwd']CSR f64+ uns['T_fwd_params'] """
    import h5py
    import scipy.sparse as sp
    with h5py.File(path, "r") as f:
        g = f["obsp/T_fwd"]
        P = sp.csr_matrix((np.asarray(g["data"][...]),
                           np.asarray(g["indices"][...]),
                           np.asarray(g["indptr"][...])),
                          shape=tuple(g.attrs["shape"]))
        params = {}
        tp = f["uns/T_fwd_params"]
        def _rec(grp, prefix=""):
            for k in grp:
                o = grp[k]
                if hasattr(o, "keys"):
                    _rec(o, prefix + k + ".")
                else:
                    v = o[()]
                    params[prefix + k] = (v.decode() if isinstance(v, bytes)
                                          else (v.tolist() if isinstance(v, np.ndarray) and v.ndim == 0 else str(v)))
        _rec(tp)
    return P, params


def main() -> None:
    t_start = time.perf_counter()
    tracemalloc.start()
    stop = threading.Event()
    th = threading.Thread(target=_rss_sampler, args=(stop,))
    th.start()

    import h5py
    import scipy.sparse.csgraph as csgraph
    from scipy.stats import spearmanr
    from sfate import absorption_probabilities
    from sfate.io import load_latent, load_obs_categorical

    # ---------- 1.  ----------
    t0 = time.perf_counter()
    P_cr, t_params = load_cr_transition(CR_H5AD)          # step12  T_fwd
    idx_cr = load_h5ad_obs_index(CR_H5AD)                  # CR GSM  barcode
    with h5py.File(CR_H5AD, "r") as f:
        X_cr = np.asarray(f["obsm/X_scVI"][...])           # (74984,30) f32
        cr_fate = {c: np.asarray(f["obs"][f"fate_{c}"][...]) for c in CR_CLASSES}
        # CR  microglia_statecategorical
        mg = f["obs"]["microglia_state"]
        states_cr = np.asarray(mg["categories"][...]).astype(str)[np.asarray(mg["codes"][...])]
    X_ann = load_latent(ANNOTATED, "X_scVI")               # 06_annotation
    states_ann = np.asarray(load_obs_categorical(ANNOTATED, "microglia_state").astype(str))
    idx_ann = load_h5ad_obs_index(ANNOTATED)
    with h5py.File(SFATE_FATE, "r") as f:
        idx_sf = np.asarray(f["obs"]["cell_barcode"][...]).astype(str)
        F_sf = np.asarray(f["obsm"]["sfate_fate"][...])    # (74984,6) f32
        meta = f["uns/sfate_meta"]
        sf_classes = [c.decode() if isinstance(c, bytes) else str(c)
                      for c in meta["classes"][...]]
    t_load = time.perf_counter() - t0

    # ---------- 2. barcode  assert----------
    for name, idx in [("cr", idx_cr), ("ann", idx_ann), ("sfate", idx_sf)]:
        assert len(set(idx)) == len(idx), f"{name}  barcode "
    n = len(idx_cr)
    assert set(idx_cr) == set(idx_ann) == set(idx_sf), " barcode "

    def reorder(idx_from: np.ndarray, idx_to: np.ndarray) -> np.ndarray:
        """ perm idx_from[perm] == idx_to"""
        pos = {b: i for i, b in enumerate(idx_from)}
        perm = np.array([pos[b] for b in idx_to])
        assert np.array_equal(idx_from[perm], idx_to), "barcode "
        return perm

    perm_ann = reorder(idx_ann, idx_cr)
    X_ann = X_ann[perm_ann]
    states_ann = states_ann[perm_ann]
    perm_sf = reorder(idx_sf, idx_cr)
    F_sf = F_sf[perm_sf]

    # A2 CR  X_scVI  06_annotation
    assert np.array_equal(X_cr, X_ann), "X_scVI "
    assert np.array_equal(states_cr, states_ann), "microglia_state "
    print(f"[align] n={n}X_scVI/", flush=True)

    # ---------- 3.  ----------
    rs = np.asarray(P_cr.sum(axis=1)).ravel()
    assert np.all(np.abs(rs - 1.0) < 1e-8), "T_fwd "
    print(f"[kernel] T_fwd shape={P_cr.shape} nnz={P_cr.nnz}  OK", flush=True)

    # ---------- 4.  30  b/c ----------
    labels = select_terminals(X_ann, states_ann)
    n_abs = int(sum(x is not None for x in labels))
    print(f"[targets]  {n_abs} 6  × 30", flush=True)

    # ---------- 5.  b_layer _solve_guarded----------
    valid = np.array([x is not None for x in labels], dtype=bool)
    n_comp, comp = csgraph.connected_components(P_cr, directed=False)
    keep = np.isin(comp, np.unique(comp[valid]))
    n_drop = int((~keep).sum())
    print(f"[guard]  {n_comp}  {n_drop} ", flush=True)
    P_k = P_cr[keep][:, keep].tocsr()
    labels_k = labels[keep]

    # ---------- 6. sfate  ----------
    t0 = time.perf_counter()
    B, info = absorption_probabilities(P_k, labels_k, tol=1e-6, return_info=True,
                                       verbose=True)
    t_solve = time.perf_counter() - t0
    B = np.asarray(B, dtype=np.float64)
    classes = [str(c) for c in info["classes"]]
    print(f"[solve] {t_solve:.1f}sGMRES iters={info['gmres_iters']}"
          f"resid_max={max(info['residual_max']):.2e}f64 ={info['n_f64_fallback']}",
          flush=True)

    # ---------- 7a. kernel B vs sfate_fate target kernel----------
    kern_eff = []
    for j, c in enumerate(classes):
        k = sf_classes.index(c)
        rho = float(spearmanr(B[:, j], F_sf[keep, k].astype(np.float64)).statistic)
        kern_eff.append({"class": c, "spearman": rho})
    ks = [x["spearman"] for x in kern_eff]

    # ---------- 7b. target B vs step12 GPCCA fate kernel target----------
    tgt_eff = []
    for c in CR_CLASSES:
        j = classes.index(c)
        rho = float(spearmanr(B[:, j], cr_fate[c][keep]).statistic)
        tgt_eff.append({"class": c, "spearman": rho})
    ts = [x["spearman"] for x in tgt_eff]

    # ---------- 8.  ----------
    _, peak_tm = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    stop.set()
    th.join()
    wall = time.perf_counter() - t_start
    swap = subprocess.run(["swapon", "--show"], capture_output=True, text=True).stdout.strip()

    results = {
        "env": {"python": platform.python_version(), "numpy": np.__version__,
                "sfate_solver": "absorption_probabilities(tol=1e-6)"},
        "transition_matrix": {"source": CR_H5AD, "key": "obsp/T_fwd",
                              "shape": list(P_cr.shape), "nnz": int(P_cr.nnz),
                              "dtype": str(P_cr.dtype),
                              "row_stochastic_max_dev": float(np.abs(rs - 1).max()),
                              "uns_T_fwd_params": t_params,
                              "pipeline_note": "step12 postmortem 0.8*VelocityKernel"
                                               "(gene_subset=1662HVG+11TF)+0.2*ConnectivityKernel"},
        "targets": {"rule": " X_scVI(f64)  30 6 ",
                    "n_absorbing": n_abs},
        "connectivity_guard": {"n_components": int(n_comp), "n_dropped": n_drop,
                               "n_solved": int(keep.sum())},
        "solve": {"time_s": t_solve,
                  "gmres_iters": [int(x) for x in info["gmres_iters"]],
                  "residual_max": [float(x) for x in info["residual_max"]],
                  "n_f64_fallback": int(info["n_f64_fallback"]),
                  "classes": classes},
        "kernel_effect_same_target": {"per_class": kern_eff,
                                      "spearman_min": min(ks),
                                      "spearman_median": float(np.median(ks)),
                                      "spearman_mean": float(np.mean(ks))},
        "target_effect_same_kernel": {"per_pair": tgt_eff,
                                      "spearman_min": min(ts),
                                      "spearman_median": float(np.median(ts)),
                                      "spearman_mean": float(np.mean(ts)),
                                      "note": " 3 step12 GPCCA  3  fate "},
        "times": {"load": t_load, "solve": t_solve, "wall": wall},
        "tracemalloc_peak_mb": peak_tm / 1e6,
        "rss_sample_peak_mb": _RSS["peak_mb"], "swap": swap,
    }
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str))

    L = []
    L.append("# E7CellRank  +  → sfate 75k "
             "exp_cr_kernel_annotation_targets.py \n")
    L.append("## ")
    L.append(f"- step12  `obsp['T_fwd']`{CR_H5AD}——"
             "VelocityKernel(gene_subset=1662HVG+11TF)  ConnectivityKernel  "
             "0.8/0.2 docs/step12_postmortem.md §2shape "
             f"{P_cr.shape[0]}×{P_cr.shape[1]}nnz={P_cr.nnz:,}f64"
             f" {np.abs(rs-1).max():.2e}")
    L.append(f"- uns['T_fwd_params'] vkey={t_params.get('init.vkey')}"
             f"model={t_params.get('params.model')}"
             f"similarity={t_params.get('params.similarity')}"
             f"softmax_scale={t_params.get('params.softmax_scale')}"
             "cellrank  velocity ")
    L.append("-  targetmicroglia_state06_annotation  "
             "X_scVI(f64)  30  180 "
             " b_layer/c_layer/tests ")
    L.append("- T_fwd X_scVImicroglia_statesfate_fate  "
             "GSM  barcode  assert "
             " CR  X_scVI/microglia_state  06_annotation ")
    L.append(f"- T_fwd  {n_comp}  "
             f"**{n_drop}**  {int(keep.sum())} ")
    L.append(f"- sfate absorption_probabilitiestol=1e-6 {t_solve:.1f}s"
             f" GMRES  {results['solve']['gmres_iters']} max "
             f"{max(results['solve']['residual_max']):.2e}f64  "
             f"{info['n_f64_fallback']}")
    L.append(f"-  BLAS=1tracemalloc  {peak_tm/1e6:.0f}MB"
             f"RSS  {_RSS['peak_mb']:.0f}MB {wall:.1f}s\n")
    L.append("## a. kernel  target=CR  vs sfate kNN ")
    L.append(" = output/fate_full_75k.h5ad obsm['sfate_fate']sfate latent kNN  + \n")
    L.append("|  | Spearman |")
    L.append("|---|---|")
    for x in kern_eff:
        L.append(f"| {x['class']} | {x['spearman']:.4f} |")
    L.append(f"\n- **min = {min(ks):.4f} / median = {np.median(ks):.4f} / mean = {np.mean(ks):.4f}**\n")
    L.append("## b. target  kernel=T_fwd vs GPCCA ")
    L.append(" = step12 adata_cellrank.h5ad obs  fate_*  T_fwd "
             "GPCCA  5  3  fate  c_layer\n")
    L.append("| = | Spearman |")
    L.append("|---|---|")
    for x in tgt_eff:
        L.append(f"| {x['class']} | {x['spearman']:.4f} |")
    L.append(f"\n- **min = {min(ts):.4f} / median = {np.median(ts):.4f} / mean = {np.mean(ts):.4f}**")
    L.append("- DAM_like / PU1low_lymphoidGPCCA  unsupported"
             "Proliferating fate —— step12 \n")
    L.append("## ")
    lo_b = min(ts) < 0.9
    L.append(f"- a  min Spearman = {min(ks):.3f}median {np.median(ks):.3f}"
             "75k  kernel ****—— target  CR  vs sfate "
             "latent kNN  fate Homeostatic/IFN_responsive "
             "≈0.24–0.28 5k  2 ConnectivityKernel min "
             "0.830E7  0.8  velocity velocity "
             " fate ——**kernel  "
             "velocity**5k  ConnectivityKernel ")
    L.append(f"- b  min Spearman = {min(ts):.3f}median {np.median(ts):.3f}"
             + (" CR  target GPCCA  ↔ "
                "——**target  kernel **min −0.084 vs "
                "0.240 5k 5k 0.136 <<  0.830"
                " 75k  2×2 "
                if lo_b else
                " kernel  target  75k "))
    L.append("- 75k  target —— 5k "
             " kernel  5k  75k velocity "
             " ConnectivityKernel  velocity ")
    L.append(f"-  max {max(results['solve']['residual_max']):.2e}"
             " tol=1e-6f64  GMRES  75k  "
             "sfate kNN  75k  1.85e-07 fate_full_75k uns['sfate_meta']——"
             " Spearman ")
    L.append("- b  3  step12 "
             "GPCCA ")
    REPORT.write_text("\n".join(L) + "\n")

    print(f"kernel  Spearman min/med/mean: {min(ks):.4f}/{np.median(ks):.4f}/{np.mean(ks):.4f}")
    print(f"target  Spearman min/med/mean: {min(ts):.4f}/{np.median(ts):.4f}/{np.mean(ts):.4f}")
    print(f"written: {REPORT} / {RESULTS} {wall:.1f}s")


if __name__ == "__main__":
    main()
