#!/usr/bin/env python
"""C1  step275k columns  + fate_full_75k.h5ad

-  output/c_layer_graph.npz
- aggregated c_layer_sfate_fate.npz
  output/fate_full_75k_aggregated_fallback.h5adP0
- columns  output/fate_full_75k.h5aduns['sfate_meta']  solve_mode


  source ~/sfate_env/bin/activate && OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 /usr/bin/time -v python scripts/m3_solve_75k_columns.py \
    2> output/c_layer_columns_time_v.txt
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import scipy.sparse as sp

GRAPH_CACHE = Path("output/c_layer_graph.npz")
AGG_NPZ = Path("output/c_layer_sfate_fate.npz")  # aggregated
OUT_AGG = Path("output/fate_full_75k_aggregated_fallback.h5ad")
OUT_NEW = Path("output/fate_full_75k.h5ad")
OUT_META = Path("output/fate_full_75k_columns_meta.json")
ANNOTATED = ("<data>/"
             "06_annotation/adata_annotated.h5ad")


def _write_h5ad(path: Path, fate: np.ndarray, classes: list[str], meta: dict) -> None:
    """ AnnDataobs  barcodeobsm['sfate_fate']uns['sfate_meta']"""
    import anndata as ad
    import h5py
    import pandas as pd

    with h5py.File(ANNOTATED, "r") as f:
        g = f["obs"]
        idx = np.asarray(g[g.attrs["_index"]][...]).astype(str)
    keep = meta.pop("_keep")
    assert keep.all(), "75k "
    adata = ad.AnnData(
        obs=pd.DataFrame(index=pd.Index(idx, name="cell_barcode")),
        obsm={"sfate_fate": np.asarray(fate, dtype=np.float32)},
        uns={"sfate_meta": {**meta, "classes": classes}},
    )
    adata.write_h5ad(path)
    print(f"[] {path}n={adata.n_obs}, classes={len(classes)}", flush=True)


def main() -> None:
    # --- aggregated  → fallback h5ad ---
    if not OUT_AGG.exists():
        z = np.load(AGG_NPZ, allow_pickle=True)
        _write_h5ad(OUT_AGG, z["fate"], [str(c) for c in z["classes"]], {
            "solve_mode": "aggregated", "preconditioner": "ilu",
            "note": "P0  RHS  75k "
                    "5/6  f64  3915s columns ",
            "gmres_iters": [int(x) for x in z["gmres_iters"]],
            "residual_max": [float(x) for x in z["residual_max"]],
            "n_f64_fallback": int(z["n_f64_fallback"]),
            "_keep": z["keep"].astype(bool),
        })
    else:
        print(f"[skip] {OUT_AGG} ", flush=True)

    # --- columns  ---
    z = np.load(GRAPH_CACHE, allow_pickle=True)
    P = sp.csr_matrix((z["P_data"], z["P_indices"], z["P_indptr"]),
                      shape=tuple(z["P_shape"]))
    labels = z["labels"]
    keep = z["keep"].astype(bool)
    print(f"[cache] n={P.shape[0]}, nnz={P.nnz}", flush=True)

    from sfate import absorption_probabilities

    t0 = time.perf_counter()
    B, info = absorption_probabilities(P, labels, tol=1e-6, return_info=True,
                                       verbose=True)  #  solve_mode="columns", precond none
    t_solve = time.perf_counter() - t0
    print(f"[columns]  {t_solve:.1f}s {info['gmres_iters'].tolist()}"
          f" {info['gmres_iters_max'].tolist()}f64  {info['n_f64_fallback']}"
          f" max {info['residual_max'].max():.2e}", flush=True)

    # ---  aggregated  ---
    za = np.load(AGG_NPZ, allow_pickle=True)
    diff = np.abs(B.astype(np.float64)[keep] - za["fate"].astype(np.float64)[keep])
    print(f"[] columns vs aggregated L∞ = {diff.max():.2e}p99={np.quantile(diff, 0.99):.2e}",
          flush=True)

    meta = {
        "solve_mode": "columns", "preconditioner": "none", "k": 30,
        "seed": 20260909, "tol": 1e-6,
        "gmres_iters_per_class": [int(x) for x in info["gmres_iters"]],
        "gmres_iters_max_column": [int(x) for x in info["gmres_iters_max"]],
        "residual_max": [float(x) for x in info["residual_max"]],
        "n_f64_fallback": int(info["n_f64_fallback"]),
        "wall_time_s": t_solve,
        "linf_vs_aggregated_snapshot": float(diff.max()),
        "_keep": keep,
    }
    _write_h5ad(OUT_NEW, B, info["classes"], meta)
    meta["classes"] = info["classes"]
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[] {OUT_META}", flush=True)


if __name__ == "__main__":
    main()
