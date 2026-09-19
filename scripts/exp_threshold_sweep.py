#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_threshold_sweep.py — P1-E1GPCCA  stability 5k fixture
 {0.90, 0.93, 0.96, 0.99}
 →  threshold-dependent R4
output/threshold_sweep.md / .json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

SEED = 20260909
K = 30
THRESHOLDS = [0.90, 0.93, 0.96, 0.99]


def main() -> int:
    import anndata as ad
    import scanpy as sc
    from cellrank import __version__ as cr_ver
    from cellrank import estimators, kernels

    #  b_layer_validation.py  fixture AnnData
    # predict_terminal_states
    adata = ad.read_h5ad("output/fixture_5k.h5ad")

    sc.pp.neighbors(adata, n_neighbors=K, use_rep="X_scVI", random_state=SEED)
    ck = kernels.ConnectivityKernel(adata).compute_transition_matrix(
        density_normalize=True)
    rows = []
    for th in THRESHOLDS:
        # predict_terminal_states —— GPCCA~7s
        g = estimators.GPCCA(ck)
        g.fit(cluster_key="microglia_state", n_states=6)
        try:
            g.predict_terminal_states(method="stability", stability_threshold=th)
            term = sorted(str(s) for s in g.terminal_states.cat.categories)
        except ValueError:
            term = []  # cellrank set_terminal_states([])
        rows.append({"threshold": th, "n_terminal": len(term),
                     "terminal_states": term})
        print(f"threshold={th}: n_terminal={len(term)} {term}", flush=True)
    macro = [str(s) for s in g.macrostates.cat.categories]  #  fit

    Path("output/threshold_sweep.json").write_text(json.dumps({
        "env": {"cellrank": cr_ver, "seed": SEED, "n_states": 6,
                "note": " B eigengap →n_states=6 fit"},
        "macrostates": macro, "sweep": rows}, ensure_ascii=False, indent=2))

    L = ["# E1GPCCA 5k fixtureexp_threshold_sweep.py\n",
         f"- cellrank {cr_ver} / seed {SEED} / n_states=6B / "
         f" fit",
         f"- {macro}\n",
         "| stability  |  |  |", "|---|---|---|"]
    for r in rows:
        L.append(f"| {r['threshold']} | {r['n_terminal']} | {r['terminal_states']} |")
    L.append("\n## ")
    ns = [r["n_terminal"] for r in rows]
    if len(set(ns)) > 1:
        L.append(f"- {ns} → **threshold-dependent**"
                 " R4 ")
    else:
        L.append(f"- {ns} → threshold-stable"
                 "R4 ")
    Path("output/threshold_sweep.md").write_text("\n".join(L), encoding="utf-8")
    print("written: output/threshold_sweep.md", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
