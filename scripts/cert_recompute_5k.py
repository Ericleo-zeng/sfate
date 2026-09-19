#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cert_recompute_5k.py — P0-3 1


- a2 n=1600 f64 GMRES tol=1e-12 + ILU
   solve_mode  maxκ∞(I−Q)M-
  10·κ∞·backwardL∞ vs spsolve(f64)PASS
- a3 fixture_5kf32 GMRES tol=1e-6 columns
    +  η=‖Ax−b‖/(‖A‖‖x‖+‖b‖)κ∞10·κ∞·η
  L∞ vs  1e-5 PASS

output/cert_recompute_5k.md / .json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, spsolve

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))

from sfate import (absorption_probabilities, absorb_states,
                   build_transition_graph, split_transient_absorbing)

SEED = 20260909
OUT_MD = ROOT / "output/cert_recompute_5k.md"
OUT_JSON = ROOT / "output/cert_recompute_5k.json"


def kappa_inf(A: sp.csr_matrix) -> float:
    """M-‖A‖∞·max(A⁻¹·1) tests/test_fate.py::_kappa_inf"""
    lu = splu(A.tocsc())
    n1 = lu.solve(np.ones(A.shape[0]))
    row_abs = np.asarray(np.abs(A).sum(axis=1)).ravel()
    return float(row_abs.max() * n1.max())


def main() -> int:
    from test_fate import _chain, _golden_f64, _class_indicator, _csc_column_dense

    t0 = time.perf_counter()
    report = {"env": {"seed": SEED, "date": time.strftime("%Y-%m-%d")}}

    # ============ a2 ============
    strict = {}
    for mode in ("aggregated", "columns"):
        P, labels = _chain(n=2000, n_term_cells=100)
        gold, trans_idx, classes = _golden_f64(P, labels)
        # A = I−Q
        valid = np.array([x is not None for x in labels], dtype=bool)
        vidx = np.nonzero(valid)[0]
        Pa = absorb_states(P.astype(np.float64), vidx)
        Q, R, _, _ = split_transient_absorbing(Pa, vidx)
        A = (sp.eye(Q.shape[0], format="csr", dtype=np.float64) - Q).tocsr()
        B, info = absorption_probabilities(
            P, labels, tol=1e-12, return_info=True,
            solve_mode=mode, preconditioner="ilu")
        backward = float(info["residual_max"].max())
        kap = kappa_inf(A)
        bound = 10 * kap * backward
        linf = float(np.abs(B[trans_idx].astype(np.float64) - gold).max())
        gate = 1e-12 if mode == "aggregated" else 1e-12 * 100
        strict[mode] = {
            "backward_error_max": backward,
            "kappa_inf": kap,
            "10_kappa_backward": bound,
            "Linf_vs_spsolve": linf,
            "backward_gate": gate,
            "PASS": bool(backward <= gate and linf <= bound),
        }
        print(f"[strict/{mode}] backward={backward:.2e} κ∞={kap:.2e} "
              f"10κb={bound:.2e} L∞={linf:.2e} PASS={strict[mode]['PASS']}",
              flush=True)
    report["strict_a2"] = strict

    # ============ a3 5k fixture============
    import anndata as ad

    fx = ad.read_h5ad(ROOT / "output/fixture_5k.h5ad")
    X = np.asarray(fx.obsm["X_scVI"], dtype=np.float32)
    states = np.asarray(fx.obs["microglia_state"].astype(str))
    n = X.shape[0]
    labels = np.array([None] * n, dtype=object)
    X64 = X.astype(np.float64)
    for c in np.unique(states):
        idx = np.nonzero(states == c)[0]
        centroid = X64[idx].mean(axis=0)
        dist = np.linalg.norm(X64[idx] - centroid, axis=1)
        reps = idx[np.argsort(dist, kind="stable")[:30]]
        for i in reps:
            labels[i] = c

    P = build_transition_graph(X, k=30, random_state=SEED)
    gold, trans_idx, classes = _golden_f64(P, labels)
    valid = np.array([x is not None for x in labels], dtype=bool)
    vidx = np.nonzero(valid)[0]
    Pa = absorb_states(P.astype(np.float64), vidx)
    Q, R, _, _ = split_transient_absorbing(Pa, vidx)
    A = (sp.eye(Q.shape[0], format="csr", dtype=np.float64) - Q).tocsr()
    B, info = absorption_probabilities(P, labels, tol=1e-6, return_info=True)
    linf = float(np.abs(B[trans_idx].astype(np.float64) - gold).max())
    backward_raw = float(info["residual_max"].max())
    # η_c = ‖Ax−b‖ / (‖A‖‖x‖+‖b‖)
    a_norm = float(np.asarray(np.abs(A).sum(axis=1)).ravel().max())
    cls_id = np.array([np.nonzero(classes == labels[i])[0][0] for i in vidx],
                      dtype=np.int64)
    S = _class_indicator(cls_id, classes.shape[0])
    R_agg = (R @ S).tocsc()
    etas = []
    for ci in range(len(classes)):
        x = B[trans_idx][:, ci].astype(np.float64)
        b = _csc_column_dense(R_agg, ci, np.float64)
        res = float(np.abs(A @ x - b).max())
        etas.append(res / (a_norm * float(np.abs(x).max()) + float(np.abs(b).max())))
    eta_max = float(max(etas))
    kap = kappa_inf(A)
    bound = 10 * kap * eta_max
    prod = {
        "Linf_vs_golden": linf,
        "backward_raw_max": backward_raw,
        "backward_normalized_max": eta_max,
        "kappa_inf": kap,
        "10_kappa_eta": bound,
        "lenient_gate_Linf": 1e-5,
        "margin_vs_1e-5": 1e-5 / linf,
        "PASS": bool(linf <= 1e-5),
    }
    print(f"[prod 5k] L∞={linf:.2e} raw={backward_raw:.2e} η={eta_max:.2e} "
          f"κ∞={kap:.2e} 10κη={bound:.2e} ={prod['margin_vs_1e-5']:.1f}×",
          flush=True)
    report["production_5k"] = prod
    report["elapsed_s"] = round(time.perf_counter() - t0, 1)

    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    L = []
    L.append("# cert_recompute_5k.py \n")
    L.append(f"-  {report['env']['date']}seed={SEED}")
    L.append("\n## a2 n=1600 f64 GMRES tol=1e-12 + ILU\n")
    L.append("| solve_mode |  max | κ∞(I−Q) | 10·κ∞·backward | L∞ vs spsolve |  | PASS |")
    L.append("|---|---|---|---|---|---|---|")
    for mode, r in strict.items():
        L.append(f"| {mode} | {r['backward_error_max']:.2e} | {r['kappa_inf']:.2e} | "
                 f"{r['10_kappa_backward']:.2e} | {r['Linf_vs_spsolve']:.2e} | "
                 f"{r['backward_gate']:.0e} | {'✅' if r['PASS'] else '❌'} |")
    L.append("\n## a3 fixture_5kf32 GMRES tol=1e-6columns\n")
    L.append("|  |  |")
    L.append("|---|---|")
    L.append(f"| L∞ vs f64  | {prod['Linf_vs_golden']:.2e} |")
    L.append(f"|  max  | {prod['backward_raw_max']:.2e} |")
    L.append(f"|  η=max ‖Ax−b‖/(‖A‖‖x‖+‖b‖) | {prod['backward_normalized_max']:.2e} |")
    L.append(f"| κ∞(I−Q)M- | {prod['kappa_inf']:.2e} |")
    L.append(f"| 10·κ∞·η | {prod['10_kappa_eta']:.2e} |")
    L.append(f"|  L∞ ≤ 1e-5 |  {prod['margin_vs_1e-5']:.1f}× |")
    L.append(f"| PASS | {'✅' if prod['PASS'] else '❌'} |")
    L.append("\n## \n")
    L.append("- ≤1e-12  GMRES f64 ")
    L.append("- tol=1e-6, f32 ~1e-7 \n"
             "  L∞ ≤ 1e-5 ——1e-12 1e-5 ")
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"written: {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
