#!/usr/bin/env python
"""B sfate vs CellRank  fate E2/E4 PR CI


- CellRank 2.1.0scanpy neighbors → ConnectivityKernel → GPCCA(brandts, eigengap
  n_states) → predict_terminal_states(stability≥0.96) → compute_fate_probabilities
  (gmres+ilu, use_petsc=False, tol=1e-6)
- sfatelatent kNN(gaussian/max, k=30, f32) → absorption_probabilities6
  microglia_state  30  tests/test_fate.py

macrostate memberships × microglia_stateHungarian
E4 Spearman  >0.95  / 0.90–0.95  / <0.90

BLAS=1time -v
  source ~/sfate_env/bin/activate && OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /usr/bin/time -v python scripts/b_layer_validation.py 2> output/b_layer_time_v.txt
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

SEED = 20260909
K = 30
N_REPS_PER_CLASS = 30
FIXTURE = Path("output/fixture_5k.h5ad")
REPORT = Path("output/b_layer_report.md")
RESULTS = Path("output/b_layer_results.json")


def select_terminals(X: np.ndarray, states: np.ndarray) -> np.ndarray:
    """ state  30  tests/test_fate.py"""
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


def run_cellrank(adata) -> dict:
    """CellRank  fate  (n, n_term) membership """
    import scanpy as sc
    from cellrank import estimators, kernels

    t0 = time.perf_counter()
    sc.pp.neighbors(adata, n_neighbors=K, use_rep="X_scVI", random_state=SEED)
    t_neighbors = time.perf_counter() - t0

    t0 = time.perf_counter()
    ck = kernels.ConnectivityKernel(adata).compute_transition_matrix(
        density_normalize=True
    )
    t_kernel = time.perf_counter() - t0

    t0 = time.perf_counter()
    g = estimators.GPCCA(ck)
    g.fit(cluster_key="microglia_state", n_states=None)  # eigengap
    n_states_auto = int(len(g.macrostates.cat.categories))
    if n_states_auto < 3:
        # eigengap  1 →  []
        print(f"[warn] eigengap  {n_states_auto}  n_states=6",
              file=sys.stderr)
        g = estimators.GPCCA(ck)
        g.fit(cluster_key="microglia_state", n_states=6)
        n_states_auto = 6
        n_states_note = " n_states=6eigengap  1 "
    else:
        n_states_note = f"eigengap  n_states={n_states_auto}"
    t_gpcca_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    g.predict_terminal_states(method="stability", stability_threshold=0.96)
    terminal_primary = [str(s) for s in g.terminal_states.cat.categories]
    t_terminal = time.perf_counter() - t0

    memberships = np.asarray(g.macrostates_memberships)  # (n, n_macro)
    macro_names = [str(s) for s in g.macrostates.cat.categories]
    return {
        "estimator": g,
        "memberships": memberships,
        "macro_names": macro_names,
        "terminal_primary": terminal_primary,
        "n_states_note": n_states_note,
        "times": {"neighbors": t_neighbors, "kernel": t_kernel,
                  "gpcca_fit": t_gpcca_fit, "terminal": t_terminal},
    }


def cr_fate_for_terminals(g, terminal_names) -> tuple[np.ndarray, list[str], float]:
    """ fate gmres+ilu, use_petsc=False, tol=1e-6"""
    t0 = time.perf_counter()
    g.set_terminal_states(terminal_names)
    g.compute_fate_probabilities(
        solver="gmres", use_petsc=False, preconditioner="ilu", tol=1e-6,
        n_jobs=1, backend="threading", show_progress_bar=False,
    )
    return (np.asarray(g.fate_probabilities),
            [str(s) for s in g.fate_probabilities.names], time.perf_counter() - t0)


def run_sfate(X: np.ndarray, labels: np.ndarray) -> dict:
    from sfate import absorption_probabilities, build_transition_graph

    t0 = time.perf_counter()
    P = build_transition_graph(X, k=K, random_state=SEED)
    t_graph = time.perf_counter() - t0
    t0 = time.perf_counter()
    B, info = absorption_probabilities(P, labels, tol=1e-6, return_info=True)
    t_solve = time.perf_counter() - t0
    return {"fate": np.asarray(B, dtype=np.float64), "classes": info["classes"],
            "info_iters": [int(x) for x in info["gmres_iters"]],
            "info_resid": [float(x) for x in info["residual_max"]],
            "n_f64_fallback": info["n_f64_fallback"],
            "times": {"graph": t_graph, "solve": t_solve}}


def hungarian_map(conf: np.ndarray, term_idx: list[int], term_names: list[str],
                  classes: list[str]):
    """terminal  ↔   Hungarian """
    from scipy.optimize import linear_sum_assignment

    sub = conf[:, term_idx]  # (n_classes, n_term)
    r, c = linear_sum_assignment(-sub)
    pairs, unmatched_term, matched_cls = [], [], set()
    for ri, ci in zip(r, c):
        pairs.append({"terminal": term_names[ci], "class": classes[ri],
                      "overlap": float(sub[ri, ci]),
                      "col_term": int(ci), "col_class": int(ri)})
        matched_cls.add(classes[ri])
    matched_term_cols = {int(ci) for ci in c}
    for j, name in enumerate(term_names):
        if j not in matched_term_cols:
            unmatched_term.append(name)
    unmatched_cls = [c0 for c0 in classes if c0 not in matched_cls]
    return pairs, unmatched_term, unmatched_cls


def run_ablation(adata, res_cr, res_sf, labels_sf, classes, mapping) -> dict:
    """2×2 {CR , sfate } × {CR , } sfate

     1E1 sfate (P_cr, T_cr) vs CellRank fate ——  P  A
         L∞≲1e-4 CR
     2 = (P_cr) vs (P_sf)  Spearman
     3 = P_cr(T_cr) vs (T_sf)  Spearman
    """
    import scipy.sparse as sp
    import scipy.sparse.csgraph as csgraph
    from scipy.stats import spearmanr
    from sfate import absorption_probabilities

    g = res_cr["estimator"]
    macro_names = res_cr["macro_names"]
    P_cr = sp.csr_matrix(g.kernel.transition_matrix)

    # CR  membership top-30CR n_cells=30
    term_memb = np.asarray(g.terminal_states_memberships)  # (n, n_term)
    term_names = [str(s) for s in g.terminal_states_memberships.names]
    n = P_cr.shape[0]
    labels_cr = np.array([None] * n, dtype=object)
    for j, nm in enumerate(term_names):
        top = np.argsort(term_memb[:, j], kind="stable")[-30:]
        for i in top:
            labels_cr[i] = nm

    # CR  →
    def _solve_guarded(P, labels):
        n_comp, comp = csgraph.connected_components(P, directed=False)
        valid = np.array([x is not None for x in labels], dtype=bool)
        keep = np.isin(comp, np.unique(comp[valid]))
        n_drop = int((~keep).sum())
        if n_drop:
            P = P[keep][:, keep].tocsr()
            labels = labels[keep]
        B, info = absorption_probabilities(P, labels, tol=1e-6, return_info=True)
        return B, info, keep, n_drop

    out = {}
    #  1 P  A
    B1, info1, keep1, drop1 = _solve_guarded(P_cr, labels_cr)
    F_cr_full = np.asarray(g.fate_probabilities)
    # B1  np.unique CR fate  fate_probabilities.names
    fate_names = [str(s) for s in g.fate_probabilities.names]
    col_order = [fate_names.index(nm) for nm in sorted(set(term_names))]
    L1 = np.abs(B1 - F_cr_full[keep1][:, col_order]).max()
    out["e1_cross"] = {"Linf": float(L1), "n_dropped_orphan": drop1,
                       "n_f64_fallback": info1["n_f64_fallback"],
                       "residual_max": float(max(info1["residual_max"]))}
    #  2
    B2, _, keep2, drop2 = _solve_guarded(P_cr, labels_sf)
    B_sf = res_sf["fate"]  # (P_sf, T_sf)sfate  5000
    kern_eff = []
    for j, c in enumerate(res_sf["classes"]):
        rho = spearmanr(B2[:, j], B_sf[keep2, j]).statistic
        kern_eff.append({"class": c, "spearman": float(rho)})
    out["kernel_effect"] = {"per_class": kern_eff, "n_dropped_orphan": drop2,
                            "min_spearman": min(x["spearman"] for x in kern_eff)}
    #  3 P_crT_cr  →
    # B1/B2
    idx1, idx2 = np.nonzero(keep1)[0], np.nonzero(keep2)[0]
    common = keep1 & keep2
    ci = np.nonzero(common)[0]
    r1 = np.searchsorted(idx1, ci)
    r2 = np.searchsorted(idx2, ci)
    term_eff = []
    for j, nm in enumerate(term_names):
        c = mapping.get(nm)
        if c is None:
            continue
        jj = res_sf["classes"].index(c)
        rho = spearmanr(B1[r1, j], B2[r2, jj]).statistic
        term_eff.append({"terminal": nm, "class": c, "spearman": float(rho)})
    out["terminal_effect"] = {"per_pair": term_eff,
                              "min_spearman": min((x["spearman"] for x in term_eff),
                                                  default=float("nan"))}
    return out


def evaluate_arm(F_cr, term_names, term_idx, conf, classes, F_sf):
    """Hungarian  + Spearman/Pearson + argmax ARI/Jaccard + """
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import adjusted_rand_score

    pairs, unmatched_term, unmatched_cls = hungarian_map(conf, term_idx, term_names, classes)
    per_class = []
    for p in pairs:
        v_cr = F_cr[:, p["col_term"]]
        v_sf = F_sf[:, p["col_class"]]
        per_class.append({
            "class": p["class"], "terminal": p["terminal"], "overlap": p["overlap"],
            "spearman": float(spearmanr(v_cr, v_sf).statistic),
            "pearson": float(pearsonr(v_cr, v_sf).statistic),
        })
    min_spearman = min(p["spearman"] for p in per_class) if per_class else float("nan")

    map_term2cls = {p["col_term"]: p["class"] for p in pairs}
    used_t = sorted(map_term2cls)
    cr_arg = np.array([map_term2cls[used_t[j]] for j in F_cr[:, used_t].argmax(axis=1)])
    sf_arg = np.array([classes[j] for j in F_sf.argmax(axis=1)])
    ari = float(adjusted_rand_score(cr_arg, sf_arg))
    jac = {}
    for c in classes:
        a, b = set(np.nonzero(cr_arg == c)[0]), set(np.nonzero(sf_arg == c)[0])
        jac[c] = len(a & b) / len(a | b) if (a or b) else 1.0
    conf_mask = F_sf.max(axis=1) >= 0.8
    hc_agree = float((cr_arg[conf_mask] == sf_arg[conf_mask]).mean()) if conf_mask.any() else float("nan")
    return {"pairs": pairs, "unmatched_terminal": unmatched_term,
            "unmatched_classes": unmatched_cls, "per_class": per_class,
            "min_spearman": min_spearman, "ari": ari, "jaccard": jac,
            "highconf_agree": hc_agree, "n_highconf": int(conf_mask.sum()),
            "n_terminals": len(term_names)}


def main() -> None:
    import anndata as ad
    import cellrank as cr
    import pygpcca
    import scanpy as sc
    import scipy

    t_start = time.perf_counter()
    tracemalloc.start()

    adata = ad.read_h5ad(FIXTURE)
    X = np.asarray(adata.obsm["X_scVI"], dtype=np.float32)
    states = np.asarray(adata.obs["microglia_state"].astype(str))
    classes = sorted(np.unique(states).tolist())
    n = X.shape[0]

    # ---  ---
    res_cr = run_cellrank(adata)
    labels = select_terminals(X, states)
    res_sf = run_sfate(X, labels)
    g = res_cr["estimator"]
    macro_names = res_cr["macro_names"]

    # macrostate memberships × microglia_state
    memb = res_cr["memberships"]
    conf = np.zeros((len(classes), memb.shape[1]))
    for i, c in enumerate(classes):
        conf[i] = memb[states == c].sum(axis=0)

    # ---  kernel  fit GPCCA ---
    from cellrank import estimators as _est
    g2 = _est.GPCCA(g.kernel)
    n_states_det = len(macro_names)
    g2.fit(cluster_key="microglia_state", n_states=n_states_det)
    g2.predict_terminal_states(method="stability", stability_threshold=0.96)
    determinism = {
        "macro_same": list(map(str, g2.macrostates.cat.categories)) == macro_names,
        "terminal_same": (list(map(str, g2.terminal_states.cat.categories))
                          == res_cr["terminal_primary"]),
    }
    del g2

    # --- stability≥0.96  ---
    arms = {}
    term_primary = res_cr["terminal_primary"]
    if len(term_primary) >= 2:
        F_cr, term_names, t_fate = cr_fate_for_terminals(g, term_primary)
        res_cr["times"]["fate_primary"] = t_fate
        term_idx = [macro_names.index(t) for t in term_primary]
        arms["primary"] = evaluate_arm(F_cr, term_names, term_idx, conf, classes,
                                       res_sf["fate"])
        arms["primary"]["terminal_names"] = term_names
    else:
        arms["primary"] = {"n_terminals": len(term_primary),
                           "terminal_names": term_primary,
                           "note": " <2Spearman "}

    # ---  6  ---
    F_cr2, term_names2, t_fate2 = cr_fate_for_terminals(g, macro_names)
    res_cr["times"]["fate_supplementary"] = t_fate2
    term_idx2 = list(range(len(macro_names)))
    arms["supplementary"] = evaluate_arm(F_cr2, term_names2, term_idx2, conf,
                                         classes, res_sf["fate"])
    arms["supplementary"]["terminal_names"] = term_names2

    # --- E2  ×  2×2 ---
    sup_map = {p["terminal"]: p["class"] for p in arms["supplementary"].get("pairs", [])}
    ablation = run_ablation(adata, res_cr, res_sf, labels, classes, sup_map)

    _, peak_tracemalloc = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    wall = time.perf_counter() - t_start
    swap = subprocess.run(["swapon", "--show"], capture_output=True, text=True).stdout.strip()

    # --- E4  Spearman ---
    pa = arms["primary"]
    if pa.get("per_class"):
        ms = pa["min_spearman"]
        verdict = ("PASS" if ms > 0.95 else
                   "0.90–0.95" if ms >= 0.90 else
                   "<0.90")
    else:
        verdict = (f"CR  = {pa['n_terminals']}{pa['terminal_names']}"
                   f"Spearman —— B GPCCA ")

    results = {
        "env": {"cellrank": cr.__version__, "cellrank_baseline": "2.3.2",
                "pygpcca": pygpcca.__version__, "scanpy": sc.__version__,
                "scipy": scipy.__version__, "numpy": np.__version__, "seed": SEED},
        "n": n, "k": K, "classes": classes,
        "cr": {"macro_names": res_cr["macro_names"],
               "terminal_primary": res_cr["terminal_primary"],
               "n_states_note": res_cr["n_states_note"], "times": res_cr["times"]},
        "sfate": {"classes": res_sf["classes"], "gmres_iters": res_sf["info_iters"],
                  "residual_max": res_sf["info_resid"],
                  "n_f64_fallback": res_sf["n_f64_fallback"], "times": res_sf["times"]},
        "determinism_refit": determinism,
        "arms": arms,
        "ablation": ablation,
        "verdict": verdict, "wall_s": wall,
        "tracemalloc_peak_mb": peak_tracemalloc / 1e6, "swap": swap,
    }
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=float))

    # ---  ---
    def _arm_section(title: str, arm: dict, gated: bool) -> list[str]:
        S = [f"### {title}"]
        S.append(f"- {arm['n_terminals']}{arm['terminal_names']}")
        if "per_class" not in arm:
            S.append(f"- {arm.get('note', '')}\n")
            return S
        S.append("")
        S.append("|  | CR  | Spearman | Pearson |  |")
        S.append("|---|---|---|---|---|")
        for p in arm["per_class"]:
            S.append(f"| {p['class']} | {p['terminal']} | {p['spearman']:.4f} | "
                     f"{p['pearson']:.4f} | {p['overlap']:.0f} |")
        S.append(f"\n-  Spearman  = **{arm['min_spearman']:.4f}**"
                 f"argmax ARI = {arm['ari']:.4f}n={arm['n_highconf']}"
                 f"= {arm['highconf_agree']:.4f}")
        S.append("-  Jaccard" +
                 "".join(f"{c} {v:.3f}" for c, v in arm["jaccard"].items()))
        if arm["unmatched_terminal"]:
            S.append(f"- ** CR  Spearman**{arm['unmatched_terminal']}")
        if arm["unmatched_classes"]:
            S.append(f"- ** Spearman**{arm['unmatched_classes']}")
        S.append(f"- {'E4 ' if gated else ''}")
        S.append("")
        return S

    L = []
    L.append("# B sfate vs CellRank b_layer_validation.py \n")
    L.append("## ")
    L.append(f"- cellrank **{cr.__version__}** 2.3.2****GPCCA "
             f" fate  2.1.0  SLEPc/ pygpcca {pygpcca.__version__}"
             f"/ scanpy {sc.__version__} / scipy {scipy.__version__} / numpy {np.__version__}")
    L.append(f"- fixture_5kn={n}k={K} {SEED} BLAS=1")
    L.append(f"- swap`{' ; '.join(l for l in swap.splitlines() if l.strip())}`")
    L.append(f"-  {wall:.1f}stracemalloc  {peak_tracemalloc/1e6:.0f}MB"
             f" output/b_layer_time_v.txt\n")
    L.append("## ")
    L.append("|  |  | (s) |")
    L.append("|---|---|---|")
    tc, ts = res_cr["times"], res_sf["times"]
    L.append(f"| CellRank | neighbors→ConnectivityKernel→GPCCA({res_cr['n_states_note']})"
             f"→→fate(gmres+ilu) | neighbors {tc['neighbors']:.1f} / kernel "
             f"{tc['kernel']:.1f} / gpcca_fit {tc['gpcca_fit']:.1f} / terminal {tc['terminal']:.1f}"
             + " / fate " + "+".join(f"{v:.1f}" for k, v in tc.items() if k.startswith('fate')) + " |")
    L.append(f"| sfate | kNN →absorption_probabilities | graph {ts['graph']:.1f} / solve {ts['solve']:.1f} |")
    L.append("")
    L.append("## B ②")
    L.append(f"- CellRank {len(res_cr['macro_names'])}{res_cr['macro_names']}")
    L.append(f"- CellRank stability≥0.96{res_cr['terminal_primary']}")
    L.append(f"- sfate  6 {res_sf['classes']}")
    L.append(f"- sfate GMRES /{res_sf['info_iters']} max"
             f"{max(res_sf['info_resid']):.2e}f64 {res_sf['n_f64_fallback']}")
    L.append(f"- GPCCA  kernel  fit={determinism['macro_same']}"
             f"={determinism['terminal_same']}\n")
    L.append("## ")
    L.extend(_arm_section("stability≥0.96 ", arms["primary"], gated=True))
    L.extend(_arm_section("", arms["supplementary"], gated=False))
    L.append("### 2×2 ×  sfate ")
    e1x = ablation["e1_cross"]
    L.append(f"- ** 1E1 **sfate (P_cr, T_cr top-30 ) vs CellRank fate"
             f"L∞ = {e1x['Linf']:.2e}—— A  CR 2.1.0  "
             f"top-30 CR  A "
             f" {e1x['n_dropped_orphan']}  max {e1x['residual_max']:.2e}")
    ke = ablation["kernel_effect"]
    L.append(f"- ** 2=**P_cr vs P_sf  Spearman "
             f" = **{ke['min_spearman']:.4f}**"
             + "".join(f"{x['class']} {x['spearman']:.3f}" for x in ke['per_class']) + "")
    te = ablation["terminal_effect"]
    L.append(f"- ** 3=P_cr**T_cr vs T_sf  Spearman "
             f" = **{te['min_spearman']:.4f}**"
             + "".join(f"{x['terminal']}→{x['class']} {x['spearman']:.3f}"
                         for x in te['per_pair']) + "")
    L.append("-  2  ⇒  3  ⇒ \n")
    L.append(f"## E4 >0.95  / 0.90–0.95  / <0.90 \n\n**{verdict}**\n")
    L.append("## E2 ")
    L.append("1. **A ** P  sfate vs spsolve(f64) L∞=3.2e-7 PASS"
             "output/benchmark_m2.md—— bugB ")
    L.append("2. ****sfate  scVI latent kNNgaussian max "
             "f32CellRank  scanpy connectivities + density_normalize")
    L.append(f"3. **** 6  vs GPCCA {res_cr['n_states_note']}"
             f"stability≥0.96sfate 6  vs CR  {len(res_cr['terminal_primary'])} "
             f"{res_cr['terminal_primary']}")
    L.append("4. **** gmres tol=1e-6CR  ilu use_petsc=False"
             "A ")
    L.append("")
    L.append("### ")
    L.append(f"1. ****CR  + CR sfate  CellRank fate "
             f"L∞ = {e1x['Linf']:.2e} 1—— A  a3 E1  CR ")
    L.append(f"2. ****CR  vs sfate  fate Spearman "
             f"min = {ke['min_spearman']:.3f} 2—— X_scVI  kNN"
             f"/")
    L.append(f"3. ****GPCCA  vs  "
             f"fate Spearman min = {te['min_spearman']:.3f} 3——"
             f"min Spearman = {arms['supplementary'].get('min_spearman', float('nan')):.4f}"
             f"")
    L.append("4. **GPCCA  5k ** stability≥0.96  1 "
             "IFN_responsive_1 75k postmortemDAM/PU1low  transition/unsupported"
             "—— postmortem §6.3 "
             " C  75k ")
    L.append("5. ****E4  <2"
             "")
    REPORT.write_text("\n".join(L) + "\n")
    sa = arms["supplementary"]
    print(f"verdict: {verdict}")
    if sa.get("min_spearman") is not None:
        print(f" min Spearman = {sa['min_spearman']:.4f} | ARI = {sa['ari']:.4f}")
    print(f"written: {REPORT} / {RESULTS} {wall:.1f}s")


if __name__ == "__main__":
    main()
