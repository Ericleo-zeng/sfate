#!/usr/bin/env python
"""Layer B2 - Test A: matched-kernel genotype-delta sign consistency (75k).

Compares per-arm genotype-delta DIRECTION between CellRank fate (same P,
annotation-matched terminals; audit_b2_step3b_final_v2.npz) and sfate production
fate, at the sample-aggregation level of Results 5.

Two grids:
  15-cell: 3 arms (IFN, C1q, Homeostatic - the step12-comparable set)
           x 2 contrasts x 3 ages - NA(cKO absent)  [comparable to Results 5]
  27-cell: all 6 arms x 2 contrasts x 3 ages - NA   [new information]
Signs: mean-per-sample fate, delta = sign(mean(g1) - mean(g0)), same protocol
as c_layer_validation.s4 / exp_signflip_sign_agreement.py.
Inference: sign-permutation test (10,000 flips) for the overall count;
exact enumeration per arm. Cross-check: sfate-side signs vs
output/c_layer_results.json (archived analysis) must agree cell-for-cell.

Run from ~/research_storage/工作文件/Cellrank重构
Output: output/audit_b2_testA_log.md
"""
from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np

ANNOT_H5AD = Path("/mnt/t9/datasets/ad_microglia_fate_landscape_output/06_annotation/adata_annotated.h5ad")
FATE75K = Path("output/fate_full_75k.h5ad")
STEP3B_NPZ = Path("output/audit_b2_step3b_final_v2.npz")
CLAYER_RESULTS = Path("output/c_layer_results.json")
OUT_LOG = Path("output/audit_b2_testA_log.md")
M3_ORDER = ["C1q_inflammatory", "DAM_like", "Homeostatic",
            "IFN_responsive", "Proliferating", "PU1low_lymphoid"]
CONTRASTS = [("5xFAD", "control", "5xFAD-control"),
             ("5xFAD;Cd28-cKO", "5xFAD", "cKO-5xFAD")]
ARMS_15 = ["IFN_responsive", "C1q_inflammatory", "Homeostatic"]
N_PERM = 10_000
SEED = 20260909


def read_obs_column(f: h5py.File, key: str) -> np.ndarray:
    g = f["obs"][key]
    if isinstance(g, h5py.Group):  # categorical
        cats = [c.decode() if isinstance(c, bytes) else str(c)
                for c in g["categories"][:]]
        codes = np.asarray(g["codes"][:])
        out = np.empty(len(codes), dtype=object)
        m = codes >= 0
        out[m] = np.array(cats, dtype=object)[codes[m]]
        out[~m] = None
        return out
    arr = np.asarray(g[...])
    return np.array([x.decode() if isinstance(x, bytes) else str(x)
                     for x in arr], dtype=object)


def delta_sign(F: np.ndarray, df: dict, arm_col: int, g1: str, g0: str,
               age: str):
    """sign(mean-per-sample delta) for one contrast x age; None if missing."""
    s1 = df["samples_by"].get((g1, age), [])
    s0 = df["samples_by"].get((g0, age), [])
    if not s1 or not s0:
        return None
    v1 = [F[df["cell_rows"][s], arm_col].mean() for s in s1]
    v0 = [F[df["cell_rows"][s], arm_col].mean() for s in s0]
    return int(np.sign(np.mean(v1) - np.mean(v0)))


def build_sample_index(labels_sample, labels_genotype, labels_age):
    rows, by = {}, {}
    for i, (s, g, a) in enumerate(zip(labels_sample, labels_genotype, labels_age)):
        rows.setdefault(s, []).append(i)
    for (g, a), _ in []:
        pass
    samples_by = {}
    for s in rows:
        gi = labels_genotype[rows[s][0]]
        ai = str(labels_age[rows[s][0]])
        samples_by.setdefault((gi, ai), []).append(s)
    uniq_samples = sorted(rows)
    for s in uniq_samples:  # sanity: one genotype/age per sample
        gs = {labels_genotype[i] for i in rows[s]}
        as_ = {str(labels_age[i]) for i in rows[s]}
        assert len(gs) == 1 and len(as_) == 1, f"sample {s} has mixed meta"
    return {"cell_rows": {s: np.array(v, dtype=np.int64) for s, v in rows.items()},
            "samples_by": samples_by,
            "uniq_samples": uniq_samples}


def sign_test(count: int, total: int, n_perm: int = N_PERM, seed: int = SEED):
    """Flip one pipeline's sign vector; recompute the AGREEMENT COUNT per flip.

    Null: agreements ~ Binomial(total, 0.5). p = P(agreements >= observed).
    """
    rng = np.random.default_rng(seed)
    if total == 0:
        return None
    s = np.array([1] * count + [-1] * (total - count), dtype=np.int8)
    ge = 0
    for _ in range(n_perm):
        flips = rng.choice([-1, 1], size=total)
        agreements = int(np.sum(flips == s))
        if agreements >= count:
            ge += 1
    return {"agree": count, "total": total, "p_perm": (ge + 1) / (n_perm + 1)}

def main() -> None:
    z = np.load(STEP3B_NPZ, allow_pickle=True)
    states = [str(s) for s in z["states"]]
    assert states == M3_ORDER, f"states order {states}"
    F_cr = np.asarray(z["F_cellrank"], dtype=np.float64)   # (74984, 6) CellRank
    F_sf = np.asarray(z["F_sfate"], dtype=np.float64)      # (74984, 6) sfate

    idx_prod = np.asarray(ad.read_h5ad(FATE75K).obs_names).astype(str)
    assert idx_prod.shape == (74_984,)
    with h5py.File(ANNOT_H5AD, "r") as f:
        gobs = f["obs"]
        idx_ann = np.asarray(gobs[gobs.attrs["_index"]][...]).astype(str)
        s_sample = read_obs_column(f, "sample")
        s_genotype = read_obs_column(f, "genotype")
        s_age = read_obs_column(f, "age_months")
    lut = {b: i for i, b in enumerate(idx_ann)}
    missing = [b for b in idx_prod if b not in lut]
    assert not missing, f"{len(missing)} barcodes unmatched"
    sel = np.array([lut[b] for b in idx_prod], dtype=np.int64)

    meta = build_sample_index(s_sample[sel], s_genotype[sel],
                              np.array([str(a) for a in s_age[sel]]))
    n_samples = len(meta["uniq_samples"])
    assert n_samples == 12, f"expected 12 samples, got {n_samples}"

    def run_grid(arms):
        ages = sorted({a for (g, a) in meta["samples_by"]})
        rows = []
        for arm in arms:
            j = M3_ORDER.index(arm)
            for (g1, g0, tag) in CONTRASTS:
                for age in ages:
                    s_cr = delta_sign(F_cr, meta, j, g1, g0, age)
                    s_sf = delta_sign(F_sf, meta, j, g1, g0, age)
                    rows.append({"arm": arm, "contrast": tag, "age": age,
                                 "sign_cellrank": s_cr, "sign_sfate": s_sf,
                                 "scored": s_cr is not None and s_sf is not None})
        scored = [r for r in rows if r["scored"]]
        agree = sum(1 for r in scored if r["sign_cellrank"] == r["sign_sfate"])
        return {"ages": ages, "rows": rows,
                "agree": agree, "total": len(scored),
                "test": sign_test(agree, len(scored))}

    grid15 = run_grid(ARMS_15)
    grid27 = run_grid(M3_ORDER)

    # per-arm exact enumeration for the 15-cell grid (mirrors Results 5)
    per_arm = {}
    for arm in ARMS_15:
        cells = [r for r in grid15["rows"] if r["arm"] == arm and r["scored"]]
        a_ = sum(1 for r in cells if r["sign_cellrank"] == r["sign_sfate"])
        n_ = len(cells)
        # exact: P(X >= a_) under Binomial(n_, 0.5)
        from math import comb
        p_ = sum(comb(n_, k) for k in range(a_, n_ + 1)) / 2 ** n_
        per_arm[arm] = {"agree": a_, "total": n_, "p_exact_ge": p_}

    # cross-check sfate signs vs archived c_layer_results.json
    xref = "not_available"
    if CLAYER_RESULTS.exists():
        arch = json.loads(CLAYER_RESULTS.read_text())["sample_level"]["rows"]
        arch_map = {(r["class"], r["contrast"]): r["sfate_signs"] for r in arch}
        mism = 0
        for r in grid15["rows"]:
            key = (r["arm"], r["contrast"].replace("-", "\u2212"))
            if key in arch_map and r["sign_sfate"] is not None:
                ages = grid15["ages"]
                ai = ages.index(r["age"])
                if arch_map[key][ai] is not None and \
                   int(arch_map[key][ai]) != r["sign_sfate"]:
                    mism += 1
        xref = {"sfate_sign_mismatches_vs_c_layer": mism}

    OUT_LOG.write_text(json.dumps({
        "grid15_step12_comparable": grid15,
        "grid27_all_six_arms": grid27,
        "per_arm_15_exact": per_arm,
        "archived_step12_reference": {"agree_9_of_15": "Results 5"},
        "cross_check": xref,
    }, ensure_ascii=False, indent=2))
    print(f"DONE 15-cell: {grid15['agree']}/{grid15['total']} "
          f"(step12 archived: 9/15) | 27-cell: {grid27['agree']}/{grid27['total']} "
          f"| xref={xref}", flush=True)


if __name__ == "__main__":
    main()
