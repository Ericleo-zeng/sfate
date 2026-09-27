#!/usr/bin/env python
"""Repair spearman / S_fate columns in audit_b2_step3b_final_v2.npz -> _v3.

Background (B2 audit, 2026-09-27, provenance verified in-session):
  audit_b2_fix_label_swap.py corrected F_sfate / F_cellrank to the true
  M3 column convention (name-keyed against production c_layer_sfate_fate.npz
  classes), but did NOT touch `spearman` or `S_fate`. Those two arrays were
  computed from the h5ad whose VALUE order is np.unique class order
  [..., PU1low_lymphoid, Proliferating] while their labels follow
  states = M3_ORDER [..., Proliferating, PU1low_lymphoid].
  => indices 4 and 5 are name-swapped relative to `states`.

This script writes audit_b2_step3b_final_v3.npz in which EVERY array
follows the `states` (M3) column convention, and self-verifies:
  1. repaired spearman == independently recomputed per-arm Spearman
     between F_cellrank and F_sfate (assert < 1e-9);
  2. after Hungarian-matching S_fate rows (CR macro columns, natural
     order) to arms, diagonal == repaired spearman (assert < 1e-9) and
     the matching is a bijection.

Input : output/audit_b2_step3b_final_v2.npz
Output: output/audit_b2_step3b_final_v3.npz   (quarantine _v2 after _v3 passes)
"""

from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
CWD = Path(__file__).resolve().parent.parent.parent.parent  # Cellrank重构/

SRC = CWD / "output" / "audit_b2_step3b_final_v2.npz"
DST = CWD / "output" / "audit_b2_step3b_final_v3.npz"

SWAP = [0, 1, 2, 3, 5, 4]  # exchange indices 4 <-> 5 to restore M3 convention


def main() -> None:
    z = np.load(SRC, allow_pickle=True)
    states = [str(s) for s in z["states"]]
    assert states == ["C1q_inflammatory", "DAM_like", "Homeostatic",
                      "IFN_responsive", "Proliferating",
                      "PU1low_lymphoid"], states

    F_cr = np.asarray(z["F_cellrank"], dtype=np.float64)
    F_sf = np.asarray(z["F_sfate"], dtype=np.float64)
    S = np.asarray(z["S_fate"], dtype=np.float64)[:, SWAP]   # repair columns
    sp = np.asarray(z["spearman"], dtype=np.float64)[SWAP]   # repair columns

    # -- verification 1: per-arm Spearman recomputed from the F matrices ------
    diag = np.array([spearmanr(F_cr[:, j], F_sf[:, j]).statistic
                     for j in range(6)])
    err = float(np.abs(diag - sp).max())
    assert err < 1e-9, f"spearman repair inconsistent with F matrices: {err}"
    print(f"[ok] repaired spearman == recomputed diag (max |diff| = {err:.2e})")
    for j, name in enumerate(states):
        print(f"     {name:<18} {sp[j]:+.6f}")

    # -- verification 2: S_fate Hungarian diagonal ----------------------------
    rows, cols = linear_sum_assignment(-S)
    assert sorted(cols.tolist()) == list(range(6)), "Hungarian matching not a bijection"
    row_to_arm = dict(zip(rows.tolist(), cols.tolist()))
    for r, j in row_to_arm.items():
        assert abs(S[r, j] - sp[j]) < 1e-9, (
            f"S_fate row {r} -> arm {states[j]}: {S[r, j]:.6f} != {sp[j]:.6f}")
    print(f"[ok] S_fate Hungarian diagonal == repaired spearman")
    print("     row->arm:", {r: states[j] for r, j in sorted(row_to_arm.items())})

    np.savez(DST,
             F_cellrank=z["F_cellrank"],
             F_sfate=z["F_sfate"],
             states=z["states"],
             spearman=sp,
             memberships_natural=z["memberships_natural"],
             macro_names=z["macro_names"],
             F_cr_raw=z["F_cr_raw"],
             fate_names=z["fate_names"],
             S_fate=S)
    print(f"[written] {DST}")


if __name__ == "__main__":
    main()
