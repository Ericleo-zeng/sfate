#!/usr/bin/env python
"""Fix the Proliferating/PU1low_lymphoid column-name swap in audit artifacts.

Root cause: m3 labeled the solver output (np.unique class order) with the
ANNOTATED constant order; positions 4/5 ('Proliferating'/'PU1low_lymphoid')
have been name-swapped in every artifact derived from fate_full_75k.h5ad.
Values are untouched; only the two annotation-aligned matrices get their
columns 4<->5 swapped back.

Input : output/audit_b2_step3b_final.npz
Output: output/audit_b2_step3b_final_v2.npz  (corrected; raw macro-order
        arrays unchanged)
"""
import numpy as np

SWAP = [4, 5]

z = np.load("output/audit_b2_step3b_final.npz", allow_pickle=True)
out = {}
for k in z.files:
    out[k] = z[k]

for key in ("F_sfate", "F_cellrank"):
    M = np.asarray(z[key]).copy()
    M[:, SWAP] = M[:, SWAP[:, None]] if False else M[:, [5, 4]]
    out[key] = M

np.savez("output/audit_b2_step3b_final_v2.npz", **out)
print("DONE wrote audit_b2_step3b_final_v2.npz (F_sfate/F_cellrank cols 4<->5 swapped)")
