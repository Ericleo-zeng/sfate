#!/usr/bin/env python3
"""Sign-permutation test for sample-level direction agreement.

Input: output/fate_probabilities/fate_by_sample_sfate_vs_cellrank.tsv
Output: output/signflip_sign_agreement.md/.json
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TSV = ROOT / "output/fate_probabilities/fate_by_sample_sfate_vs_cellrank.tsv"
OUT_MD = ROOT / "output/signflip_sign_agreement.md"
OUT_JSON = ROOT / "output/signflip_sign_agreement.json"


def main() -> int:
    rows = []
    with TSV.open(encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] != "genotype_delta_sign":
                continue
            rows.append({
                "contrast": parts[1],
                "age": int(parts[3]),
                "fate": parts[4],
                "sfate_delta": float(parts[5].replace("+", "")),
                "cellrank_delta": float(parts[6].replace("+", "")),
                "sign_agree": parts[8].lower() == "true",
            })

    n = len(rows)
    obs_all = int(sum(r["sign_agree"] for r in rows))

    sfate_signs = np.sign([r["sfate_delta"] for r in rows])
    cr_signs = np.sign([r["cellrank_delta"] for r in rows])

    # global permutation: randomize CellRank signs over 15 cells
    rng = np.random.default_rng(20260919)
    n_perm = 10_000
    exceed = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(cr_signs))
        agree = int((sfate_signs == cr_signs[perm]).sum())
        if agree >= obs_all:
            exceed += 1
    p_all = (1 + exceed) / (n_perm + 1)

    # per-fate exact enumeration
    fates = sorted({r["fate"] for r in rows})
    per_fate = {}
    for fate in fates:
        idx = [i for i, r in enumerate(rows) if r["fate"] == fate]
        s = sfate_signs[idx]
        c = cr_signs[idx]
        obs = int((s == c).sum())
        # enumerate 2^m sign patterns for CellRank
        m = len(idx)
        counts = []
        for bits in itertools.product([-1.0, 1.0], repeat=m):
            bits_arr = np.array(bits)
            counts.append(int((s == bits_arr).sum()))
        total = len(counts)
        p_ge = sum(1 for x in counts if x >= obs) / total
        if fate == "Homeostatic":
            p_report = sum(1 for x in counts if x <= obs) / total
        else:
            p_report = p_ge
        per_fate[fate] = {"observed_agree": obs, "p": p_report, "p_ge": p_ge}

    result = {
        "n_cells": n,
        "observed_agree_all": obs_all,
        "p_all": p_all,
        "per_fate": per_fate,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# Sign-permutation test for sample-level direction agreement",
        "",
        f"- Scored cells: {n}",
        f"- Observed agreement: {obs_all}/{n}",
        f"- Global p (10,000 permutations, A ≥ {obs_all}): {p_all:.4f}",
        "",
        "## Per-fate exact enumeration",
        "",
        "| fate | observed agree | p |",
        "|---|---|---|",
    ]
    for fate, d in per_fate.items():
        note = " (reversal test A≤obs)" if fate == "Homeostatic" else ""
        lines.append(f"| {fate} | {d['observed_agree']}/5 | {d['p']:.4f}{note} |")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
