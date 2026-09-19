#!/usr/bin/env python
"""
output/fate_probabilities/fate_by_sample_sfate_vs_cellrank.tsv

record
- sample_mean sample ×  sfate / CellRank(step12)  delta
- genotype_delta_sign  ×  ×   delta
sfate  columns  fate_full_75k.h5adC1
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

OUT = Path("output/fate_probabilities/fate_by_sample_sfate_vs_cellrank.tsv")
FATE_H5AD = Path("output/fate_full_75k.h5ad")
ANNOTATED = ("<data>/"
             "06_annotation/adata_annotated.h5ad")
CR_WIDE = ("<data>/"
           "11_cellrank/fate_prob_by_sample_WIDE.tsv")
CLASSES = ["IFN_responsive", "C1q_inflammatory", "Homeostatic"]  # CR  3
CONTRASTS = [("5xFAD", "control", "5xFAD-control"), ("5xFAD;Cd28-cKO", "5xFAD", "cKO-5xFAD")]


def main() -> None:
    import anndata as ad

    from sfate.io import load_obs_columns

    a = ad.read_h5ad(FATE_H5AD)  #  obsm fate
    fate = pd.DataFrame(np.asarray(a.obsm["sfate_fate"]),
                        columns=a.uns["sfate_meta"]["classes"], index=a.obs.index)
    obs = load_obs_columns(ANNOTATED, ["sample", "genotype", "age_months"])
    obs.index = fate.index
    df = obs.join(fate)
    sf = (df.groupby(["sample", "genotype", "age_months"], observed=True)[CLASSES]
            .mean().reset_index().rename(columns={"age_months": "age"}))
    sf["age"] = sf["age"].astype(str)

    cr = pd.read_csv(CR_WIDE, sep="\t")
    cr["age"] = cr["age"].astype(str)
    cr = cr[["sample", "genotype", "age"] + [f"fate_{c}" for c in CLASSES]]

    m = sf.merge(cr, on=["sample", "genotype", "age"], how="inner")
    assert m.shape[0] == 12, f" join  {m.shape[0]} != 12"

    rows = []
    for _, r in m.iterrows():
        for c in CLASSES:
            rows.append({
                "record": "sample_mean", "sample": r["sample"],
                "genotype": r["genotype"], "age": r["age"], "fate": c,
                "sfate_mean": f"{r[c]:.6f}", "cellrank_mean": f"{r[f'fate_{c}']:.6f}",
                "delta_sfate_minus_cellrank": f"{r[c] - r[f'fate_{c}']:+.6f}",
                "sign_agree": "",
            })
    #  delta
    for c in CLASSES:
        for g1, g0, tag in CONTRASTS:
            for age in sorted(m["age"].unique()):
                sub = m[m["age"] == age]
                v1 = sub.loc[sub["genotype"] == g1]
                v0 = sub.loc[sub["genotype"] == g0]
                if v1.empty or v0.empty:
                    continue  # 8  cKO
                d_sf = v1[c].mean() - v0[c].mean()
                d_cr = v1[f"fate_{c}"].mean() - v0[f"fate_{c}"].mean()
                rows.append({
                    "record": "genotype_delta_sign", "sample": f"({tag})",
                    "genotype": f"{g1} - {g0}", "age": age, "fate": c,
                    "sfate_mean": f"{d_sf:+.6f}", "cellrank_mean": f"{d_cr:+.6f}",
                    "delta_sfate_minus_cellrank": "",
                    "sign_agree": bool(np.sign(d_sf) == np.sign(d_cr)),
                })
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    header = ("# sfate(columns , k=30 latent kNN, 6 ) vs "
              "CellRank step12 ConnectivityKernel+GPCCAfate  3 \n"
              "# record=sample_meanrecord=genotype_delta_sign"
              " deltasfate_mean/cellrank_mean  delta \n")
    OUT.write_text(header + out.to_csv(sep="\t", index=False))
    n_agree = sum(1 for r in rows if r["record"] == "genotype_delta_sign" and r["sign_agree"] is True)
    n_sign = sum(1 for r in rows if r["record"] == "genotype_delta_sign")
    print(f"written: {OUT}sample_mean {m.shape[0]*len(CLASSES)}  + "
          f"genotype_delta_sign {n_sign}  {n_agree}/{n_sign}")


if __name__ == "__main__":
    main()
