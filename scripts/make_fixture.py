#!/usr/bin/env python
""" A/B  output/fixture_5k.h5ad

- <mnt>/.../06_annotation/adata_annotated.h5adbacked
-  microglia_state  5000 SEED Proliferating
- obsm['X_scVI']30  float32+ obs /layers/obsp
-

source ~/sfate_env/bin/activate && python scripts/make_fixture.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from sfate.io import load_latent, load_obs_columns

SEED = 20260909
N_TARGET = 5000
SRC = (
    "<data>/"
    "06_annotation/adata_annotated.h5ad"
)
OUT = Path("output/fixture_5k.h5ad")
OBS_COLS = [
    "microglia_state",
    "pstate_subtype",
    "leiden",
    "genotype",
    "sample",
    "age_months",
]
STRATA = "microglia_state"


def stratified_sample(state: pd.Series, n_target: int, seed: int) -> np.ndarray:
    """"""
    rng = np.random.default_rng(seed)
    counts = state.value_counts()
    exact = n_target * counts / counts.sum()
    base = exact.astype(int)
    #  1
    for lvl in counts.index:
        if counts[lvl] >= 1 and base[lvl] == 0:
            base[lvl] = 1
    deficit = n_target - base.sum()
    if deficit > 0:
        remainders = (exact - base).sort_values(ascending=False)
        for lvl in remainders.index:
            if deficit == 0:
                break
            if base[lvl] < counts[lvl]:
                base[lvl] += 1
                deficit -= 1
    assert deficit == 0, f"cannot reach n={n_target}, short by {deficit}"
    idx_all = np.arange(len(state))
    chosen: list[np.ndarray] = []
    for lvl, n_take in base.items():
        lvl_idx = idx_all[state.to_numpy() == lvl]
        assert len(lvl_idx) >= n_take, f"stratum {lvl!r} too small"
        chosen.append(rng.choice(lvl_idx, size=n_take, replace=False))
    sel = np.sort(np.concatenate(chosen))
    return sel


def main() -> int:
    t0 = time.perf_counter()
    a = ad.read_h5ad(SRC, backed="r")
    print(f"[1/5] h5py direct-open ok ({time.perf_counter() - t0:.1f}s)")

    state = load_obs_columns(SRC, [STRATA])[STRATA]
    sel = stratified_sample(state, N_TARGET, SEED)
    print(f"[2/5] stratified sample n={len(sel)}, seed={SEED}")

    latent = load_latent(SRC)[sel]  # (5000, 30) float32
    assert latent.shape == (N_TARGET, 30) and latent.dtype == np.float32
    obs_sub = load_obs_columns(SRC, OBS_COLS).iloc[sel].copy()

    # h5py
    probe = np.random.default_rng(SEED + 1).choice(sel, size=200, replace=False)
    src_probe = load_obs_columns(SRC, OBS_COLS).iloc[probe]
    assert src_probe[STRATA].to_numpy().tolist() == obs_sub[STRATA].iloc[
        [int(np.where(sel == p)[0][0]) for p in probe]
    ].to_numpy().tolist(), "obs row mismatch vs source"
    print("[3/5] obs/latent subset consistency assertions passed")

    # X  csr  h5ad
    import scipy.sparse as sp

    X = sp.csr_matrix((N_TARGET, 1), dtype=np.float32)

    out = ad.AnnData(X=X, obs=obs_sub)
    out.obsm["X_scVI"] = latent
    out.uns["fixture"] = {
        "seed": SEED,
        "n_target": N_TARGET,
        "source": SRC,
        "strata": STRATA,
        "created_by": "scripts/make_fixture.py",
    }
    OUT.parent.mkdir(exist_ok=True)
    out.write_h5ad(OUT, compression="gzip")
    print(f"[4/5] written {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")

    # stratified_sample
    sel_check = stratified_sample(
        load_obs_columns(SRC, [STRATA])[STRATA], N_TARGET, SEED
    )
    assert np.array_equal(sel, sel_check), "sampling not reproducible with same seed"
    print("[5/5] reproducibility check passed (same seed -> identical selection)")

    print("\nfixture summary:")
    print(f"  n={N_TARGET}, latent={latent.shape} {latent.dtype}")
    print(f"  microglia_state counts: {obs_sub[STRATA].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
