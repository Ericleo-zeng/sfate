"""io.pyh5py  latent  obs  anndata backed  layers

M1  data_recon.mdanndata 0.12.16  read_h5ad(backed='r')
 06_annotation/adata_annotated.h5ad  layers6.9GB
backed  latent/obs  h5py
O()

 X / layers / obsp
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def load_latent(path: str | Path, key: str = "X_scVI") -> np.ndarray:
    """ obsm[key] float32 latent """
    with h5py.File(path, "r") as f:
        ds = f[f"obsm/{key}"]
        return np.asarray(ds[...], dtype=np.float32)


def load_obs_categorical(path: str | Path, col: str) -> pd.Categorical:
    """ obs HDF5 categorical: codes + categories"""
    with h5py.File(path, "r") as f:
        g = f[f"obs/{col}"]
        if isinstance(g, h5py.Dataset):
            #  categorical /
            return pd.Categorical(np.asarray(g[...]).astype(str))
        codes = np.asarray(g["codes"][...])
        cats = np.asarray(g["categories"][...]).astype(str)
        return pd.Categorical.from_codes(codes, categories=cats)


def load_obs_columns(path: str | Path, cols: list[str]) -> pd.DataFrame:
    """ obs  DataFramecategorical  category dtype/"""
    out: dict[str, object] = {}
    with h5py.File(path, "r") as f:
        for col in cols:
            g = f[f"obs/{col}"]
            if isinstance(g, h5py.Dataset):
                arr = np.asarray(g[...])
                out[col] = arr if arr.dtype.kind in "biuf" else arr.astype(str)
            else:
                codes = np.asarray(g["codes"][...])
                cats = np.asarray(g["categories"][...]).astype(str)
                out[col] = pd.Categorical.from_codes(codes, categories=cats)
    return pd.DataFrame(out)
