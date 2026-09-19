"""pytest  latent  + fixture_5k """

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

FIXTURE_PATH = Path("output/fixture_5k.h5ad")


def make_branching_latent(
    n: int = 1000,
    d: int = 10,
    n_terminal: int = 3,
    frac_terminal: float = 0.25,
    branch_len: float = 5.0,
    noise: float = 0.3,
    seed: int = 0,
    return_parts: bool = False,
) -> np.ndarray:
    """ latent →  → n_terminal validation-protocol  1/2

     ∝  nnz/
    return_parts=True  (X, parts)parts = {"part_ids": (n,) int
    =0=1 j=2+j"branch_dirs": (n_terminal, d)"branch_base": (d,)}
     a1
    """
    rng = np.random.default_rng(seed)
    n_term = int(n * frac_terminal / n_terminal) * n_terminal
    n_root = int(n * 0.2)
    n_mid = n - n_root - n_term
    parts = []
    #
    parts.append(rng.normal(0, noise, size=(n_root, d)) + rng.normal(0, 1, size=(1, d)))
    #
    v = rng.normal(0, 1, size=d)
    v /= np.linalg.norm(v)
    t_mid = rng.uniform(0.3, 0.7, size=n_mid)[:, None]
    parts.append(t_mid * v * branch_len + rng.normal(0, noise, size=(n_mid, d)))
    #
    base = 0.7 * v * branch_len
    branch_dirs = []
    part_ids = np.concatenate([
        np.zeros(n_root, dtype=np.int64),
        np.ones(n_mid, dtype=np.int64),
    ])
    branch_ids = []
    for j in range(n_terminal):
        w = rng.normal(0, 1, size=d)
        w -= w @ v * v
        w /= np.linalg.norm(w) + 1e-12
        branch_dirs.append(w)
        nj = n_term // n_terminal
        branch_ids.append(np.full(nj, 2 + j, dtype=np.int64))
        t = rng.uniform(0.7, 1.0, size=nj)[:, None]
        parts.append(base + (t - 0.7) * branch_len * 3 * w[None, :]
                     + rng.normal(0, noise, size=(nj, d)))
    X = np.concatenate(parts, axis=0).astype(np.float32)
    if not return_parts:
        return X
    part_ids = np.concatenate([part_ids] + branch_ids)
    return X, {
        "part_ids": part_ids,
        "branch_dirs": np.asarray(branch_dirs),
        "branch_base": base,
    }


@pytest.fixture(scope="session")
def latent_small():
    """n=800, d=10  kNN"""
    return make_branching_latent(n=800, d=10, n_terminal=3, seed=42)


@pytest.fixture(scope="session")
def latent_2k():
    """n=2000, d=10 """
    return make_branching_latent(n=2000, d=10, n_terminal=3, seed=7)


@pytest.fixture(scope="session")
def latent_10k():
    """n=10000, d=10 mem_smoke """
    return make_branching_latent(n=10000, d=10, n_terminal=3, seed=99)


@pytest.fixture(scope="session")
def fixture5k():
    """ 5k  scripts/make_fixture.py """
    if not FIXTURE_PATH.exists():
        pytest.skip("output/fixture_5k.h5ad  scripts/make_fixture.py")
    import anndata as ad

    return ad.read_h5ad(FIXTURE_PATH)
