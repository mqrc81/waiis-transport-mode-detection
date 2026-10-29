"""Step D: context over neighbouring usable minutes of the same session (past and future).

Neighbour at offset j (j in -3..-1, +1..+3 min) = the usable minute of the same (user, session) with
|ts - (t + 60 j)| <= 30 s. Columns (160 for the 40 base features): value at -1 and +1, mean and max over the available
neighbours in +-3 min (the minute itself excluded). Missing neighbours stay NaN; nothing counts them.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

OFFSETS = (-3, -2, -1, 1, 2, 3)
TOL_S = 30


def neighbours(df: pd.DataFrame) -> np.ndarray:
    """(N, 6) row positions (in df's order) of the neighbours at OFFSETS, -1 when absent. df: user, session, ts."""
    g = pd.factorize(df.user.astype(str) + "/" + df.session.astype(str))[0].astype(np.int64)
    key = g * 10_000_000_000 + df.ts.to_numpy(np.int64)
    order = np.argsort(key, kind="stable")
    sk = key[order]
    out = np.full((len(df), len(OFFSETS)), -1, np.int64)
    for c, j in enumerate(OFFSETS):
        target = key + 60 * j
        hi = np.clip(np.searchsorted(sk, target), 0, len(sk) - 1)
        lo = np.clip(hi - 1, 0, len(sk) - 1)
        best = np.where(np.abs(sk[hi] - target) <= np.abs(sk[lo] - target), hi, lo)
        cand = order[best]
        ok = (g[cand] == g) & (np.abs(key[cand] - target) <= TOL_S) & (cand != np.arange(len(df)))
        out[:, c] = np.where(ok, cand, -1)
    return out


def columns(base: np.ndarray, nb: np.ndarray, base_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """base (N, K) features of every row; nb (N, 6) from `neighbours` (possibly masked to -1) -> (N, 4K)."""
    v = np.where((nb >= 0)[..., None], base[np.maximum(nb, 0)], np.nan)          # (N, 6, K)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                           # all-NaN rows stay NaN
        mean, mx = np.nanmean(v, 1), np.nanmax(v, 1)
    prev, nxt = v[:, OFFSETS.index(-1)], v[:, OFFSETS.index(1)]
    names = [f"ctx_{p}__{n}" for p in ("prev1", "next1", "mean3", "max3") for n in base_names]
    return np.concatenate([prev, nxt, mean, mx], 1).astype(np.float32), names


def mask_like(nb: np.ndarray, rows: np.ndarray, donors: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """D': give each row in `rows` the whole presence pattern (all 6 offsets jointly) of a random donor row
    (ExtraSensory training rows); neighbours absent in the donor's pattern become -1."""
    out = nb.copy()
    pick = donors[rng.integers(0, len(donors), len(rows))]
    present = nb[pick] >= 0
    out[rows] = np.where(present, nb[rows], -1)
    return out
