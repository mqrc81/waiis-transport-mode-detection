"""60 s context features: stats of window features over the previous 60 s (experiment 02, step 3; moved here from src/tmd).

Causal: window i's context is every window of the same sequence with start time in (t_i - 60 s, t_i], itself
included. Sequences: SHL = one recording x position; ExtraSensory = one user, minutes chained by time (each minute is
a ~16-23 s snippet, so context reaches back into the previous minute only when that minute was recorded).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tmd.features import CHANNELS, NAMES
from tmd.windows import STRIDE_S

HORIZON_S = 60.0
CTX_BASE = ["mean", "std", "p50", "p95", "mcr", "acf_peak", "hjorth_mob", "dom_freq", "spec_entropy",
            "bandlog_0_1", "bandlog_1_3", "bandlog_3_6", "bandlog_6_12.5"]
CTX_SOURCE = [f"{c}__{f}" for c in CHANNELS for f in CTX_BASE]
CTX_NAMES = [f"ctx_mean__{n}" for n in CTX_SOURCE] + [f"ctx_std__{n}" for n in CTX_SOURCE] + ["ctx_n", "ctx_span_s"]
assert all(n in NAMES for n in CTX_SOURCE)


def window_time_s(df: pd.DataFrame) -> np.ndarray:
    """Start time [s] of each window: SHL from t_start_ms; ExtraSensory = minute timestamp + window index x 2.5 s
    (the snippet's offset inside the minute is not known)."""
    if "t_start_ms" in df and df.t_start_ms.notna().all():
        return df.t_start_ms.to_numpy(np.float64) / 1000.0
    return df.ts.to_numpy(np.float64) + df.win.to_numpy(np.float64) * STRIDE_S


def sequence_id(df: pd.DataFrame) -> np.ndarray:
    """SHL: recording x position; ExtraSensory: user."""
    if df.dataset.iat[0] == "shl":
        return (df.rec + "|" + df.position).to_numpy()
    return df.user.to_numpy()


def context_features(df: pd.DataFrame, horizon_s: float = HORIZON_S) -> pd.DataFrame:
    """CTX_NAMES for every row of df (same index), computed within sequences over (t - horizon, t]."""
    t = window_time_s(df)
    seq = sequence_id(df)
    x = df[CTX_SOURCE].to_numpy(np.float64)
    out = np.zeros((len(df), len(CTX_NAMES)), np.float32)
    nf = len(CTX_SOURCE)
    for s in pd.unique(seq):
        idx = np.flatnonzero(seq == s)
        idx = idx[np.argsort(t[idx], kind="stable")]
        ts, xs = t[idx], x[idx]
        c1 = np.vstack([np.zeros(nf), np.cumsum(xs, 0)])
        c2 = np.vstack([np.zeros(nf), np.cumsum(xs * xs, 0)])
        hi = np.arange(1, len(idx) + 1)
        lo = np.searchsorted(ts, ts - horizon_s, side="right")
        n = (hi - lo)[:, None]
        mean = (c1[hi] - c1[lo]) / n
        var = np.maximum((c2[hi] - c2[lo]) / n - mean ** 2, 0.0)
        out[idx, :nf] = mean
        out[idx, nf:2 * nf] = np.sqrt(var)
        out[idx, 2 * nf] = n[:, 0]
        out[idx, 2 * nf + 1] = ts - ts[lo]
    return pd.DataFrame(out, columns=CTX_NAMES, index=df.index)


class StreamingContext:
    """Per-window streaming version (for timing): keeps the last 60 s of CTX_SOURCE vectors."""

    def __init__(self, horizon_windows: int = int(HORIZON_S / STRIDE_S)):
        self.buf = np.zeros((horizon_windows, len(CTX_SOURCE)))
        self.n = 0
        self.src = np.array([NAMES.index(c) for c in CTX_SOURCE])

    def __call__(self, f: np.ndarray) -> np.ndarray:
        self.buf = np.roll(self.buf, -1, 0)
        self.buf[-1] = f[0, self.src]
        self.n = min(self.n + 1, len(self.buf))
        b = self.buf[-self.n:]
        ctx = np.concatenate([b.mean(0), b.std(0), [self.n, (self.n - 1) * STRIDE_S]])
        return np.concatenate([f, ctx[None].astype(np.float32)], 1)
