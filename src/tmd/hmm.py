"""HMM smoothing of per-window class posteriors (experiment 02, step 3).

States = the 8 classes. Emissions = the classifier's posteriors (trained with balanced class weights, so they act as
scaled likelihoods under a uniform prior). Transitions per 2.5 s step are estimated from training sequences only;
a gap of k steps uses A^k, and a gap over `max_gap_s` restarts the chain.
"""
from __future__ import annotations

import numpy as np

from tmd.windows import STRIDE_S

N = 8


def transition_matrix(y: np.ndarray, t: np.ndarray, seq: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Counts of class i -> j between consecutive labelled windows (one 2.5 s step apart, same sequence),
    plus `alpha` pseudo-counts, row-normalised. y in 1..8 (0 = unlabelled, skipped)."""
    c = np.full((N, N), alpha)
    order = np.lexsort((t, seq))
    y, t, seq = y[order], t[order], seq[order]
    step = (seq[1:] == seq[:-1]) & (np.abs(np.diff(t) - STRIDE_S) < 0.6) & (y[1:] > 0) & (y[:-1] > 0)
    np.add.at(c, (y[:-1][step] - 1, y[1:][step] - 1), 1)
    return c / c.sum(1, keepdims=True)


def _powers(a: np.ndarray, kmax: int) -> list[np.ndarray]:
    p = [np.eye(N), a]
    for _ in range(2, kmax + 1):
        p.append(p[-1] @ a)
    return p


def _steps(dt: np.ndarray, max_gap_s: float) -> np.ndarray:
    k = np.maximum(1, np.rint(dt / STRIDE_S)).astype(int)
    k[dt > max_gap_s] = 0                      # 0 = restart
    return k


def forward(p: np.ndarray, t: np.ndarray, seq: np.ndarray, a: np.ndarray, max_gap_s: float = 60.0) -> np.ndarray:
    """Causal filtered posteriors P(state_t | windows up to t), same row order as p."""
    out = np.empty_like(p, dtype=np.float64)
    kmax = int(max_gap_s / STRIDE_S) + 1
    ap = _powers(a, kmax)
    for s in np.unique(seq):
        idx = np.flatnonzero(seq == s)
        idx = idx[np.argsort(t[idx], kind="stable")]
        k = np.concatenate([[0], _steps(np.diff(t[idx]), max_gap_s)])
        e = np.maximum(p[idx].astype(np.float64), 1e-9)
        cur = None
        for i in range(len(idx)):
            v = e[i] if k[i] == 0 else (cur @ ap[k[i]]) * e[i]
            cur = v / v.sum()
            out[idx[i]] = cur
    return out


def viterbi(p: np.ndarray, t: np.ndarray, seq: np.ndarray, a: np.ndarray, max_gap_s: float = 60.0) -> np.ndarray:
    """Most likely state path per sequence (offline), returned as one-hot rows in p's order."""
    out = np.zeros_like(p, dtype=np.float32)
    kmax = int(max_gap_s / STRIDE_S) + 1
    lap = [np.log(np.maximum(m, 1e-300)) for m in _powers(a, kmax)]
    for s in np.unique(seq):
        idx = np.flatnonzero(seq == s)
        idx = idx[np.argsort(t[idx], kind="stable")]
        k = np.concatenate([[0], _steps(np.diff(t[idx]), max_gap_s)])
        le = np.log(np.maximum(p[idx].astype(np.float64), 1e-9))
        bounds = list(np.flatnonzero(k == 0)) + [len(idx)]
        for b0, b1 in zip(bounds[:-1], bounds[1:]):          # one segment between restarts
            back = np.zeros((b1 - b0, N), np.int64)
            d = le[b0]
            for i in range(b0 + 1, b1):
                m = d[:, None] + lap[k[i]]
                back[i - b0] = m.argmax(0)
                d = m.max(0) + le[i]
            st = int(d.argmax())
            for i in range(b1 - 1, b0 - 1, -1):
                out[idx[i], st] = 1.0
                st = int(back[i - b0, st])
    return out


def fixed_lag(p: np.ndarray, t: np.ndarray, seq: np.ndarray, a: np.ndarray, lag_s: float = 60.0,
              max_gap_s: float = 60.0) -> np.ndarray:
    """Fixed-lag smoothing: P(state_t | windows up to t + lag_s), same row order as p.

    Forward-filtered posterior times a backward message over the windows of the same sequence that start within
    lag_s after t (stopping at a restart). The decision for a window is available lag_s after it.
    """
    alpha = forward(p, t, seq, a, max_gap_s)
    out = np.empty_like(alpha)
    kmax = int(max_gap_s / STRIDE_S) + 1
    ap = np.stack(_powers(a, kmax))
    for s in np.unique(seq):
        idx = np.flatnonzero(seq == s)
        idx = idx[np.argsort(t[idx], kind="stable")]
        n = len(idx)
        ts = t[idx]
        k = np.concatenate([[0], _steps(np.diff(ts), max_gap_s)])      # k[i]: step from i-1 to i (0 = restart)
        e = np.maximum(p[idx].astype(np.float64), 1e-9)
        hi = np.searchsorted(ts, ts + lag_s, side="right") - 1          # last window within the lag
        r = np.flatnonzero(k == 0)                                       # restarts cut the look-ahead
        nxt = np.append(r, n)[np.searchsorted(r, np.arange(n), side="right")]  # first restart after i
        hi = np.minimum(hi, nxt - 1)
        L = int((hi - np.arange(n)).max()) if n else 0
        m = np.ones((n, N))
        base = np.arange(n)
        for j in range(L, 0, -1):                                        # backward from t + j down to t + 1
            pos = base + j
            ok = pos <= hi
            pj = np.minimum(pos, n - 1)
            v = e[pj] * m
            nv = np.einsum("nij,nj->ni", ap[k[pj]], v)
            nv /= nv.sum(1, keepdims=True)
            m = np.where(ok[:, None], nv, 1.0)
        post = alpha[idx] * m
        out[idx] = post / post.sum(1, keepdims=True)
    return out
