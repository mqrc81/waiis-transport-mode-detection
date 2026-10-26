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
