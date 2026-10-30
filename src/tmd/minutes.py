"""HMM over the minutes of a session (from experiment 05's pipeline.py): labelled minutes, parameters, smoothing.

One example per minute, so the HMM step is 60 s; a gap over 10 min or a new session restarts the chain. Transitions
and the initial distribution come from all labelled minutes of the training folds, before caps and drops.
Frames passed in need the columns source, user, session, ts (and fold for smooth_cv).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tmd import bench, hmm
from tmd.labels import CLASS_ID

K = 5
STEP_S, MAX_GAP_S = 60.0, 600.0


def labelled_minutes(source: str) -> pd.DataFrame:
    """All labelled minutes of the dev users, before caps and drops (ExtraSensory `cls`; SHL `cls`, else the
    majority class of mixed minutes). The excluded user is left out."""
    lab = bench.read(source, "labels")
    lab = lab[~lab.user.isin(bench.reserved_users(source)) & ~lab.user_excluded]
    c = lab.cls.astype(str)
    if "majority_cls" in lab:
        c = c.where(c != "", lab.majority_cls.astype(str))
    lab = lab.assign(y8=c.map(CLASS_ID).fillna(0).astype(int))
    return lab[lab.y8 > 0][["user", "session", "ts", "fold", "y8"]].assign(source=source)


def seq_ids(df: pd.DataFrame) -> np.ndarray:
    return pd.factorize(df.source.astype(str) + "/" + df.user.astype(str) + "/" + df.session.astype(str))[0]


def hmm_params(lab: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Transitions per 1-min step (+1 pseudo-count) and initial distribution (class frequencies, +1)."""
    a = hmm.transition_matrix(lab.y8.to_numpy(), lab.ts.to_numpy(float), seq_ids(lab), alpha=1.0, step_s=STEP_S,
                              tol_s=30.5)
    pi = np.bincount(lab.y8.to_numpy(), minlength=9)[1:] + 1.0
    return a, pi / pi.sum()


def smooth(df: pd.DataFrame, p: np.ndarray, a: np.ndarray, pi: np.ndarray, method: str) -> np.ndarray:
    seq, t = seq_ids(df), df.ts.to_numpy(float)
    f = {"forward": hmm.forward, "fb": hmm.forward_backward, "viterbi": hmm.viterbi}[method]
    return f(p, t, seq, a, max_gap_s=MAX_GAP_S, step_s=STEP_S, pi=pi)


def smooth_cv(df: pd.DataFrame, p: np.ndarray, lab: pd.DataFrame, method: str) -> np.ndarray:
    """Per outer fold k: HMM parameters from the labelled minutes of the training folds, applied to fold k."""
    out = np.full_like(p, np.nan, dtype=np.float64)
    fold = df.fold.to_numpy()
    for k in range(K):
        a, pi = hmm_params(lab[(lab.fold != k) & (lab.fold >= 0)])
        sc = fold == k
        out[sc] = smooth(df[sc], p[sc], a, pi, method)
    return out
