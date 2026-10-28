"""Group-level splits (whole users or whole days), chosen by seeded random search.

Splits are fixed once and committed as files; these helpers only produce the proposal.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def balanced_folds(counts: pd.DataFrame, k: int, seed: int = 0, n_iter: int = 20000,
                   strata: pd.Series | None = None) -> pd.Series:
    """Assign each row (group: a day or a user) of `counts` (groups x classes) to one of k folds.

    Every fold must contain every class that occurs in at least k groups (a class in fewer groups
    cannot be in every fold; it is still balanced as far as possible). Among valid assignments the one
    whose per-fold class shares are closest to 1/k wins (sum of squared deviations). With `strata`
    (e.g. platform), groups are dealt round-robin within each stratum so folds stay balanced in it.
    """
    rng = np.random.default_rng(seed)
    c = counts.to_numpy(float)
    tot = c.sum(0)
    need = (c > 0).sum(0) >= k
    score_on = tot > 0
    best, best_score = None, np.inf
    for _ in range(n_iter):
        f = np.empty(len(c), int)
        if strata is None:
            f[rng.permutation(len(c))] = np.arange(len(c)) % k
        else:
            off = rng.integers(k)
            for s in sorted(strata.unique()):
                idx = np.flatnonzero(strata.to_numpy() == s)
                f[rng.permutation(idx)] = (np.arange(len(idx)) + off) % k
                off += len(idx)
        per = np.stack([c[f == j].sum(0) for j in range(k)])
        if (per[:, need] == 0).any():
            continue
        score = ((per[:, score_on] / tot[score_on] - 1 / k) ** 2).sum()
        if score < best_score:
            best, best_score = f, score
    if best is None:
        raise ValueError("no assignment puts every class in every fold")
    return pd.Series(best, index=counts.index, name="fold")


def pick_test_groups(counts: pd.DataFrame, n: int, valid: Callable[[pd.Index], bool], seed: int = 0,
                     n_iter: int = 200000, candidates: pd.Index | None = None) -> pd.Index:
    """Pick n groups (rows of `counts`) as test set: among subsets that pass `valid`, the one whose
    per-class share of the total is closest to n / len(counts) (sum of squared deviations)."""
    rng = np.random.default_rng(seed)
    pool = counts.index if candidates is None else candidates
    tot = counts.sum(0).to_numpy(float)
    need = tot > 0
    target = n / len(counts)
    best, best_score = None, np.inf
    for _ in range(n_iter):
        sel = pd.Index(rng.choice(pool, n, replace=False))
        if not valid(sel):
            continue
        share = counts.loc[sel].sum(0).to_numpy(float)[need] / tot[need]
        score = ((share - target) ** 2).sum()
        if score < best_score:
            best, best_score = sel, score
    if best is None:
        raise ValueError("no subset satisfies the constraints")
    return best.sort_values()


# --- reserved final-test set ---
# 1: splits/reserved_test.json, approved 2026-10-25 (experiment 02), superseded.
# 2: splits/reserved_test_v2.json, ExtraSensory users re-picked on es_bench_v2 (experiment 04), approved by the
#    supervisor 2026-10-28. The reference. SHL Users 2-3 are the same in both.
RESERVED_FILES = {1: "reserved_test.json", 2: "reserved_test_v2.json"}
RESERVED_VERSION = 2


def reserved_test(version: int = RESERVED_VERSION) -> dict[str, pd.DataFrame]:
    """The final-test set no experiment may train, tune or pre-train on: {"shl": user, day; "extrasensory": uuid,
    platform}. SHL Users 2 and 3 are reserved with all their days and positions. version=1 reads the superseded
    2026-10-25 set."""
    import json

    from tmd.paths import PROJECT
    d = json.loads((PROJECT / "splits" / RESERVED_FILES[version]).read_text())
    return {"shl": pd.DataFrame(d["shl"], dtype=str), "extrasensory": pd.DataFrame(d["extrasensory"])}
