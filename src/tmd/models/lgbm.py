"""Paired-fold LightGBM with balanced class weights and inner early stopping (from experiment 05's pipeline.py).

Outer fold k trains on the trainable rows of folds != k, scores every row of fold k. The number of rounds comes from
early stopping on fold (k + 1) mod 5 (fit on the other 3 training folds), then a refit on all 4 training folds. The
early-stopping fit's predictions on fold (k + 1) mod 5 are returned too (experiment 06 reuses them as one inner fit).
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np

K = 5
PARAMS = {"objective": "multiclass", "num_class": 8, "learning_rate": 0.1, "num_leaves": 31, "min_child_samples": 50,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "num_threads": 8,
          "verbose": -1}
MAX_ROUNDS, PATIENCE = 1000, 30


def class_weights(y: np.ndarray, w: np.ndarray, n: int = 8) -> np.ndarray:
    """Balanced over the (already source-weighted) rows: every present class gets the same total weight.
    n = number of classes (8, or 9 with tram)."""
    c = np.bincount(y, weights=w, minlength=n + 1)[1:]
    cw = np.where(c > 0, c.sum() / (n * np.maximum(c, 1e-12)), 0.0)
    return w * cw[y - 1]


def dataset(x, y, w, ref=None, n: int = 8) -> lgb.Dataset:
    """y in 1..n; w = source weights (class weights are applied on top)."""
    return lgb.Dataset(x, y - 1, weight=class_weights(y, w, n), reference=ref, free_raw_data=True)


def paired_cv(x: np.ndarray, y: np.ndarray, fold: np.ndarray, trainable: np.ndarray, sw: np.ndarray, seed: int,
              params: dict | None = None, max_rounds: int = MAX_ROUNDS, patience: int = PATIENCE, log=None,
              name: str = "", progress=None, on_fold=None, stop_ok: np.ndarray | None = None) -> dict:
    """Out-of-fold probabilities for every row with fold >= 0.

    trainable: rows that may be trained on (official, sw > 0); sw: per-row source weight. Returns a dict with
    p (N, num_class; NaN for fold < 0), rows (per-fold info), models, and inner = {k: (row indices of fold (k+1)%5,
    probabilities of the early-stopping fit on them)}.
    progress (a tmd.progress.Progress): one step per fit, 2 per fold. on_fold(row, p) is called as each fold
    finishes, with the fold's info and p filled up to that fold. Neither changes the results.
    stop_ok: rows that may enter the early-stopping fold (default all trainable ones; experiment 10 leaves tram out).
    """
    params = {**(params or PARAMS), "seed": seed}
    n = params["num_class"]
    p = np.full((len(y), n), np.nan, np.float32)
    rows, models, inner_pred = [], [], {}
    for k in range(K):
        inner = (k + 1) % K
        tr = trainable & (fold != k) & (fold >= 0)
        fit, val = tr & (fold != inner), tr & (fold == inner)
        if stop_ok is not None:
            val &= stop_ok
        d_in = dataset(x[fit], y[fit], sw[fit], n=n)
        d_va = dataset(x[val], y[val], sw[val], ref=d_in, n=n)
        m0 = lgb.train(params, d_in, max_rounds, valid_sets=[d_va],
                       callbacks=[lgb.early_stopping(patience, verbose=False)])
        rounds = max(m0.best_iteration, 1)
        idx = np.flatnonzero(fold == inner)
        inner_pred[k] = (idx, m0.predict(x[idx], num_iteration=rounds))
        if progress is not None:
            progress.step(f"{name} fold {k} early stopping ({rounds} rounds)")
        m = lgb.train(params, dataset(x[tr], y[tr], sw[tr], n=n), rounds)
        sc = fold == k
        p[sc] = m.predict(x[sc])
        models.append(m)
        row = {"fold": k, "rounds": rounds, "n_trees": m.num_trees(), "model_bytes": len(m.model_to_string()),
               "hit_cap": bool(rounds >= max_rounds), "train_rows": int(tr.sum())}
        rows.append(row)
        if log is not None:
            log.info("%s fold %s", name, row)
        print(name, row, flush=True)
        if progress is not None:
            progress.step(f"{name} fold {k} refit")
        if on_fold is not None:
            on_fold(row, p)
    return {"p": p, "rows": rows, "models": models, "inner": inner_pred}
