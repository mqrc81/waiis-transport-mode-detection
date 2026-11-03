"""Evaluation: macro-F1, per-class recall, confusion matrices (8 classes, family; v2: 9 fine classes and the six-class
score), user-bootstrap CIs, latency, identification AUC."""
from __future__ import annotations

import time
from collections.abc import Callable

import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from tmd.labels import CLASSES, FAMILIES, FINE_CLASSES, SCORE6, to_family, to_score6

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLASS_IDS = list(range(1, 9))
FAMILY_IDS = list(range(1, 6))
# level -> (ids, names): "class" = v1's 8, "fine" = v2's 9 (tram = 9), "six" = v2's primary score, "family"
LEVELS = {"class": (CLASS_IDS, CLASSES), "family": (FAMILY_IDS, FAMILIES),
          "fine": (list(range(1, 10)), FINE_CLASSES), "six": (list(range(1, 7)), SCORE6)}


def family_proba(p8: np.ndarray) -> np.ndarray:
    """(N, 8 or 9) class probabilities (columns = class ids 1..8, 9 = tram) -> (N, 5) family probabilities."""
    fam = to_family(np.arange(1, p8.shape[1] + 1)) - 1
    out = np.zeros((len(p8), 5))
    for j in range(p8.shape[1]):
        out[:, fam[j]] += p8[:, j]
    return out


def six_pred(p: np.ndarray) -> np.ndarray:
    """Six-class prediction (ids 1..6) from (N, 8 or 9) class probabilities: the class argmax, merged (09's
    rescoring of the logged confusion matrices, so v1's reference numbers are comparable)."""
    return to_score6(p.argmax(1) + 1)


def scores(y: np.ndarray, pred: np.ndarray, level: str) -> dict:
    """Macro-F1 over the classes present in y (a dataset may lack some, e.g. subway in ExtraSensory),
    accuracy, per-class recall (NaN when absent) and the confusion matrix over all classes."""
    ids, names = LEVELS[level]
    present = [i for i in ids if (y == i).any()]
    rec = recall_score(y, pred, labels=ids, average=None, zero_division=0)
    return {
        "n": int(len(y)),
        "macro_f1": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
        "acc": float((y == pred).mean()),
        "recall": {n: (float(r) if (y == i).any() else float("nan")) for i, n, r in zip(ids, names, rec)},
        "cm": confusion_matrix(y, pred, labels=ids),
    }


def class_list(y: np.ndarray, level: str) -> str:
    """The classes (or families) present in y, comma-separated: what a macro-F1 averages over."""
    ids, names = LEVELS[level]
    return ",".join(n for i, n in zip(ids, names) if (y == i).any())


def user_draws(y, users, n_boot: int = 1000, seed: int = 0, stats: dict | None = None):
    """Yield n_boot arrays of users (np.unique(users) order), each drawn with replacement. A draw that lacks a class
    present in y (the full evaluation set's true labels) is redrawn, so every resample is scored over the same
    classes; `stats["redraw_rate"]` = redraws / (redraws + n_boot)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(users)
    has = {u: set(np.unique(y[users == u])) for u in uniq}
    need = set(np.unique(y))
    redraws = 0
    for _ in range(n_boot):
        while True:
            draw = rng.choice(uniq, len(uniq))
            if need <= set().union(*(has[u] for u in set(draw))):
                break
            redraws += 1
        yield draw
    if stats is not None:
        stats["redraw_rate"] = redraws / (redraws + n_boot)


def user_resamples(y, users, n_boot: int = 1000, seed: int = 0, stats: dict | None = None):
    """Yield n_boot index arrays, each from drawing the users with replacement (`user_draws`)."""
    idx = {u: np.flatnonzero(users == u) for u in np.unique(users)}
    for draw in user_draws(y, users, n_boot, seed, stats):
        yield np.concatenate([idx[u] for u in draw])


def bootstrap_users(y, pred, users, level: str, n_boot: int = 1000, seed: int = 0,
                    stats: dict | None = None) -> tuple[float, float]:
    """95% CI of macro-F1 from resampling users with replacement (approved 2026-10-25). Resamples lacking a class of y
    are redrawn (experiment 05; `user_resamples`)."""
    vals = [scores(y[sel], pred[sel], level)["macro_f1"] for sel in user_resamples(y, users, n_boot, seed, stats)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bootstrap_users_recall(y, pred, users, level: str, n_boot: int = 1000, seed: int = 0,
                           stats: dict | None = None) -> dict[str, tuple]:
    """95% CI of each class's recall (and macro-F1) from resampling users with replacement. Resamples lacking a class
    of y are redrawn (`user_resamples`), so every class is in every resample; classes with few users (ExtraSensory
    train, run) get wide intervals."""
    vals: dict[str, list] = {}
    for sel in user_resamples(y, users, n_boot, seed, stats):
        s = scores(y[sel], pred[sel], level)
        vals.setdefault("macro_f1", []).append(s["macro_f1"])
        for k, r in s["recall"].items():
            if not np.isnan(r):
                vals.setdefault(k, []).append(r)
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in vals.items()}


def bootstrap_users_diff(y, pred_a, pred_b, users, level: str, n_boot: int = 1000, seed: int = 0,
                         stats: dict | None = None) -> dict:
    """Paired user bootstrap of macro-F1(pred_a) - macro-F1(pred_b) on the same examples: each resample draws users
    with replacement (redrawn if it lacks a class of y) and scores both predictions on it. Returns the point
    difference and its 95% CI."""
    vals = [scores(y[sel], pred_a[sel], level)["macro_f1"] - scores(y[sel], pred_b[sel], level)["macro_f1"]
            for sel in user_resamples(y, users, n_boot, seed, stats)]
    d = scores(y, pred_a, level)["macro_f1"] - scores(y, pred_b, level)["macro_f1"]
    return {"diff": float(d), "lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5))}


def cm_metrics(cm: np.ndarray, present: np.ndarray) -> dict:
    """From a confusion matrix (rows true, columns predicted; ids 1..K in order): macro-F1 over the `present` classes
    (bool mask; as `scores`, zero_division 0), per-class recall and precision (NaN when undefined)."""
    tp = np.diag(cm).astype(float)
    t, p = cm.sum(1).astype(float), cm.sum(0).astype(float)
    den = t + p
    f1 = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)
    return {"macro_f1": float(f1[present].mean()),
            "recall": np.divide(tp, t, out=np.full_like(tp, np.nan), where=t > 0),
            "precision": np.divide(tp, p, out=np.full_like(tp, np.nan), where=p > 0)}


def user_cms(y, preds: list, users, level: str) -> tuple[np.ndarray, list]:
    """Per-user confusion matrices of each prediction in preds (users in np.unique order): (users, [U x K x K])."""
    ids = LEVELS[level][0]
    k = len(ids)
    uniq, inv = np.unique(users, return_inverse=True)
    out = []
    for pr in preds:
        c = np.zeros((len(uniq), k, k), np.int64)
        np.add.at(c, (inv, np.asarray(y) - 1, np.asarray(pr) - 1), 1)
        out.append(c)
    return uniq, out


def bootstrap_cms(y, preds: list, users, level: str, n_boot: int = 1000, seed: int = 0,
                  stats: dict | None = None) -> list[list[dict]]:
    """User bootstrap on per-user confusion matrices (same draws and redraw rule as `user_resamples`): for each
    resample, `cm_metrics` of every prediction in preds (same examples, so differences are paired)."""
    ids = LEVELS[level][0]
    present = np.isin(ids, np.unique(y))
    uniq, cms = user_cms(y, preds, users, level)
    pos = {u: i for i, u in enumerate(uniq)}
    out = []
    for draw in user_draws(y, users, n_boot, seed, stats):
        ix = np.fromiter((pos[u] for u in draw), int, len(draw))
        out.append([cm_metrics(c[ix].sum(0), present) for c in cms])
    return out


def identification_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0) -> tuple[float, float, int]:
    """Cross-validated identification by group (from experiment 09's checks.py; StratifiedGroupKFold, LightGBM;
    groups = users, SHL User 1's days): macro one-vs-rest AUC and balanced accuracy; folds = min(5, groups of the
    smallest class of y)."""
    import lightgbm as lgb
    import pandas as pd
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    k = int(min(5, pd.Series(groups).groupby(y).nunique().min()))
    if k < 2:
        return np.nan, np.nan, k
    classes = np.unique(y)
    p = np.zeros((len(y), len(classes)))
    for tr, te in StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(X, y, groups):
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, class_weight="balanced",
                               random_state=seed, verbose=-1, n_jobs=8)
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])
    auc = roc_auc_score(y, p, multi_class="ovr", average="macro") if len(classes) > 2 else roc_auc_score(y, p[:, 1])
    return float(auc), float(balanced_accuracy_score(y, classes[p.argmax(1)])), k


def plot_cm(cm: np.ndarray, level: str, title: str):
    names = LEVELS[level][1]
    rown = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1 + 0.7 * len(names), 0.8 + 0.6 * len(names)))
    ax.imshow(rown, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, f"{rown[i, j]:.2f}\n{cm[i, j]}", ha="center", va="center", fontsize=6,
                        color="white" if rown[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (row-normalised recall, count)")
    ax.set_title(title, fontsize=8)
    fig.tight_layout()
    return fig


def latency_ms(fn: Callable[[], object], n: int = 500, warmup: int = 50) -> tuple[float, float]:
    """Median and p95 wall time [ms] of fn() after warm-up. Set single-thread env vars before importing numpy."""
    for _ in range(warmup):
        fn()
    t = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        fn()
        t[i] = time.perf_counter() - t0
    return float(np.median(t) * 1e3), float(np.percentile(t, 95) * 1e3)


def inference_latency(predict: Callable[[np.ndarray], object], featurize: Callable[[np.ndarray], np.ndarray],
                      n: int = 500, warmup: int = 50, seed: int = 0, shape: tuple = (1, 125, 6)) -> dict:
    """Metric (c): inference time per window, single thread, after warm-up, median and p95 [ms].

    End to end (features + model on one window at 25 Hz; default 5 s acc + gyro, `shape=(1, 375, 3)` for a 15 s
    accelerometer window) plus both parts. `predict` must run single-threaded (e.g. LightGBM `num_threads=1`).
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(size=shape).astype(np.float32)
    w[..., 2] += 9.81
    x = featurize(w)
    out = {}
    for name, fn in [("inference", lambda: predict(featurize(w))), ("features", lambda: featurize(w)),
                     ("model", lambda: predict(x))]:
        out[f"lat_{name}_ms_p50"], out[f"lat_{name}_ms_p95"] = latency_ms(fn, n, warmup)
    return out


def model_size_bytes(booster) -> int:
    """Metric (d): model size = bytes of the saved LightGBM text model."""
    return len(booster.model_to_string().encode())


def pipeline_latency(run_session: Callable[[object], object], sessions: list, n_windows: list[int]) -> dict:
    """Inference time per window of the pipeline as it runs: run_session(session) processes a whole session (features,
    context, model, smoothing). One warm-up pass over all sessions, then each session timed once; time / windows.
    Median and p95 over sessions [ms]. Single thread is the caller's job (env vars, num_threads=1)."""
    for s in sessions:
        run_session(s)
    per = np.empty(len(sessions))
    for i, s in enumerate(sessions):
        t0 = time.perf_counter()
        run_session(s)
        per[i] = (time.perf_counter() - t0) / n_windows[i]
    return {"lat_pipeline_ms_p50": float(np.median(per) * 1e3), "lat_pipeline_ms_p95": float(np.percentile(per, 95) * 1e3)}
