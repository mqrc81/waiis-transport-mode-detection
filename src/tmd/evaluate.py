"""Evaluation: macro-F1, per-class recall, confusion matrices (8 classes and family), user-bootstrap CIs, latency."""
from __future__ import annotations

import time
from collections.abc import Callable

import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from tmd.labels import CLASSES, FAMILIES, to_family

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLASS_IDS = list(range(1, 9))
FAMILY_IDS = list(range(1, 6))


def family_proba(p8: np.ndarray) -> np.ndarray:
    """(N, 8) class probabilities (columns = class ids 1..8) -> (N, 5) family probabilities."""
    fam = to_family(np.arange(1, 9)) - 1
    out = np.zeros((len(p8), 5))
    for j in range(8):
        out[:, fam[j]] += p8[:, j]
    return out


def scores(y: np.ndarray, pred: np.ndarray, level: str) -> dict:
    """Macro-F1 over the classes present in y (a dataset may lack some, e.g. subway in ExtraSensory),
    accuracy, per-class recall (NaN when absent) and the confusion matrix over all classes."""
    ids, names = (CLASS_IDS, CLASSES) if level == "class" else (FAMILY_IDS, FAMILIES)
    present = [i for i in ids if (y == i).any()]
    rec = recall_score(y, pred, labels=ids, average=None, zero_division=0)
    return {
        "n": int(len(y)),
        "macro_f1": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
        "acc": float((y == pred).mean()),
        "recall": {n: (float(r) if (y == i).any() else float("nan")) for i, n, r in zip(ids, names, rec)},
        "cm": confusion_matrix(y, pred, labels=ids),
    }


def bootstrap_users(y, pred, users, level: str, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% CI of macro-F1 from resampling users with replacement (approved 2026-10-25)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(users)
    idx = {u: np.flatnonzero(users == u) for u in uniq}
    vals = []
    for _ in range(n_boot):
        sel = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq))])
        vals.append(scores(y[sel], pred[sel], level)["macro_f1"])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bootstrap_users_recall(y, pred, users, level: str, n_boot: int = 1000, seed: int = 0) -> dict[str, tuple]:
    """95% CI of each class's recall (and macro-F1) from resampling users with replacement. A class absent from a
    resample is skipped for that resample; classes with few users (ExtraSensory train, run) get wide intervals."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(users)
    idx = {u: np.flatnonzero(users == u) for u in uniq}
    vals: dict[str, list] = {}
    for _ in range(n_boot):
        sel = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq))])
        s = scores(y[sel], pred[sel], level)
        vals.setdefault("macro_f1", []).append(s["macro_f1"])
        for k, r in s["recall"].items():
            if not np.isnan(r):
                vals.setdefault(k, []).append(r)
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in vals.items()}


def plot_cm(cm: np.ndarray, level: str, title: str):
    names = CLASSES if level == "class" else FAMILIES
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
                      n: int = 500, warmup: int = 50, seed: int = 0) -> dict:
    """Metric (c): inference time per window, single thread, after warm-up, median and p95 [ms].

    End to end (features + model on one 5 s window at 25 Hz) plus both parts. `predict` must run single-threaded
    (e.g. LightGBM `num_threads=1`).
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(1, 125, 6)).astype(np.float32)
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
