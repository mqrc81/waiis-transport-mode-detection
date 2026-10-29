"""Experiment 05: shared pieces of the ablation (data, paired-fold LightGBM, HMM, evaluation, logging).

Rows = configurations (Cfg). Out-of-fold probabilities of every usable minute are saved per row and seed, so later
rows compare against them (paired bootstrap) and the HMM rows reuse them.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import context as ctxmod
import lightgbm as lgb
import numpy as np
import pandas as pd
from make_features import OUT, table_path

from tmd import acc_features as af
from tmd import bench, hmm
from tmd.evaluate import (
    bootstrap_users_diff,
    bootstrap_users_recall,
    family_proba,
    plot_cm,
    scores,
)
from tmd.labels import CLASS_ID, CLASSES, FAMILIES, to_family
from tmd.tracking import DataInput

HERE = Path(__file__).resolve().parent
EXP = HERE.name
K = 5
PARAMS = dict(objective="multiclass", num_class=8, learning_rate=0.1, num_leaves=31, min_child_samples=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1)
MAX_ROUNDS, PATIENCE = 1000, 30
STEP_S, MAX_GAP_S = 60.0, 600.0
OOF = OUT / "oof"


@dataclass(frozen=True)
class Cfg:
    name: str
    sub: bool = False
    grav: bool = False
    ctx: str = "none"            # none | plain | masked (D')
    lp: float | None = None      # low-pass [Hz] before features
    shl_w: float = 1.0           # SHL row weight; 0 = ExtraSensory only

    def feature_names(self) -> list[str]:
        return af.names(self.sub, self.grav)


# ---------------------------------------------------------------- data

def load(cfg: Cfg, part: str = "dev") -> pd.DataFrame:
    """Usable minutes of ExtraSensory and SHL User 1 (dev), with key columns, `source` and the row's features."""
    cols = list(dict.fromkeys(["user", "ts", "session", "official", "y", "platform", "fold", "quiet"]
                              + cfg.feature_names() + af.CONTEXT_BASE))
    srcs = ["extrasensory", "shl_user1"] if part == "dev" else ["extrasensory", "shl_user23"]
    d = [pd.read_parquet(table_path(s, part, cfg.lp), columns=cols).assign(source=s) for s in srcs]
    df = pd.concat(d, ignore_index=True)
    df["y"] = df.y.astype(int)
    df["fold"] = df.fold.astype(int)
    return df


def context_block(df: pd.DataFrame, nb: np.ndarray) -> tuple[np.ndarray, list[str]]:
    return ctxmod.columns(df[af.CONTEXT_BASE].to_numpy(np.float32), nb, af.CONTEXT_BASE)


def class_weights(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Balanced over the (already source-weighted) rows: every present class gets the same total weight."""
    c = np.bincount(y, weights=w, minlength=9)[1:]
    cw = np.where(c > 0, c.sum() / (8 * np.maximum(c, 1e-12)), 0.0)
    return w * cw[y - 1]


def _dataset(x, y, w, ref=None):
    return lgb.Dataset(x, y - 1, weight=class_weights(y, w), reference=ref, free_raw_data=True)


def train_cv(cfg: Cfg, df: pd.DataFrame, seed: int, log) -> tuple[np.ndarray, list[dict], list[lgb.Booster], list[str]]:
    """Paired-fold CV. Returns OOF probabilities for every row of df (ES dev + SHL usable minutes), fold rows,
    models and feature names."""
    names = cfg.feature_names()
    x = df[names].to_numpy(np.float32)
    nb = None
    if cfg.ctx != "none":
        nb = ctxmod.neighbours(df)
        xc, cn = context_block(df, nb)
        x, names = np.concatenate([x, xc], 1), names + cn
    is_shl = (df.source != "extrasensory").to_numpy()
    off = df.official.to_numpy()
    sw = np.where(is_shl, cfg.shl_w, 1.0)
    y, fold = df.y.to_numpy(), df.fold.to_numpy()
    p = np.full((len(df), 8), np.nan, np.float32)
    rows, models = [], []
    rng = np.random.default_rng(seed)
    for k in range(K):
        inner = (k + 1) % K
        tr = off & (fold != k) & (fold >= 0) & (sw > 0)
        xk = x
        if cfg.ctx == "masked":                    # D': SHL training rows get ExtraSensory gap patterns
            donors = np.flatnonzero(tr & ~is_shl)
            shl_tr = np.flatnonzero(tr & is_shl)
            nbk = ctxmod.mask_like(nb, shl_tr, donors, rng)
            xk = x.copy()
            xk[shl_tr, len(cfg.feature_names()):] = context_block(df, nbk[shl_tr])[0]
        fit, val = tr & (fold != inner), tr & (fold == inner)
        d_in = _dataset(xk[fit], y[fit], sw[fit])
        d_va = _dataset(xk[val], y[val], sw[val], ref=d_in)
        m0 = lgb.train({**PARAMS, "seed": seed}, d_in, MAX_ROUNDS, valid_sets=[d_va],
                       callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
        rounds = max(m0.best_iteration, 1)
        m = lgb.train({**PARAMS, "seed": seed}, _dataset(xk[tr], y[tr], sw[tr]), rounds)
        sc = fold == k
        p[sc] = m.predict(x[sc])                   # scored rows: context as recorded
        models.append(m)
        row = {"fold": k, "rounds": rounds, "n_trees": m.num_trees(), "model_bytes": len(m.model_to_string()),
               "train_es": int((tr & ~is_shl).sum()), "train_shl": int((tr & is_shl).sum())}
        rows.append(row)
        log.info("%s fold %s", cfg.name, row)
        print(row, flush=True)
    return p, rows, models, names


def save_oof(cfg: Cfg, seed: int, df: pd.DataFrame, p: np.ndarray, extra: dict | None = None) -> Path:
    OOF.mkdir(parents=True, exist_ok=True)
    f = OOF / f"{cfg.name}_seed{seed}.npz"
    np.savez_compressed(f, p=p, user=df.user.to_numpy(str), ts=df.ts.to_numpy(), cfg=json.dumps(asdict(cfg)),
                        **(extra or {}))
    return f


def load_oof(name: str, seed: int, df: pd.DataFrame) -> np.ndarray:
    z = np.load(OOF / f"{name}_seed{seed}.npz")
    assert (z["user"] == df.user.to_numpy(str)).all() and (z["ts"] == df.ts.to_numpy()).all(), "row order differs"
    return z["p"]


# ---------------------------------------------------------------- HMM

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


def _seq(df: pd.DataFrame) -> np.ndarray:
    return pd.factorize(df.source.astype(str) + "/" + df.user.astype(str) + "/" + df.session.astype(str))[0]


def hmm_params(lab: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Transitions per 1-min step (+1 pseudo-count) and initial distribution (class frequencies, +1)."""
    a = hmm.transition_matrix(lab.y8.to_numpy(), lab.ts.to_numpy(float), _seq(lab), alpha=1.0, step_s=STEP_S,
                              tol_s=30.5)
    pi = np.bincount(lab.y8.to_numpy(), minlength=9)[1:] + 1.0
    return a, pi / pi.sum()


def smooth(df: pd.DataFrame, p: np.ndarray, a: np.ndarray, pi: np.ndarray, method: str) -> np.ndarray:
    seq, t = _seq(df), df.ts.to_numpy(float)
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


# ---------------------------------------------------------------- evaluation

def class_list(y: np.ndarray, level: str) -> str:
    ids, names = (range(1, 9), CLASSES) if level == "class" else (range(1, 6), FAMILIES)
    return ",".join(n for i, n in zip(ids, names) if (y == i).any())


def evaluate(df: pd.DataFrame, p: np.ndarray, base_p: np.ndarray | None = None) -> tuple[dict, dict, dict]:
    """Metrics on the official ES dev rows (pooled OOF, user-bootstrap CIs) and SHL User 1 (day folds, no CI)."""
    m, figs, tab = {}, {}, {}
    for src, pre in [("extrasensory", "es"), ("shl_user1", "shl")]:
        sel = ((df.source == src) & df.official).to_numpy()
        d, ps = df[sel], p[sel]
        y8, pred8 = d.y.to_numpy(), ps.argmax(1) + 1
        yf, predf = to_family(y8), family_proba(ps).argmax(1) + 1
        users = d.user.to_numpy()
        for lvl, yy, pp in [("family", yf, predf), ("class", y8, pred8)]:
            s = scores(yy, pp, "family" if lvl == "family" else "class")
            m[f"{pre}_{lvl}_f1"] = s["macro_f1"]
            tab[f"{pre}_{lvl}_classes"] = class_list(yy, lvl)
            for c, r in s["recall"].items():
                if r == r:
                    m[f"{pre}_{lvl}_recall_{c}"] = r
            figs[f"cm_{pre}_{lvl}.png"] = plot_cm(s["cm"], lvl, f"{pre} {lvl} (pooled OOF)")
            tab[f"cm_{pre}_{lvl}"] = s["cm"].tolist()
            if pre == "es":
                ci = bootstrap_users_recall(yy, pp, users, lvl)
                for c, (lo, hi) in ci.items():
                    key = f"es_{lvl}_f1" if c == "macro_f1" else f"es_{lvl}_recall_{c}"
                    m[f"{key}_lo"], m[f"{key}_hi"] = lo, hi
                per = [scores(yy[d.fold.to_numpy() == k], pp[d.fold.to_numpy() == k], lvl)["macro_f1"]
                       for k in range(K)]
                m[f"es_{lvl}_f1_fold_min"], m[f"es_{lvl}_f1_fold_max"] = min(per), max(per)
                for plat in ("iphone", "android"):
                    pm = (d.platform == plat).to_numpy()
                    m[f"es_{lvl}_f1_{plat}"] = scores(yy[pm], pp[pm], lvl)["macro_f1"]
                if base_p is not None:
                    bs = base_p[sel]
                    bp = (family_proba(bs).argmax(1) + 1) if lvl == "family" else bs.argmax(1) + 1
                    dd = bootstrap_users_diff(yy, pp, bp, users, lvl)
                    m[f"es_{lvl}_f1_diff"], m[f"es_{lvl}_f1_diff_lo"], m[f"es_{lvl}_f1_diff_hi"] = \
                        dd["diff"], dd["lo"], dd["hi"]
        if pre == "shl" and base_p is not None:
            bs = base_p[sel]
            m["shl_family_f1_diff"] = m["shl_family_f1"] - scores(yf, family_proba(bs).argmax(1) + 1,
                                                                  "family")["macro_f1"]
    return m, figs, tab


def datasets(features: str) -> list[DataInput]:
    kw = dict(label_rules="bench-v1", features=features, rate_hz=25, window_s=15, stride_s=60, oof_eval=True)
    return [DataInput("extrasensory_dev_v2", "training", **kw), DataInput("shl_user1_dev", "training", **kw)]


def feature_tag(cfg: Cfg) -> str:
    t = [af.FEATURES_VERSION, "whole"] + (["sub5s"] if cfg.sub else []) + (["grav"] if cfg.grav else [])
    t += [f"ctx-{cfg.ctx}"] if cfg.ctx != "none" else []
    return "_".join(t + [f"lp-{'none' if cfg.lp is None else f'{cfg.lp:g}hz'}"])
