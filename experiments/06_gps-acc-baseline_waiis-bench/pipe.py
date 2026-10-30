"""Experiment 06: shared pieces of the rows (data, models, fusion, evaluation, latency, logging).

Rows of the frame = 05's usable minutes (ExtraSensory dev + SHL User 1, same order as 05's feature tables), with
the 580 acc features (05's B) and the 16 GPS features (gps-v1). Probabilities of every row are saved per row and seed
(`oof/<row>_seed<k>.npz`: raw = before the HMM, p = after) so later rows fuse and compare them.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from make_gps import OUT, P05, gps_path
from tmd import acc_features as af
from tmd import bench, lgbm, minutes
from tmd import gps_features as gf
from tmd.evaluate import (bootstrap_users_diff, bootstrap_users_recall, class_list, family_proba, plot_cm,
                          scores)
from tmd.labels import CLASSES, to_family
from tmd.tracking import DataInput

HERE = Path(__file__).resolve().parent
EXP = HERE.name
OOF = OUT / "oof"
ACC = af.names(True, False)                     # 05's B: 580 features
GPS = gf.NAMES                                  # 16
LR = {0.1: {"learning_rate": 0.1, "max_rounds": 1000, "patience": 30},
      0.02: {"learning_rate": 0.02, "max_rounds": 10000, "patience": 150}}


def load(part: str = "dev") -> pd.DataFrame:
    srcs = ["extrasensory", "shl_user1"] if part == "dev" else ["extrasensory", "shl_user23"]
    keys = ["user", "ts", "session", "official", "y", "platform", "fold", "quiet"]
    d = []
    for s in srcs:
        a = pd.read_parquet(P05 / f"{s}_{part}_25hz_w15s_acc-v1_lp-none.parquet", columns=keys + ACC)
        g = pd.read_parquet(gps_path(s, part), columns=["user", "ts"] + GPS)
        assert (a.user.to_numpy() == g.user.to_numpy()).all() and (a.ts.to_numpy() == g.ts.to_numpy()).all()
        d.append(pd.concat([a, g[GPS]], axis=1).assign(source=s))
    df = pd.concat(d, ignore_index=True)
    df["y"], df["fold"] = df.y.astype(int), df.fold.astype(int)
    df["covered"] = gf.covered(df[GPS].to_numpy(np.float32))
    return df


def labelled() -> pd.DataFrame:
    return pd.concat([minutes.labelled_minutes(s) for s in ("extrasensory", "shl_user1")], ignore_index=True)


def params(lr: float) -> tuple[dict, int, int]:
    c = LR[lr]
    return {**lgbm.PARAMS, "learning_rate": c["learning_rate"]}, c["max_rounds"], c["patience"]


def train(df, cols, seed, lr, rows_mask, shl=True, log=None, name=""):
    """Paired-fold LightGBM on `cols`, trained on official rows in rows_mask (SHL rows only if shl)."""
    x = df[cols].to_numpy(np.float32)
    is_shl = (df.source != "extrasensory").to_numpy()
    sw = np.where(is_shl, 1.0 if shl else 0.0, 1.0)
    trainable = df.official.to_numpy() & rows_mask & (sw > 0) & (df.fold.to_numpy() >= 0)
    prm, mr, pat = params(lr)
    return lgbm.paired_cv(x, df.y.to_numpy(), df.fold.to_numpy(), trainable, sw, seed, prm, mr, pat, log, name), \
        (x, trainable, sw)


def save(name, seed, df, raw, p, extra=None):
    OOF.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OOF / f"{name}_seed{seed}.npz", raw=raw, p=p, user=df.user.to_numpy(str),
                        ts=df.ts.to_numpy(), **(extra or {}))


def load_probs(name, seed, df, key="p"):
    z = np.load(OOF / f"{name}_seed{seed}.npz", allow_pickle=True)
    assert (z["user"] == df.user.to_numpy(str)).all() and (z["ts"] == df.ts.to_numpy()).all(), "row order"
    return z[key]


def product(p_acc, p_gps, cov):
    """H1: p ∝ p_acc · p_gps on covered rows; p_acc elsewhere."""
    out = p_acc.astype(np.float64).copy()
    q = np.maximum(p_acc[cov], 1e-9) * np.maximum(p_gps[cov], 1e-9)
    out[cov] = q / q.sum(1, keepdims=True)
    return out


def stack_features(p_acc, p_gps):
    return np.concatenate([np.log(np.clip(p_acc, 1e-6, 1)), np.log(np.clip(p_gps, 1e-6, 1))], 1)


def fit_stacker(z, y) -> LogisticRegression:
    """Multinomial LR, balanced class weights, scikit-learn's default regularisation (L2, C = 1.0)."""
    return LogisticRegression(class_weight="balanced", max_iter=1000).fit(z, y)


def stacker_proba(m: LogisticRegression, z) -> np.ndarray:
    out = np.zeros((len(z), 8))
    out[:, m.classes_ - 1] = m.predict_proba(z)
    return out


# ---------------------------------------------------------------- evaluation

def _rec_ci(m, pre, yy, pp, users, lvl):
    st = {}
    for c, (lo, hi) in bootstrap_users_recall(yy, pp, users, lvl, stats=st).items():
        key = f"{pre}_{lvl}_f1" if c == "macro_f1" else f"{pre}_{lvl}_recall_{c}"
        m[f"{key}_lo"], m[f"{key}_hi"] = lo, hi
    m[f"{pre}_{lvl}_ci_redraw_rate"] = st["redraw_rate"]


def evaluate(df, p, base_p=None, score_mask=None):
    """ES dev pooled OOF (CIs, covered / uncovered split, paired diff vs base_p) and SHL User 1 (day folds, no CI).
    score_mask restricts the scored rows (G: covered minutes only)."""
    m, figs, tab = {}, {}, {}
    sm = np.ones(len(df), bool) if score_mask is None else score_mask
    for src, pre in (("extrasensory", "es"), ("shl_user1", "shl")):
        sel = ((df.source == src) & df.official).to_numpy() & sm
        d, ps = df[sel], p[sel]
        y8 = d.y.to_numpy()
        preds = {"class": ps.argmax(1) + 1, "family": family_proba(ps).argmax(1) + 1}
        users, cov = d.user.to_numpy(), d.covered.to_numpy()
        for lvl, pp in preds.items():
            yy = y8 if lvl == "class" else to_family(y8)
            s = scores(yy, pp, lvl)
            m[f"{pre}_{lvl}_f1"] = s["macro_f1"]
            tab[f"{pre}_{lvl}_classes"] = class_list(yy, lvl)
            for c, r in s["recall"].items():
                if r == r:
                    m[f"{pre}_{lvl}_recall_{c}"] = r
            figs[f"cm_{pre}_{lvl}.png"] = plot_cm(s["cm"], lvl, f"{pre} {lvl}")
            for part, mask in (("covered", cov), ("uncovered", ~cov)):
                if mask.any():
                    m[f"{pre}_{lvl}_f1_{part}"] = scores(yy[mask], pp[mask], lvl)["macro_f1"]
                    tab[f"{pre}_{lvl}_classes_{part}"] = class_list(yy[mask], lvl)
            if pre == "es":
                _rec_ci(m, pre, yy, pp, users, lvl)
                per = [scores(yy[d.fold.to_numpy() == k], pp[d.fold.to_numpy() == k], lvl)["macro_f1"]
                       for k in range(lgbm.K)]
                m[f"es_{lvl}_f1_fold_min"], m[f"es_{lvl}_f1_fold_max"] = min(per), max(per)
                for plat in ("iphone", "android"):
                    pm = (d.platform == plat).to_numpy()
                    m[f"es_{lvl}_f1_{plat}"] = scores(yy[pm], pp[pm], lvl)["macro_f1"]
            if base_p is not None:
                bs = base_p[sel]
                bp = bs.argmax(1) + 1 if lvl == "class" else family_proba(bs).argmax(1) + 1
                if pre == "es":
                    st = {}
                    dd = bootstrap_users_diff(yy, pp, bp, users, lvl, stats=st)
                    m[f"es_{lvl}_f1_diff"], m[f"es_{lvl}_f1_diff_lo"], m[f"es_{lvl}_f1_diff_hi"] = \
                        dd["diff"], dd["lo"], dd["hi"]
                else:
                    m[f"shl_{lvl}_f1_diff"] = m[f"shl_{lvl}_f1"] - scores(yy, bp, lvl)["macro_f1"]
        still = y8 == 1
        q = d.quiet.to_numpy().astype(bool)
        for nm, mk in (("quiet", still & q), ("not_quiet", still & ~q)):
            if mk.any():
                m[f"{pre}_still_recall_{nm}"] = float((preds["class"][mk] == 1).mean())
    return m, figs, tab


def datasets(features: str) -> list[DataInput]:
    kw = {"label_rules": "bench-v1", "features": features, "rate_hz": 25, "window_s": 15, "stride_s": 60,
          "oof_eval": True}
    return [DataInput("extrasensory_dev_v2", "training", **kw), DataInput("shl_user1_dev", "training", **kw)]


# ---------------------------------------------------------------- latency

def session_pool(df, seed, n=200):
    """200 fold-0 ExtraSensory sessions with >= 3 usable minutes: their windows and GPS fixes."""
    es = df[(df.source == "extrasensory") & (df.fold == 0)]
    cnt = es.groupby(["user", "session"]).size()
    pick = cnt[cnt >= 3].sample(min(n, int((cnt >= 3).sum())), random_state=seed).index
    keep = es.set_index(["user", "session"]).loc[pick].reset_index()[["user", "ts", "session"]]
    keys, w = bench.load_windows("extrasensory", keep)
    keys = keys.assign(source="extrasensory")
    gps = bench.read("extrasensory", "gps_per_minute")
    gps = gps[gps.user.isin(set(keep.user))]
    ses = []
    for (u, s), idx in keys.groupby(["user", "session"], sort=False).indices.items():
        ses.append((keys.iloc[idx], w[idx], gps[(gps.user == u) & (gps.session == s)]))
    return ses


def latency(pool, predict_session, predict_one) -> dict:
    """predict_session(keys, windows, gps) runs the whole pipeline on a session; predict_one(window, gps, t) one
    minute without the HMM. Single thread is the caller's job."""
    from tmd.evaluate import latency_ms, pipeline_latency
    out = {}
    k0, w0, g0 = next((k, w, g) for k, w, g in pool if len(g) >= 3)
    t = int(k0.ts.iloc[len(k0) // 2])
    out["lat_inference_ms_p50"], out["lat_inference_ms_p95"] = latency_ms(lambda: predict_one(w0[:1], g0, t))
    out.update(pipeline_latency(lambda s: predict_session(*s), pool, [len(s[0]) for s in pool]))
    return out


def model_bytes(models) -> float:
    return float(np.mean([len(m.model_to_string()) for m in models]))


def dump(tab):
    return json.loads(json.dumps(tab, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
