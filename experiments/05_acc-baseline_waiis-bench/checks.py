"""Leak checks of steps D and E (small fixed LightGBM, out of fold; one MLflow run per check).

    uv run python checks.py dataset --row D        # D: SHL vs ExtraSensory from context / gap pattern / own minute
    uv run python checks.py dataset --row Dm       # D': same, SHL training rows with ExtraSensory gap patterns
    uv run python checks.py platform --row C       # E: iPhone vs Android within walk and still, with and without 10 Hz
"""
from __future__ import annotations

import argparse
import json
import logging

import context as ctxmod
import lightgbm as lgb
import mlflow
import numpy as np
import pipeline as pl
from sklearn.metrics import f1_score, roc_auc_score

from tmd import acc_features as af
from tmd.labels import to_family
from tmd.tracking import run

SMALL = dict(objective="binary", learning_rate=0.1, num_leaves=15, min_child_samples=50, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, num_threads=8, verbose=-1)
ROUNDS = 100
log = logging.getLogger()


def oof(x, target, fold, seed, params=SMALL, weight=None):
    p = np.full(len(target), np.nan)
    for k in range(pl.K):
        tr, te = fold != k, fold == k
        m = lgb.train({**params, "seed": seed}, lgb.Dataset(x[tr], target[tr],
                                                             weight=None if weight is None else weight[tr]), ROUNDS)
        p[te] = m.predict(x[te])
    return p


def check_dataset(cfg: pl.Cfg, seed: int) -> dict:
    df = pl.load(cfg)
    nb = ctxmod.neighbours(df)
    off = df.official.to_numpy()
    is_shl = (df.source != "extrasensory").to_numpy()
    fold = df.fold.to_numpy()
    if cfg.ctx == "masked":   # the training distribution of D': SHL rows carry ExtraSensory gap patterns
        rng = np.random.default_rng(seed)
        nb = ctxmod.mask_like(nb, np.flatnonzero(is_shl), np.flatnonzero(off & ~is_shl), rng)
    ctx, _ = pl.context_block(df, nb)
    inputs = {"context": ctx, "gap_pattern": (nb >= 0).astype(np.float32),
              "own_minute": df[af.CONTEXT_BASE].to_numpy(np.float32)}
    m = {}
    for name, x in inputs.items():
        p = oof(x[off], is_shl[off].astype(int), fold[off], seed)
        m[f"auc_dataset_{name}"] = roc_auc_score(is_shl[off], p)
    m["check_pass"] = float(m["auc_dataset_context"] <= m["auc_dataset_own_minute"] + 0.02
                            and m["auc_dataset_gap_pattern"] <= 0.60)
    # descriptive: family from the gap pattern alone, ExtraSensory dev
    es = off & ~is_shl
    yf = to_family(df.y.to_numpy()[es])
    pm = np.zeros((es.sum(), 5))
    xg = inputs["gap_pattern"][es]
    multi = {**SMALL, "objective": "multiclass", "num_class": 5}
    cnt = np.bincount(yf, minlength=6)[1:]
    w = (cnt.sum() / (5 * np.maximum(cnt, 1)))[yf - 1]
    for k in range(pl.K):
        tr, te = fold[es] != k, fold[es] == k
        mm = lgb.train({**multi, "seed": seed}, lgb.Dataset(xg[tr], yf[tr] - 1, weight=w[tr]), ROUNDS)
        pm[te] = mm.predict(xg[te])
    present = sorted(set(yf))
    m["family_from_gaps_f1"] = f1_score(yf, pm.argmax(1) + 1, labels=present, average="macro", zero_division=0)
    m["family_majority_f1"] = f1_score(yf, np.full_like(yf, np.bincount(yf).argmax()), labels=present,
                                       average="macro", zero_division=0)
    return m


def check_platform(cfg: pl.Cfg, seed: int) -> dict:
    m = {}
    for lp in (None, 10.0):
        c = pl.Cfg(cfg.name, cfg.sub, cfg.grav, "none", lp, cfg.shl_w)
        df = pl.load(c)
        sel = ((df.source == "extrasensory") & df.official).to_numpy()
        d = df[sel]
        x = d[c.feature_names()].to_numpy(np.float32)
        tag = "lp10" if lp else "lpnone"
        aucs, ns = [], []
        for cls, cid in (("still", 1), ("walk", 2)):
            s = (d.y == cid).to_numpy()
            ip = (d.platform[s] == "iphone").to_numpy().astype(int)
            p = oof(x[s], ip, d.fold.to_numpy()[s], seed)
            m[f"auc_platform_{cls}_{tag}"] = roc_auc_score(ip, p)
            aucs.append(m[f"auc_platform_{cls}_{tag}"])
            ns.append(s.sum())
        m[f"auc_platform_mean_{tag}"] = float(np.average(aucs, weights=ns))
    m["auc_platform_change"] = m["auc_platform_mean_lp10"] - m["auc_platform_mean_lpnone"]
    m["check_pass"] = float(m["auc_platform_change"] <= 0)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("check", choices=["dataset", "platform"])
    ap.add_argument("--row", required=True, help="saved row whose configuration is checked")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(filename=pl.HERE / "logs" / f"check_{a.check}_{a.row}.log", filemode="w",
                        level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = pl.Cfg(**json.loads(str(np.load(pl.OOF / f"{a.row}_seed{a.seed}.npz")["cfg"])))
    m = check_dataset(cfg, a.seed) if a.check == "dataset" else check_platform(cfg, a.seed)
    params = {"model": "lightgbm-check", "check": a.check, "row": a.row, "seed": a.seed, "rate_hz": 25,
              "class_set": "dataset" if a.check == "dataset" else "platform", "dataset": "es_dev+shl_u1"
              if a.check == "dataset" else "es_dev", "features": pl.feature_tag(cfg), "rounds": ROUNDS,
              **{f"lgb_{k}": v for k, v in SMALL.items()}}
    ds = pl.datasets(pl.feature_tag(cfg)) if a.check == "dataset" else pl.datasets(pl.feature_tag(cfg))[:1]
    with run(pl.EXP, params, run_name=f"check_{a.check}_{a.row}_seed{a.seed}", datasets=ds):
        mlflow.log_metrics({k: float(v) for k, v in m.items()})
    log.info("%s", m)
    print(a.check, a.row, {k: round(v, 4) for k, v in m.items()})


if __name__ == "__main__":
    main()
