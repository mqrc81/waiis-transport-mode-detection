"""Checks of 06 (small fixed LightGBM as in 05, out of fold; one MLflow run per check).

    uv run python checks.py gps          # check 1: dataset and platform from GPS vs acc features, within walk / car
    uv run python checks.py missing      # check 2 (row I, and K): from the GPS-missingness pattern, within walk / car
    uv run python checks.py coverage     # check 3, descriptive: GPS coverage per class, dataset and platform
"""
from __future__ import annotations

import argparse
import json
import logging

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import pipe
from sklearn.metrics import roc_auc_score

from tmd.labels import CLASSES
from tmd.tracking import run

SMALL = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 15, "min_child_samples": 50,
         "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "num_threads": 8, "verbose": -1}
ROUNDS = 100
CLS = {"walk": 2, "car": 5}
log = logging.getLogger()


def auc_oof(x, target, fold, seed):
    p = np.full(len(target), np.nan)
    for k in range(5):
        tr, te = fold != k, fold == k
        if te.any() and len(np.unique(target[tr])) == 2:
            m = lgb.train({**SMALL, "seed": seed}, lgb.Dataset(x[tr], target[tr]), ROUNDS)
            p[te] = m.predict(x[te])
    ok = ~np.isnan(p)
    return float(roc_auc_score(target[ok], p[ok]))


def check_gps(df, seed):
    m = {}
    off, cov = df.official.to_numpy(), df.covered.to_numpy()
    shl = (df.source != "extrasensory").to_numpy()
    xa, xg = df[pipe.ACC].to_numpy(np.float32), df[pipe.GPS].to_numpy(np.float32)
    fold = df.fold.to_numpy()
    for c, cid in CLS.items():
        s = off & cov & (df.y.to_numpy() == cid)
        for nm, x in (("gps", xg), ("acc", xa)):
            m[f"auc_dataset_{c}_{nm}"] = auc_oof(x[s], shl[s].astype(int), fold[s], seed)
        e = s & ~shl
        ip = (df.platform.to_numpy() == "iphone").astype(int)
        for nm, x in (("gps", xg), ("acc", xa)):
            m[f"auc_platform_{c}_{nm}"] = auc_oof(x[e], ip[e], fold[e], seed)
    passed = [m[f"auc_{t}_{c}_gps"] <= m[f"auc_{t}_{c}_acc"] + 0.02 for t in ("dataset", "platform") for c in CLS]
    m["check_pass"] = float(all(passed))
    return m


def check_missing(df, seed):
    m = {}
    off = df.official.to_numpy()
    shl = (df.source != "extrasensory").to_numpy()
    bits = np.isnan(df[pipe.GPS].to_numpy(np.float32)).astype(np.float32)
    fold = df.fold.to_numpy()
    ip = (df.platform.to_numpy() == "iphone").astype(int)
    for c, cid in list(CLS.items()) + [("all", None)]:
        s = off & ((df.y.to_numpy() == cid) if cid else True)
        m[f"auc_dataset_{c}_missing"] = auc_oof(bits[s], shl[s].astype(int), fold[s], seed)
        e = s & ~shl
        m[f"auc_platform_{c}_missing"] = auc_oof(bits[e], ip[e], fold[e], seed)
    m["check_pass"] = float(all(m[f"auc_{t}_{c}_missing"] <= 0.60 for t in ("dataset", "platform") for c in CLS))
    return m


def coverage(df):
    d = df[df.official].copy()
    d["cls"] = d.y.map(lambda i: CLASSES[i - 1])
    d["group"] = np.where(d.source == "extrasensory", "ES " + d.platform.astype(str), "SHL User 1")
    t = d.pivot_table(index="group", columns="cls", values="covered", aggfunc="mean")[CLASSES]
    t["all"] = d.groupby("group").covered.mean()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("check", choices=["gps", "missing", "coverage"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gps", default="gps-v1", help="GPS feature version")
    a = ap.parse_args()
    pipe.use_gps(a.gps)
    tag = "" if a.gps == "gps-v1" else f"_{a.gps}"
    logging.basicConfig(filename=pipe.HERE / "logs" / f"check_{a.check}{tag}.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    df = pipe.load()
    if a.check == "coverage":
        t = coverage(df)
        t.to_csv(pipe.HERE / "results" / "gps_coverage_dev.csv")
        print(t.round(2).to_string())
        return
    m = check_gps(df, a.seed) if a.check == "gps" else check_missing(df, a.seed)
    prm = {"model": "lightgbm-check", "check": a.check, "row": f"check_{a.check}{tag}", "seed": a.seed, "rate_hz": 25,
           "class_set": "dataset/platform", "dataset": "es_dev+shl_u1", "features": f"{a.gps}_w2min vs acc-v1", "gps_features": a.gps,
           "rounds": ROUNDS, "within_class": "walk, car", **{f"lgb_{k}": v for k, v in SMALL.items()}}
    with run(pipe.EXP, prm, run_name=f"check_{a.check}{tag}_seed{a.seed}", datasets=pipe.datasets(f"{a.gps}_w2min")):
        mlflow.log_metrics(m)
    log.info("%s", json.dumps(m))
    print(a.check, {k: round(v, 3) for k, v in m.items()})


if __name__ == "__main__":
    main()
