"""Platform transfer test of the GPS model (06 stop-point 2 follow-up, fixed in the Plan before running).

Per GPS version and direction: GPS model trained on the covered official minutes of one platform's ExtraSensory dev
users + SHL User 1 (lr 0.02, balanced weights; rounds from early stopping on the training rows of dev fold 0, then a
refit on all training rows), scored on the other platform's dev users. Fused with A's out-of-fold acc probabilities
(H1 product on covered minutes), HMM forward-backward per outer fold as in A. Reported on the held-out platform:
H1-transfer gain over A, the normal-fold H1's gain over A on the same platform, and their difference (paired
user-bootstrap CIs). One MLflow run per (version, direction).

    uv run python transfer.py --gps gps-v1      # normal-fold row: H1
    uv run python transfer.py --gps gps-v2      # normal-fold row: H1_v2
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

from tmd import lgbm, minutes
from tmd.evaluate import bootstrap_users_diff, family_proba, scores
from tmd.labels import to_family
from tmd.progress import Progress
from tmd.tracking import run

LR = 0.02
INNER_FOLD = 0
log = logging.getLogger()


def gps_transfer(df, train_plat, seed):
    """GPS-model probabilities (N, 8) for every row; trained on train_plat ES users + SHL User 1."""
    x = df[pipe.GPS].to_numpy(np.float32)
    y, fold = df.y.to_numpy(), df.fold.to_numpy()
    es = (df.source == "extrasensory").to_numpy()
    tr = df.official.to_numpy() & df.covered.to_numpy() & (fold >= 0) & \
        ((es & (df.platform == train_plat).to_numpy()) | ~es)
    sw = np.ones(len(df))
    prm, mr, pat = pipe.params(LR)
    prm = {**prm, "seed": seed}
    fit, val = tr & (fold != INNER_FOLD), tr & (fold == INNER_FOLD)
    m0 = lgb.train(prm, lgbm.dataset(x[fit], y[fit], sw[fit]), mr,
                   valid_sets=[lgbm.dataset(x[val], y[val], sw[val])],
                   callbacks=[lgb.early_stopping(pat, verbose=False)])
    rounds = max(m0.best_iteration, 1)
    m = lgb.train(prm, lgbm.dataset(x[tr], y[tr], sw[tr]), rounds)
    return m.predict(x), {"rounds": rounds, "hit_cap": bool(rounds >= mr), "train_rows": int(tr.sum()),
                          "train_es_users": int(df.user[tr & es].nunique())}


def gains(df, test_plat, p_tr, p_h1, p_a):
    """Paired user-bootstrap diffs on the held-out platform's official ES minutes."""
    sel = ((df.source == "extrasensory") & df.official & (df.platform == test_plat)).to_numpy()
    y8, users = df.y.to_numpy()[sel], df.user.to_numpy()[sel]
    out = {}
    for lvl in ("class", "family"):
        yy = y8 if lvl == "class" else to_family(y8)

        def pred(p):
            return (p[sel].argmax(1) + 1) if lvl == "class" else (family_proba(p[sel]).argmax(1) + 1)
        for nm, (pa, pb) in {"transfer_vs_A": (p_tr, p_a), "normal_vs_A": (p_h1, p_a),
                             "transfer_vs_normal": (p_tr, p_h1)}.items():
            st = {}
            d = bootstrap_users_diff(yy, pred(pa), pred(pb), users, lvl, stats=st)
            out[f"{lvl}_{nm}"], out[f"{lvl}_{nm}_lo"], out[f"{lvl}_{nm}_hi"] = d["diff"], d["lo"], d["hi"]
        for nm, p in (("A", p_a), ("normal", p_h1), ("transfer", p_tr)):
            out[f"{lvl}_f1_{nm}"] = scores(yy, pred(p), lvl)["macro_f1"]
    out["n_users"], out["n_minutes"] = float(len(np.unique(users))), float(sel.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gps", default="gps-v1")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    pipe.use_gps(a.gps)
    tag = "v1" if a.gps == "gps-v1" else "v2"
    normal = "H1" if a.gps == "gps-v1" else "H1_v2"
    logging.basicConfig(filename=pipe.HERE / "logs" / f"transfer_{tag}_seed{a.seed}.log", filemode="w",
                        level=logging.INFO, format="%(asctime)s %(message)s")
    df = pipe.load()
    lab = pipe.labelled()
    cov = df.covered.to_numpy()
    raw_a = pipe.load_probs("A", a.seed, df, "raw")
    p_a = pipe.load_probs("A", a.seed, df)
    p_h1 = pipe.load_probs(normal, a.seed, df)
    for train_plat, test_plat in (("iphone", "android"), ("android", "iphone")):
        name = f"transfer_{tag}_{train_plat}_to_{test_plat}"
        prog = Progress(pipe.EXP, f"{name}_seed{a.seed}", 3, log)
        prm = {"model": "lightgbm+H1+hmm-fb", "row": name, "seed": a.seed, "rate_hz": 25, "class_set": "8class+family",
               "dataset": f"es_dev_{train_plat}+shl_u1 -> es_dev_{test_plat}", "features": f"acc-v1_whole_sub5s+{a.gps}_w2min",
               "gps_features": a.gps, "lr": LR, "fusion": "product", "smoothing": "fb", "acc_row": "A",
               "normal_row": normal, "inner_fold": INNER_FOLD, "primary": "es 8-class macro-F1 on held-out platform",
               "bootstrap": "users, redraw resamples lacking a class", **{f"lgb_{k}": v for k, v in pipe.params(LR)[0].items()}}
        with run(pipe.EXP, prm, run_name=f"{name}_seed{a.seed}", datasets=pipe.datasets(prm["features"])):
            p_g, info = gps_transfer(df, train_plat, a.seed)
            prog.step("GPS model (early stopping + refit)", 2)
            held = cov & (df.platform == test_plat).to_numpy() & (df.source == "extrasensory").to_numpy()
            raw = raw_a.astype(np.float64).copy()
            raw[held] = pipe.product(raw_a[held], p_g[held], np.ones(held.sum(), bool))
            p_tr = minutes.smooth_cv(df, raw, lab, "fb")
            m = {**gains(df, test_plat, p_tr, p_h1, p_a), "rounds": float(info["rounds"]),
                 "hit_cap": float(info["hit_cap"]), "train_rows": float(info["train_rows"]),
                 "train_es_users": float(info["train_es_users"])}
            mlflow.log_metrics(m)
        prog.step("fusion, HMM, evaluation")
        log.info("%s %s", name, json.dumps(m))
        print(name, {k: round(v, 4) for k, v in m.items() if k.startswith("class_")}, flush=True)


if __name__ == "__main__":
    main()
