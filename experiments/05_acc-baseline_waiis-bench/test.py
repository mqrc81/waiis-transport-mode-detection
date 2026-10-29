"""Final test (stop-point 3, scored once): B + HMM forward-backward, seeds 0-2.

Per seed: LightGBM on all official ExtraSensory dev + SHL User 1 examples, rounds = median of that seed's 5
early-stopping counts (B_seed<k>_folds.json); HMM transitions and initial distribution from all labelled dev minutes.
Scored on the 12 ExtraSensory test users (overall, iPhone, Android), then the SHL Users 2-3 rail check, then the
motorbike check set (descriptive). One MLflow run per seed: test_seed<k>.

    uv run python make_features.py --part test --sources extrasensory shl_user23
    uv run python test.py
"""
from __future__ import annotations

import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")

import argparse
import json
import logging

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import pipeline as pl
from cv import MODELS, latency

from tmd import bench
from tmd.evaluate import bootstrap_users_recall, family_proba, model_size_bytes, plot_cm, scores
from tmd.labels import CLASSES, to_family
from tmd.tracking import TRACKING_URI, DataInput, run

SEEDS = (0, 1, 2)
CFG = pl.Cfg("final", sub=True)
log = logging.getLogger()


def preds(p):
    return {"family": family_proba(p).argmax(1) + 1, "class": p.argmax(1) + 1}


def block(m, figs, tab, pre, y8, users, p, ci: bool):
    """F1, per-class recall (with user-bootstrap CIs if ci), confusion matrices, class lists."""
    for lvl, pr in preds(p).items():
        yy = to_family(y8) if lvl == "family" else y8
        s = scores(yy, pr, lvl)
        m[f"{pre}_{lvl}_f1"] = s["macro_f1"]
        tab[f"{pre}_{lvl}_classes"] = pl.class_list(yy, lvl)
        tab[f"cm_{pre}_{lvl}"] = s["cm"].tolist()
        for c, r in s["recall"].items():
            if r == r:
                m[f"{pre}_{lvl}_recall_{c}"] = r
        if ci:
            st = {}
            for c, (lo, hi) in bootstrap_users_recall(yy, pr, users, lvl, stats=st).items():
                key = f"{pre}_{lvl}_f1" if c == "macro_f1" else f"{pre}_{lvl}_recall_{c}"
                m[f"{key}_lo"], m[f"{key}_hi"] = lo, hi
            m[f"{pre}_{lvl}_ci_redraw_rate"] = st["redraw_rate"]
        if not pre.endswith("_raw"):
            figs[f"cm_{pre}_{lvl}.png"] = plot_cm(s["cm"], lvl, f"{pre} {lvl}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="score again although test runs exist")
    a = ap.parse_args()
    logging.basicConfig(filename=pl.HERE / "logs" / "test.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    mlflow.set_tracking_uri(TRACKING_URI)
    done = mlflow.search_runs(experiment_names=[pl.EXP], filter_string="attributes.run_name LIKE 'test_seed%'")
    if len(done) and not a.force:
        raise SystemExit(f"test already scored ({len(done)} runs); the test is scored once")
    dev, te = pl.load(CFG, "dev"), pl.load(CFG, "test")
    lab = pd.concat([pl.labelled_minutes(s) for s in ("extrasensory", "shl_user1")], ignore_index=True)
    hmm_ap = pl.hmm_params(lab)                                       # all labelled dev minutes
    names = CFG.feature_names()
    tr = dev.official.to_numpy() & (dev.fold.to_numpy() >= 0)
    x_tr, y_tr = dev.loc[tr, names].to_numpy(np.float32), dev.y.to_numpy()[tr]
    x_te = te[names].to_numpy(np.float32)
    es = ((te.source == "extrasensory") & te.official).to_numpy()
    shl = ((te.source == "shl_user23") & te.official).to_numpy()
    mb_keys = bench.read("extrasensory", "labels", columns=["user", "ts", "reason"])
    mb_keys = mb_keys[mb_keys.reason == "check:MOTORBIKE"]
    mb = te.set_index(["user", "ts"]).index.isin(mb_keys.set_index(["user", "ts"]).index) \
        & (te.source == "extrasensory").to_numpy()
    summary = {}
    for seed in SEEDS:
        folds = json.loads((MODELS / f"B_seed{seed}_folds.json").read_text())
        rounds = int(np.median([f["rounds"] for f in folds]))
        model = lgb.train({**pl.PARAMS, "seed": seed},
                          lgb.Dataset(x_tr, y_tr - 1, weight=pl.class_weights(y_tr, np.ones(len(y_tr)))), rounds)
        model.save_model(str(MODELS / f"test_B_seed{seed}.txt"))
        p_raw = model.predict(x_te)
        p = pl.smooth(te, p_raw, *hmm_ap, "fb")
        m, figs, tab = {}, {}, {}
        for scope, sel in (("all", es), ("iphone", es & (te.platform == "iphone").to_numpy()),
                           ("android", es & (te.platform == "android").to_numpy())):
            d = te[sel]
            block(m, figs, tab, f"test_es_{scope}", d.y.to_numpy(), d.user.to_numpy(), p[sel], ci=True)
            block(m, figs, tab, f"test_es_{scope}_raw", d.y.to_numpy(), d.user.to_numpy(), p_raw[sel], ci=True)
        d = te[shl]
        block(m, figs, tab, "test_shl23", d.y.to_numpy(), d.user.to_numpy(), p[shl], ci=False)
        block(m, figs, tab, "test_shl23_raw", d.y.to_numpy(), d.user.to_numpy(), p_raw[shl], ci=False)
        tab["motorbike"] = {"n": int(mb.sum()), "users": sorted({u[:8] for u in te.user[mb]}),
                            "pred_after_hmm": dict(zip(CLASSES, np.bincount(p[mb].argmax(1), minlength=8).tolist())),
                            "pred_before_hmm": dict(zip(CLASSES, np.bincount(p_raw[mb].argmax(1),
                                                                             minlength=8).tolist()))}
        pool = te[te.source == "extrasensory"]
        m.update(latency(CFG, model, pool, "fb", hmm_ap, seed))
        m.update({"model_bytes": model_size_bytes(model), "rounds": rounds, "n_trees": model.num_trees(),
                  "n_features": len(names)})
        params = {"model": "lightgbm+hmm-fb", "row": "test", "seed": seed, "dataset": "es_dev+shl_u1",
                  "rate_hz": 25, "class_set": "8class+family", "features": pl.feature_tag(CFG), "sub": True,
                  "grav": False, "ctx": "none", "lowpass_hz": "none", "shl_weight": 1.0, "smoothing": "fb",
                  "rounds_rule": "median of the seed's 5 CV early-stopping counts",
                  "hmm_transitions": "all labelled dev minutes, before caps and drops",
                  "bootstrap": "users, redraw resamples lacking a class", "window_s": 15,
                  "test_es_family_classes": tab["test_es_all_family_classes"],
                  "test_es_iphone_class_classes": tab["test_es_iphone_class_classes"],
                  **{f"lgb_{k}": v for k, v in pl.PARAMS.items()}}
        kw = {"label_rules": "bench-v1", "features": pl.feature_tag(CFG), "rate_hz": 25, "window_s": 15,
              "stride_s": 60}
        ds = [DataInput("extrasensory_dev_v2", "training", **kw), DataInput("shl_user1_dev", "training", **kw),
              DataInput("extrasensory_test_v2", "evaluation", **kw),
              DataInput("shl_user23_test_v2", "evaluation", **kw)]
        with run(pl.EXP, params, run_name=f"test_seed{seed}", datasets=ds):
            mlflow.log_metrics({k: float(v) for k, v in m.items() if v == v})
            for fn, fig in figs.items():
                mlflow.log_figure(fig, fn)
                plt.close(fig)
            mlflow.log_dict(tab, "tables.json")
        log.info("seed %d %s %s", seed, json.dumps(m), json.dumps(tab["motorbike"]))
        summary[seed] = m
        print(seed, {k: round(m[k], 4) for k in ("test_es_all_family_f1", "test_es_all_class_f1",
                                                 "test_es_iphone_family_f1", "test_es_android_family_f1",
                                                 "test_shl23_family_f1", "rounds")}, flush=True)
    for k in ("test_es_all_family_f1", "test_es_all_class_f1", "test_shl23_family_f1"):
        print(k, "mean", round(float(np.mean([summary[s][k] for s in SEEDS])), 4))


if __name__ == "__main__":
    main()
