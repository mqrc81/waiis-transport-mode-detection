"""Experiment 07: the user label curve of 06's final configuration, one MLflow run per (curve, n, draw).

Per outer fold k: acc model (05's B features, lr 0.1, 47 rounds) and GPS model (gps-v2, lr 0.02, 138 rounds, covered
minutes) trained on the official minutes of the first n users of draw d's ordering of fold k's pool (+ SHL User 1's
days of folds != k on the main curve); product fusion with acc fallback; HMM forward-backward with transitions from
the same users' labelled minutes. Fold k's ExtraSensory users are scored; pooled out of fold over the 5 folds.

    uv run python curve.py --curve main        # n = 0 (SHL User 1 only), 1, 2, 3, 5, 10, 20 x draws 0-4, all
    uv run python curve.py --curve noshl       # n = 1 ... 20 x draws 0-4, all; without SHL User 1
    uv run python curve.py --curve escheck     # n = 2, 10 x draws 0-4: early stopping on users n+1..n+5 (after main)
    uv run python curve.py --smoke             # fold 0 of a few points, no MLflow, no files
Runs already in MLflow (clean, finished) are skipped. Probabilities per run: <data>/processed/07_.../oof/<run>.npz.
"""
from __future__ import annotations

import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")       # single-thread numpy/scipy (latency); LightGBM sets its own threads

import argparse
import json
import logging
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from draws import DRAWS, ES_EXTRA, GPS_V, K, NS, load_draws, train_users
from tmd import fusion, lgbm, minutes
from tmd import gps_features as gf
from tmd.evaluate import CLASS_IDS, FAMILY_IDS, bootstrap_users_recall, family_proba, model_size_bytes, plot_cm, \
    scores
from tmd.labels import CLASSES, FAMILIES, to_family
from tmd.paths import PROCESSED
from tmd.progress import Progress
from tmd.tracking import TRACKING_URI, DataInput, run

HERE = Path(__file__).resolve().parent
EXP = HERE.name
OOF = PROCESSED / EXP / "oof"
ACC_LR, ACC_ROUNDS = 0.1, 47           # median of 05's test_B_seed{0,1,2}: 47, 46, 48
GPS_LR, GPS_ROUNDS = 0.02, 138         # median of 06's test_G_v2_seed{0,1,2}: 139, 136, 138
EARLY = {"acc": (1000, 30), "gps": (10000, 150)}    # 06's early stopping: max rounds, patience
ES_CHECK_NS = (2, 10)
CI_AT = (0, "all")                     # user-bootstrap CIs only here (approved deviation); elsewhere: draw spread
FEATURES = f"acc-v1_whole_sub5s+{GPS_V}_w2min"
log = logging.getLogger()


def points(curve: str) -> list[tuple]:
    """(n, draw) of a curve in run order; n = 0 and n = all are one run each (draw 0)."""
    if curve == "escheck":
        return [(n, d) for n in ES_CHECK_NS for d in DRAWS]
    head = [(0, 0)] if curve == "main" else []
    return head + [(n, d) for n in NS for d in DRAWS] + [("all", 0)]


def run_name(curve, n, d) -> str:
    return f"{curve}_n{n}" if n in (0, "all") else f"{curve}_n{n}_d{d}"


class Data:
    """The dev frame as arrays (ExtraSensory dev + SHL User 1) and the labelled minutes for the HMM."""

    def __init__(self):
        df = fusion.load_frame("dev", GPS_V)
        self.xa = df[fusion.ACC].to_numpy(np.float32)
        self.xg = df[gf.names(GPS_V)].to_numpy(np.float32)
        self.keys = df[["source", "user", "session", "ts", "platform"]].copy()
        self.y, self.fold = df.y.to_numpy(), df.fold.to_numpy()
        self.user = df.user.to_numpy(str)
        self.es = (df.source == "extrasensory").to_numpy()
        self.shl = (df.source == "shl_user1").to_numpy()
        self.off = df.official.to_numpy().astype(bool) & (self.fold >= 0)
        self.cov = df.covered.to_numpy()
        self.iphone = (df.platform == "iphone").to_numpy()
        self.score = self.es & self.off            # scored rows, pooled over folds
        self.is_iphone = dict(zip(df.user[self.es], df.platform[self.es] == "iphone"))
        self.lab = pd.concat([minutes.labelled_minutes(s) for s in ("extrasensory", "shl_user1")], ignore_index=True)
        self.df = df[["source", "user", "session", "ts", "fold"]]


def fit(x, y, tr, lr, rounds, seed, val=None, early=None) -> lgb.Booster:
    """06's LightGBM settings, balanced class weights; fixed rounds, or early stopping on val (rows) if given."""
    prm = {**lgbm.PARAMS, "learning_rate": lr, "seed": seed}
    dtr = lgbm.dataset(x[tr], y[tr], np.ones(int(tr.sum())))
    if val is None:
        return lgb.train(prm, dtr, rounds)
    mr, pat = early
    dva = lgbm.dataset(x[val], y[val], np.ones(int(val.sum())), ref=dtr)
    return lgb.train(prm, dtr, mr, valid_sets=[dva], callbacks=[lgb.early_stopping(pat, verbose=False)])


def point(D: Data, curve: str, n, d: int, draws: dict, prog=None) -> dict:
    """Train per fold, fuse, smooth. Returns raw / p (rows of D, NaN outside the scored folds), per-fold info and
    fold-0 models with HMM parameters (latency)."""
    with_shl = curve != "noshl"
    raw = np.full((len(D.y), 8), np.nan)
    p = raw.copy()
    folds, fold0 = [], None
    for k in range(K):
        u = [] if n == 0 else train_users(draws, d, k, n)
        in_u = np.isin(D.user, u)
        tr = D.off & (D.fold != k) & ((D.es & in_u) | (D.shl & with_shl))
        va = None
        if curve == "escheck":
            vu = draws[str(d)][str(k)][n:n + ES_EXTRA]
            va = D.off & D.es & np.isin(D.user, vu)
        ma = fit(D.xa, D.y, tr, ACC_LR, ACC_ROUNDS, d, va, EARLY["acc"])
        if prog is not None:
            prog.step(f"{run_name(curve, n, d)} fold {k} acc")
        mg = fit(D.xg, D.y, tr & D.cov, GPS_LR, GPS_ROUNDS, d, None if va is None else va & D.cov, EARLY["gps"])
        if prog is not None:
            prog.step(f"{run_name(curve, n, d)} fold {k} gps")
        sc = D.es & (D.fold == k)
        raw[sc] = fusion.product(ma.predict(D.xa[sc]), mg.predict(D.xg[sc]), D.cov[sc])
        lab = D.lab
        hl = lab[((lab.source == "extrasensory") & lab.user.isin(u)) |
                 ((lab.source == "shl_user1") & with_shl & (lab.fold != k) & (lab.fold >= 0))]
        a, pi = minutes.hmm_params(hl)
        p[sc] = minutes.smooth(D.df[sc], raw[sc], a, pi, "fb")
        es_tr = tr & D.es
        mins = np.bincount(D.y[es_tr], minlength=9)[1:]
        info = {"fold": k, "n_users": len(u),
                "iphone_share": float(np.mean([D.is_iphone[x] for x in u])) if u else float("nan"),
                "es_minutes": mins.tolist(),
                "es_gps_minutes": np.bincount(D.y[es_tr & D.cov], minlength=9)[1:].tolist(),
                "shl_minutes": int((tr & D.shl).sum()), "hmm_minutes": len(hl),
                "acc_bytes": model_size_bytes(ma), "gps_bytes": model_size_bytes(mg),
                "acc_rounds": ma.best_iteration or ma.current_iteration(),
                "gps_rounds": mg.best_iteration or mg.current_iteration()}
        if va is not None:
            info["val_minutes"] = int(va.sum())
        folds.append(info)
        log.info("%s fold %s", run_name(curve, n, d), info)
        if k == 0:
            fold0 = (ma, mg, a, pi)
    return {"raw": raw, "p": p, "folds": folds, "fold0": fold0}


def per_class_f1(yy, pr, ids, names) -> dict:
    present = [i for i in ids if (yy == i).any()]
    f = f1_score(yy, pr, labels=present, average=None, zero_division=0)
    return {names[i - 1]: float(v) for i, v in zip(present, f)}


def evaluate(D: Data, res: dict, ci: bool) -> tuple[dict, dict, dict]:
    m, figs, tab = {}, {}, {}
    sel = D.score
    y8, users, fold, cov, iph = D.y[sel], D.user[sel], D.fold[sel], D.cov[sel], D.iphone[sel]
    for stage, pp in (("", res["p"][sel]), ("_raw", res["raw"][sel])):
        for lvl in ("class", "family"):
            ids, names = (CLASS_IDS, CLASSES) if lvl == "class" else (FAMILY_IDS, FAMILIES)
            yy = y8 if lvl == "class" else to_family(y8)
            pr = pp.argmax(1) + 1 if lvl == "class" else family_proba(pp).argmax(1) + 1
            s = scores(yy, pr, lvl)
            pre = f"es_{lvl}"
            m[f"{pre}_f1{stage}"] = s["macro_f1"]
            if stage:
                continue
            for c, r in s["recall"].items():
                if r == r:
                    m[f"{pre}_recall_{c}"] = r
            for c, f in per_class_f1(yy, pr, ids, names).items():
                m[f"{pre}_f1_{c}"] = f
            per = [scores(yy[fold == k], pr[fold == k], lvl)["macro_f1"] for k in range(K)]
            m[f"{pre}_f1_fold_min"], m[f"{pre}_f1_fold_max"] = min(per), max(per)
            for nm, mk in (("iphone", iph), ("android", ~iph), ("covered", cov), ("uncovered", ~cov)):
                m[f"{pre}_f1_{nm}"] = scores(yy[mk], pr[mk], lvl)["macro_f1"]
            tab[f"cm_{lvl}"] = s["cm"].tolist()
            figs[f"cm_es_{lvl}.png"] = plot_cm(s["cm"], lvl, f"es {lvl}")
            if ci:
                st = {}
                for c, (lo, hi) in bootstrap_users_recall(yy, pr, users, lvl, stats=st).items():
                    key = f"{pre}_f1" if c == "macro_f1" else f"{pre}_recall_{c}"
                    m[f"{key}_lo"], m[f"{key}_hi"] = lo, hi
                m[f"{pre}_ci_redraw_rate"] = st["redraw_rate"]
    f = res["folds"]
    mins = np.array([x["es_minutes"] for x in f])
    m["n_users_mean"] = float(np.mean([x["n_users"] for x in f]))
    if f[0]["n_users"]:
        m["iphone_share_mean"] = float(np.mean([x["iphone_share"] for x in f]))
    for j, c in enumerate(CLASSES):
        m[f"train_es_minutes_{c}"] = float(mins[:, j].mean())
        if c != "subway":
            m[f"train_absent_folds_{c}"] = float((mins[:, j] == 0).sum())
    m["train_shl_minutes"] = float(np.mean([x["shl_minutes"] for x in f]))
    for key in ("acc_bytes", "gps_bytes", "acc_rounds", "gps_rounds"):
        v = [x[key] for x in f]
        m[f"{key}_mean"], m[f"{key}_min"], m[f"{key}_max"] = float(np.mean(v)), float(min(v)), float(max(v))
    tab["folds"] = f
    return m, figs, tab


def done_runs() -> set[str]:
    mlflow.set_tracking_uri(TRACKING_URI)
    try:
        r = mlflow.search_runs(experiment_names=[EXP], filter_string="attributes.status = 'FINISHED'")
    except Exception:
        return set()
    if not len(r):
        return set()
    return set(r.loc[r["tags.dirty"] == "false", "tags.mlflow.runName"])


def datasets(with_shl: bool) -> list[DataInput]:
    kw = {"label_rules": "bench-v1", "features": FEATURES, "rate_hz": 25, "window_s": 15, "stride_s": 60,
          "oof_eval": True}
    ds = [DataInput("extrasensory_dev_v2_draws07", "training", **kw)]
    return ds + ([DataInput("shl_user1_dev", "training", **kw)] if with_shl else [])


def smoke(D, draws):
    """Fold 0 of a few points: shapes, absent classes, early stopping. No MLflow, no files."""
    global K
    K = 1
    for curve, n, d in (("main", 0, 0), ("noshl", 1, 0), ("noshl", 1, 3), ("escheck", 2, 0)):
        t = time.time()
        r = point(D, curve, n, d, draws)
        sel = D.score & (D.fold == 0)
        f1 = scores(D.y[sel], r["p"][sel].argmax(1) + 1, "class")["macro_f1"]
        print(curve, n, d, f"fold0 8-class {f1:.3f}", {k: r["folds"][0][k] for k in
              ("n_users", "es_minutes", "acc_rounds", "gps_rounds", "hmm_minutes")}, f"{time.time() - t:.0f}s",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", choices=["main", "noshl", "escheck"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    draws = load_draws()
    if a.smoke:
        smoke(Data(), draws)
        return
    (HERE / "logs").mkdir(exist_ok=True)
    logging.basicConfig(filename=HERE / "logs" / f"curve_{a.curve}.log", filemode="a", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    done = done_runs()
    todo = [(n, d) for n, d in points(a.curve) if run_name(a.curve, n, d) not in done]
    lat_at = {(n, d) for n, d in todo if a.curve == "main" and d == 0}
    prog = Progress(EXP, a.curve, len(todo) * (2 * K + 1) + len(lat_at), log)
    print(f"{a.curve}: {len(todo)} runs to do ({len(points(a.curve)) - len(todo)} done)", flush=True)
    D = Data()
    pool = fusion.session_pool(D.keys.assign(fold=D.fold), 0) if lat_at else None
    OOF.mkdir(parents=True, exist_ok=True)
    with_shl = a.curve != "noshl"
    for n, d in todo:
        t0, name = time.time(), run_name(a.curve, n, d)
        prm = {"model": "lightgbm-acc(05 B) x lightgbm-gps(gps-v2) product + hmm-fb", "curve": a.curve,
               "n_users": str(n), "draw": d, "seed": d, "dataset": "es_dev+shl_u1" if with_shl else "es_dev",
               "rate_hz": 25, "class_set": "8class+family", "primary": "es 8-class macro-F1 (pooled OOF)",
               "features": FEATURES, "gps_features": GPS_V, "fusion": "product, acc fallback", "smoothing": "fb",
               "hmm_transitions": "labelled minutes of the training users (+ SHL User 1 days of folds != k)",
               "acc_lr": ACC_LR, "gps_lr": GPS_LR,
               "acc_rounds": "early stopping" if a.curve == "escheck" else ACC_ROUNDS,
               "gps_rounds": "early stopping" if a.curve == "escheck" else GPS_ROUNDS,
               "early_stopping": f"users n+1..n+{ES_EXTRA}, patience 30 / 150" if a.curve == "escheck" else "none",
               "shl_user1": "days of folds != k" if with_shl else "not used",
               "bootstrap": "users, redraw resamples lacking a class" if n in CI_AT else "none (draw spread)"}
        with run(EXP, prm, run_name=name, datasets=datasets(with_shl)):
            res = point(D, a.curve, n, d, draws, prog)
            m, figs, tab = evaluate(D, res, n in CI_AT)
            if n != 0:
                m["n_users_num"] = m["n_users_mean"]
            sel = D.score
            np.savez_compressed(OOF / f"{name}.npz", p=res["p"][sel].astype(np.float32),
                                raw=res["raw"][sel].astype(np.float32), user=D.user[sel], ts=D.keys.ts.to_numpy()[sel])
            if a.curve == "escheck":
                z = np.load(OOF / f"{run_name('main', n, d)}.npz")
                assert (z["user"] == D.user[sel]).all()
                for lvl in ("class", "family"):
                    yy = D.y[sel] if lvl == "class" else to_family(D.y[sel])
                    pr = z["p"].argmax(1) + 1 if lvl == "class" else family_proba(z["p"]).argmax(1) + 1
                    m[f"es_{lvl}_f1_fixed"] = scores(yy, pr, lvl)["macro_f1"]
                    m[f"es_{lvl}_f1_diff_vs_fixed"] = m[f"es_{lvl}_f1"] - m[f"es_{lvl}_f1_fixed"]
            if (n, d) in lat_at:
                m.update(fusion.latency(pool, *res["fold0"], GPS_V))
                prog.step(f"{name} latency")
            mlflow.log_metrics({k: float(v) for k, v in m.items() if v == v})
            for fn, fig in figs.items():
                mlflow.log_figure(fig, fn)
                plt.close(fig)
            mlflow.log_dict(json.loads(json.dumps(tab, default=lambda o: o.tolist() if hasattr(o, "tolist")
                                                  else str(o))), "tables.json")
        prog.step(f"{name} evaluation, MLflow")
        log.info("%s metrics %s", name, json.dumps(m))
        key = ["es_class_f1", "es_family_f1", "es_class_f1_lo", "es_class_f1_hi", "es_class_recall_bus",
               "es_class_recall_train", "es_class_f1_diff_vs_fixed", "acc_rounds_mean", "gps_rounds_mean"]
        print(name, {k: round(m[k], 3) for k in key if k in m}, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
