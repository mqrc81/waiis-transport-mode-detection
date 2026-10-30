"""One row of the 06 ablation: train (or fuse saved rows), HMM forward-backward, evaluate, latency, MLflow run.

    uv run python run.py --row A                                     # 05's B + fb, rescored (no training)
    uv run python run.py --row A2 --lr 0.02 --vs A                   # A' (lr / 5)
    uv run python run.py --row G --lr 0.1 --acc-row A                # GPS only, covered minutes (descriptive)
    uv run python run.py --row G_esonly --lr 0.1 --no-shl --acc-row A
    uv run python run.py --row H1 --acc-row A --gps-row G --vs A     # product fusion
    uv run python run.py --row H2 --acc-row A --gps-row G --vs A        # inner fits use each row's own lr
    uv run python run.py --row I --lr 0.1 --vs A                     # early fusion
Rows named H1_<gps-row> etc. when run with another GPS model (--name).
"""
from __future__ import annotations

import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")       # single-thread numpy/scipy (latency); LightGBM sets its own threads

import argparse
import json
import logging
import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pipe

from tmd import acc_features as af
from tmd import gps_features as gf
from tmd import lgbm, minutes
from tmd.evaluate import scores
from tmd.progress import Progress
from tmd.tracking import run

MODELS = pipe.OUT / "models"
P05 = pipe.P05
log = logging.getLogger()
ACC_F, GPS_F = "acc-v1_whole_sub5s", "gps-v1_w2min"
FEATURES = {"A": ACC_F, "A2": ACC_F, "I": f"{ACC_F}+{GPS_F}", "G": GPS_F, "G_esonly": GPS_F,
            "H1": f"{ACC_F}+{GPS_F}", "H2": f"{ACC_F}+{GPS_F}"}


def save_models(name, seed, res):
    MODELS.mkdir(parents=True, exist_ok=True)
    for k, m in enumerate(res["models"]):
        m.save_model(str(MODELS / f"{name}_seed{seed}_fold{k}.txt"))
    (MODELS / f"{name}_seed{seed}_folds.json").write_text(json.dumps(res["rows"]))
    inner = {str(k): {"idx": v[0].tolist(), "p": v[1].tolist()} for k, v in res["inner"].items()}
    np.savez_compressed(MODELS / f"{name}_seed{seed}_inner.npz",
                        **{f"idx{k}": np.array(v["idx"]) for k, v in inner.items()},
                        **{f"p{k}": np.array(v["p"], np.float32) for k, v in inner.items()})


def model_files(name, seed):
    if name == "A":
        return P05 / "models", "B"
    return MODELS, name


def fold_models(name, seed):
    d, n = model_files(name, seed)
    return [lgb.Booster(model_file=str(d / f"{n}_seed{seed}_fold{k}.txt")) for k in range(lgbm.K)], \
        json.loads((d / f"{n}_seed{seed}_folds.json").read_text())


def row_lr(name, seed) -> float:
    """The learning rate a saved row was trained with (A = 05's B: 0.1)."""
    if name == "A":
        return 0.1
    return json.loads(str(np.load(pipe.OOF / f"{name}_seed{seed}.npz", allow_pickle=True)["info"]))["lr"]


def saved_inner(name, seed):
    d, n = model_files(name, seed)
    f = d / f"{n}_seed{seed}_inner.npz"
    return np.load(f) if f.exists() else None


def inner_fits(name, seed) -> int:
    """New LightGBM fits inner_oof makes for a row: 4 per outer fold, 3 if the early-stopping fit was saved."""
    return lgbm.K * (lgbm.K - 1 - (saved_inner(name, seed) is not None))


def inner_oof(df, cols, name, seed, lr, rows_mask, shl, progress=None):
    """H2: inner out-of-fold probabilities per outer fold k over the training folds T = {0..4} \\ {k}.
    j = (k+1) % 5 reuses the outer early-stopping fit; the 3 other j are new fits with the outer fold's rounds."""
    _, folds = fold_models(name, seed)
    z = saved_inner(name, seed)
    x = df[cols].to_numpy(np.float32)
    is_shl = (df.source != "extrasensory").to_numpy()
    sw = np.where(is_shl, 1.0 if shl else 0.0, 1.0)
    fold, y = df.fold.to_numpy(), df.y.to_numpy()
    trainable = df.official.to_numpy() & rows_mask & (sw > 0) & (fold >= 0)
    prm, _, _ = pipe.params(lr)
    out = {}
    for k in range(lgbm.K):
        p = np.full((len(df), 8), np.nan, np.float32)
        rounds = folds[k]["rounds"]
        for j in [j for j in range(lgbm.K) if j != k]:
            idx = np.flatnonzero(fold == j)
            if j == (k + 1) % lgbm.K and z is not None:
                assert (z[f"idx{k}"] == idx).all()
                p[idx] = z[f"p{k}"]
                continue
            tr = trainable & (fold != k) & (fold != j)
            m = lgb.train({**prm, "seed": seed}, lgbm.dataset(x[tr], y[tr], sw[tr]), rounds)
            p[idx] = m.predict(x[idx])
            if progress is not None:
                progress.step(f"inner {name} k={k} j={j}")
        out[k] = p
        print(f"inner {name} k={k} done", flush=True)
    return out


def n_units(a) -> int:
    """Progress units of a row: its LightGBM fits (paired CV: 2 per fold; H2: the new inner fits), + 1 for
    evaluation, latency and logging."""
    if a.row in ("A2", "I", "G", "G_esonly"):
        fits = 2 * lgbm.K
    elif a.row == "H2":
        fits = inner_fits(a.acc_row, a.seed) + inner_fits(a.gps_row, a.seed)
    else:
        fits = 0
    return fits + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True, choices=["A", "A2", "G", "G_esonly", "H1", "H2", "I"])
    ap.add_argument("--name", help="row name if not --row (e.g. H1_esgps)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--acc-row", default="A")
    ap.add_argument("--gps-row", default="G")
    ap.add_argument("--no-shl", action="store_true", help="G: train without SHL User 1 rows")
    ap.add_argument("--vs")
    a = ap.parse_args()
    name = a.name or a.row
    (pipe.HERE / "logs").mkdir(exist_ok=True)
    logging.basicConfig(filename=pipe.HERE / "logs" / f"run_{name}_seed{a.seed}.log", filemode="w",
                        level=logging.INFO, format="%(asctime)s %(message)s")
    t0 = time.time()
    prm = {"model": f"lightgbm+{a.row}+hmm-fb", "row": name, "seed": a.seed, "rate_hz": 25,
           "dataset": "es_dev+shl_u1", "class_set": "8class+family", "primary": "es 8-class macro-F1",
           "features": FEATURES[a.row], "lr": a.lr, "acc_row": a.acc_row if a.row not in ("A", "A2", "I") else "",
           "gps_row": a.gps_row if a.row in ("H1", "H2") else "", "gps_train_shl": not a.no_shl,
           "fusion": {"H1": "product", "H2": "stacking LR (inner OOF)", "I": "early"}.get(a.row, "none"),
           "smoothing": "fb", "vs": a.vs or "", "scored": "covered minutes" if a.row in ("G", "G_esonly") else "all",
           "bootstrap": "users, redraw resamples lacking a class", "gps": "gps_per_minute, derived speed only",
           "usable_minutes": "context+hmm", **{f"lgb_{k}": v for k, v in pipe.params(a.lr)[0].items()}}
    prog = Progress(pipe.EXP, f"{name}_seed{a.seed}", n_units(a), log)
    with run(pipe.EXP, prm, run_name=f"{name}_seed{a.seed}", datasets=pipe.datasets(FEATURES[a.row])):
        m, figs, tab = row(a, name, prog)
        mlflow.log_metrics({k: float(v) for k, v in m.items() if v == v})
        for fn, fig in figs.items():
            mlflow.log_figure(fig, fn)
            plt.close(fig)
        mlflow.log_dict(pipe.dump(tab), "tables.json")
    log.info("metrics %s", json.dumps(m))
    prog.step("evaluation, latency, MLflow")
    key = ["es_class_f1", "es_class_f1_lo", "es_class_f1_hi", "es_family_f1", "es_class_f1_diff",
           "es_class_f1_diff_lo", "es_class_f1_diff_hi", "shl_class_f1", "shl_class_f1_diff", "es_class_f1_covered",
           "es_class_f1_uncovered", "acc_row_on_covered_es_class_f1", "rounds_mean", "hit_cap_folds",
           "lat_inference_ms_p50", "lat_pipeline_ms_p50"]
    print(name, {k: round(m[k], 4) for k in key if k in m}, f"{time.time() - t0:.0f}s", flush=True)


def row(a, name, prog):
    """Train or fuse the row (fold metrics go to the active MLflow run as each fold finishes), smooth, evaluate,
    latency."""
    df = pipe.load()
    lab = pipe.labelled()
    cov = df.covered.to_numpy()
    allrows = np.ones(len(df), bool)
    score_mask, extra_m, info = None, {}, {"features": FEATURES[a.row]}
    acc_models = gps_models = stacker0 = None
    es_off, fold_all, y_all = ((df.source == "extrasensory") & df.official).to_numpy(), df.fold.to_numpy(), \
        df.y.to_numpy()

    def log_fold(k, raw_p, mask=None, **extra):
        """Fold k as soon as it finishes: ES 8-class macro-F1 before the HMM, plus the fold's own numbers."""
        sel = es_off & (fold_all == k) & (True if mask is None else mask)
        f1 = scores(y_all[sel], raw_p[sel].argmax(1) + 1, "class")["macro_f1"]
        mlflow.log_metrics({f"fold{k}_es_class_f1_raw": f1, **{f"fold{k}_{n}": float(v) for n, v in extra.items()}})

    def on_fold(r, raw_p, mask=None):
        log_fold(r["fold"], raw_p, mask, **{n: r[n] for n in ("rounds", "n_trees", "model_bytes", "hit_cap",
                                                               "train_rows")})
    if a.row == "A":                                    # 05's B (seed) + fb, no training
        raw = np.load(P05 / "oof" / f"B_seed{a.seed}.npz")["p"]
        p = np.load(P05 / "oof" / f"F_fb_seed{a.seed}.npz")["p"]
        acc_models, folds = fold_models("A", a.seed)
    elif a.row in ("A2", "I"):
        cols = pipe.ACC if a.row == "A2" else pipe.ACC + pipe.GPS
        res, _ = pipe.train(df, cols, a.seed, a.lr, allrows, True, log, name, progress=prog, on_fold=on_fold)
        save_models(name, a.seed, res)
        raw, folds, acc_models = res["p"], res["rows"], res["models"]
        p = minutes.smooth_cv(df, raw, lab, "fb")
    elif a.row in ("G", "G_esonly"):
        res, _ = pipe.train(df, pipe.GPS, a.seed, a.lr, cov, not a.no_shl, log, name, progress=prog,
                            on_fold=lambda r, raw_p: on_fold(r, raw_p, cov))
        save_models(name, a.seed, res)
        raw = res["p"].copy()
        raw[~cov] = 1 / 8                               # no GPS: uniform emission, the HMM bridges
        folds, gps_models = res["rows"], res["models"]
        p = minutes.smooth_cv(df, raw, lab, "fb")
        score_mask = cov
        acc_p = pipe.load_probs(a.acc_row, a.seed, df)
        acc_m, _, _ = pipe.evaluate(df, acc_p, score_mask=cov)
        extra_m = {"acc_row_on_covered_es_class_f1": acc_m["es_class_f1"],
                   "acc_row_on_covered_es_family_f1": acc_m["es_family_f1"],
                   "acc_row_on_covered_shl_class_f1": acc_m["shl_class_f1"]}
    else:                                               # H1 / H2: fuse saved rows
        raw_acc = pipe.load_probs(a.acc_row, a.seed, df, "raw")
        raw_gps = pipe.load_probs(a.gps_row, a.seed, df, "raw")
        acc_models, folds = fold_models(a.acc_row, a.seed)
        gps_models, gfolds = fold_models(a.gps_row, a.seed)
        if a.row == "H1":
            raw = pipe.product(raw_acc, raw_gps, cov)
        else:
            gps_shl = json.loads(str(np.load(pipe.OOF / f"{a.gps_row}_seed{a.seed}.npz",
                                             allow_pickle=True)["info"]))["gps_shl"]
            in_acc = inner_oof(df, pipe.ACC, a.acc_row, a.seed, row_lr(a.acc_row, a.seed), allrows, True, prog)
            in_gps = inner_oof(df, pipe.GPS, a.gps_row, a.seed, row_lr(a.gps_row, a.seed), cov, gps_shl, prog)
            raw = raw_acc.astype(np.float64).copy()
            fold, off = df.fold.to_numpy(), df.official.to_numpy()
            coefs = []
            for k in range(lgbm.K):
                tr = off & cov & (fold != k) & (fold >= 0)
                st = pipe.fit_stacker(pipe.stack_features(in_acc[k][tr], in_gps[k][tr]), df.y.to_numpy()[tr])
                sc = (fold == k) & cov
                raw[sc] = pipe.stacker_proba(st, pipe.stack_features(raw_acc[sc], raw_gps[sc]))
                coefs.append(st.coef_.size + st.intercept_.size)
                log_fold(k, raw, cov, stacker_params=coefs[-1])
                if k == 0:
                    stacker0 = st
            extra_m["stacker_params"] = float(np.mean(coefs))
        p = minutes.smooth_cv(df, raw, lab, "fb")
    info.update({"row": a.row, "acc_row": a.acc_row, "gps_row": a.gps_row, "gps_shl": not a.no_shl, "lr": a.lr})
    pipe.save(name, a.seed, df, raw, p, {"info": json.dumps(info)})
    base = pipe.load_probs(a.vs, a.seed, df) if a.vs else None
    m, figs, tab = pipe.evaluate(df, p, base, score_mask)
    m.update(extra_m)
    m.update({"rounds_mean": float(np.mean([f["rounds"] for f in folds])),
              "rounds_min": float(min(f["rounds"] for f in folds)), "rounds_max": float(max(f["rounds"] for f in folds)),
              "hit_cap_folds": float(sum(f.get("hit_cap", False) for f in folds))})
    if acc_models:
        m["acc_model_bytes_mean"] = pipe.model_bytes(acc_models)
    if gps_models:
        m["gps_model_bytes_mean"] = pipe.model_bytes(gps_models)

    # latency, fold-0 models and HMM parameters
    a0, pi0 = minutes.hmm_params(lab[(lab.fold != 0) & (lab.fold >= 0)])
    am = acc_models[0] if acc_models else None
    gm = gps_models[0] if gps_models else None
    early = a.row == "I"

    def fuse(pa, pg, c):
        if a.row == "H1":
            return pipe.product(pa, pg, c)
        if a.row == "H2":
            out = pa.astype(np.float64).copy()
            if c.any():
                out[c] = pipe.stacker_proba(stacker0, pipe.stack_features(pa[c], pg[c]))
            return out
        return pa if pg is None else pg

    def predict_session(keys, w, gps):
        fa = af.features(w, True, False) if am is not None else None
        fg = gf.features(keys, gps) if (gm is not None or early) else None
        if early:
            pa = am.predict(np.concatenate([fa, fg], 1), num_threads=1)
            return minutes.smooth(keys, pa, a0, pi0, "fb")
        c = gf.covered(fg) if fg is not None else np.zeros(len(keys), bool)
        pa = am.predict(fa, num_threads=1) if am is not None else None
        pg = gm.predict(fg, num_threads=1) if gm is not None else None
        if pa is None:
            pg[~c] = 1 / 8
        return minutes.smooth(keys, fuse(pa, pg, c), a0, pi0, "fb")

    def predict_one(w, gps, t):
        fa = af.features(w, True, False) if am is not None else None
        fg = None
        if gm is not None or early:
            g = gf.prepare(gps)
            fg = gf.window_features(t, *[g[c].to_numpy() for c in ("ts", "time", "pair_v", "pair_dist", "brg",
                                                                   "lat", "lon")])[None].astype(np.float32)
        if early:
            return am.predict(np.concatenate([fa, fg], 1), num_threads=1)
        pa = am.predict(fa, num_threads=1) if am is not None else None
        pg = gm.predict(fg, num_threads=1) if gm is not None else None
        c = gf.covered(fg) if fg is not None else np.zeros(1, bool)
        return fuse(pa, pg, c) if pa is not None else pg

    m.update(pipe.latency(pipe.session_pool(df, a.seed), predict_session, predict_one))
    tab["folds"] = folds
    return m, figs, tab

if __name__ == "__main__":
    main()
