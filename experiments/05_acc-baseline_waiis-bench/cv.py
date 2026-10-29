"""One ablation row: paired-fold LightGBM (or HMM smoothing of a saved row), evaluation, latency, MLflow run.

    uv run python cv.py --name A                                   # 116 whole-window features
    uv run python cv.py --name A_esonly --shl-w 0                  # diagnostic: ExtraSensory only
    uv run python cv.py --name B --sub --vs A                      # paired bootstrap against row A
    uv run python cv.py --name F_vit --from-row C --hmm viterbi --vs C
Full output: logs/cv_<name>_seed<k>.log. Prints a short summary.
"""
from __future__ import annotations

import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")       # single-thread numpy/scipy (latency); LightGBM sets its own threads

import argparse
import json
import logging
import time

import context as ctxmod
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import pipeline as pl
from make_features import load_windows

from tmd import acc_features as af
from tmd.evaluate import inference_latency, pipeline_latency
from tmd.tracking import run

MODELS = pl.OUT / "models"
log = logging.getLogger()


def latency(cfg: pl.Cfg, model: lgb.Booster, pool: pd.DataFrame, hmm_method: str | None, hmm_ap: tuple | None,
            seed: int) -> dict:
    """Single window (features + context assembly + model) and pipeline (whole sessions of `pool`, ExtraSensory
    usable minutes) per window. hmm_ap = (transitions, initial distribution) when hmm_method is set."""
    rng = np.random.default_rng(seed)
    nbr = rng.normal(size=(6, len(af.CONTEXT_BASE))).astype(np.float32)
    cb = [af.BASE.index(n) for n in af.CONTEXT_BASE]

    def featurize(w):
        f = af.features(w, cfg.sub, cfg.grav, cfg.lp)
        if cfg.ctx == "none":
            return f
        c, _ = ctxmod.columns(np.vstack([f[:, cb], nbr]), np.arange(1, 7)[None], af.CONTEXT_BASE)
        return np.concatenate([f, c], 1)

    out = inference_latency(lambda x: model.predict(x, num_threads=1), featurize, shape=(1, 375, 3))
    # pipeline: 200 sessions of the pool with >= 3 usable minutes
    es = pool
    n = es.groupby(["user", "session"]).size()
    pick = n[n >= 3].sample(min(200, int((n >= 3).sum())), random_state=seed).index
    keep = es.set_index(["user", "session"]).loc[pick].reset_index()[["user", "ts", "session"]]
    keys, w = load_windows("extrasensory", keep)
    keys = keys.assign(source="extrasensory")
    a, pi = hmm_ap if hmm_method else (None, None)
    groups = [np.flatnonzero(g) for g in (keys.groupby(["user", "session"]).ngroup().to_numpy()[None]
                                          == np.arange(len(pick))[:, None])]

    def run_session(i):
        idx = groups[i]
        f = af.features(w[idx], cfg.sub, cfg.grav, cfg.lp)
        if cfg.ctx != "none":
            c, _ = ctxmod.columns(f[:, cb], ctxmod.neighbours(keys.iloc[idx]), af.CONTEXT_BASE)
            f = np.concatenate([f, c], 1)
        p = model.predict(f, num_threads=1)
        if hmm_method:
            p = pl.smooth(keys.iloc[idx], p, a, pi, hmm_method)
        return p

    out.update(pipeline_latency(run_session, list(range(len(groups))), [len(g) for g in groups]))
    out["lat_pipeline_sessions"] = len(groups)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sub", action="store_true")
    ap.add_argument("--grav", action="store_true")
    ap.add_argument("--ctx", choices=["none", "plain", "masked"], default="none")
    ap.add_argument("--lp", type=float, default=None)
    ap.add_argument("--shl-w", type=float, default=1.0)
    ap.add_argument("--vs", help="row to compare against (paired user bootstrap)")
    ap.add_argument("--from-row", help="HMM row: smooth this saved row instead of training")
    ap.add_argument("--hmm", choices=["forward", "fb", "viterbi"])
    a = ap.parse_args()
    (pl.HERE / "logs").mkdir(exist_ok=True)
    logging.basicConfig(filename=pl.HERE / "logs" / f"cv_{a.name}_seed{a.seed}.log", filemode="w",
                        level=logging.INFO, format="%(asctime)s %(message)s")
    t0 = time.time()
    lab = pd.concat([pl.labelled_minutes(s) for s in ("extrasensory", "shl_user1")], ignore_index=True)
    if a.from_row:
        base_cfg = pl.Cfg(**json.loads(str(np.load(pl.OOF / f"{a.from_row}_seed{a.seed}.npz")["cfg"])))
        cfg = pl.Cfg(**{**base_cfg.__dict__, "name": a.name})
        df = pl.load(cfg)
        p = pl.smooth_cv(df, pl.load_oof(a.from_row, a.seed, df), lab, a.hmm)
        model0 = lgb.Booster(model_file=str(MODELS / f"{a.from_row}_seed{a.seed}_fold0.txt"))
        folds = json.loads((MODELS / f"{a.from_row}_seed{a.seed}_folds.json").read_text())
        n_feat = model0.num_feature()
    else:
        cfg = pl.Cfg(a.name, a.sub, a.grav, a.ctx, a.lp, a.shl_w)
        df = pl.load(cfg)
        log.info("loaded %d rows in %.0fs", len(df), time.time() - t0)
        p, folds, models, names = pl.train_cv(cfg, df, a.seed, log)
        MODELS.mkdir(parents=True, exist_ok=True)
        for k, m in enumerate(models):
            m.save_model(str(MODELS / f"{a.name}_seed{a.seed}_fold{k}.txt"))
        (MODELS / f"{a.name}_seed{a.seed}_folds.json").write_text(json.dumps(folds))
        model0, n_feat = models[0], len(names)
    pl.save_oof(cfg, a.seed, df, p)
    base_p = pl.load_oof(a.vs, a.seed, df) if a.vs else None
    metrics, figs, tab = pl.evaluate(df, p, base_p)
    fr = pd.DataFrame(folds)
    metrics.update({"model_bytes_mean": float(fr.model_bytes.mean()), "n_trees_mean": float(fr.n_trees.mean()),
                    "rounds_mean": float(fr.rounds.mean()), "rounds_min": float(fr.rounds.min()),
                    "rounds_max": float(fr.rounds.max()), "n_features": float(n_feat)})
    pool = df[(df.source == "extrasensory") & (df.fold == 0)]            # fold-0 model, fold-0 sessions
    hmm_ap = pl.hmm_params(lab[(lab.fold != 0) & (lab.fold >= 0)]) if a.hmm else None
    metrics.update(latency(cfg, model0, pool, a.hmm, hmm_ap, a.seed))
    tab["folds"] = folds
    params = {"model": "lightgbm" + (f"+hmm-{a.hmm}" if a.hmm else ""), "row": a.name, "seed": a.seed,
              "dataset": "es_dev+shl_u1" if cfg.shl_w > 0 else "es_dev", "rate_hz": 25,
              "class_set": "8class+family", "features": pl.feature_tag(cfg), "sub": cfg.sub, "grav": cfg.grav,
              "ctx": cfg.ctx, "lowpass_hz": cfg.lp or "none", "shl_weight": cfg.shl_w, "smoothing": a.hmm or "none",
              "from_row": a.from_row or "", "vs": a.vs or "", "window_s": 15, "usable_minutes": "context+hmm",
              "class_weight": "balanced after shl weighting",
              "early_stop": "inner fold (k+1)%5: ES users + SHL days", "max_rounds": pl.MAX_ROUNDS,
              "patience": pl.PATIENCE, "hmm_step_s": pl.STEP_S, "hmm_max_gap_s": pl.MAX_GAP_S,
              "hmm_transitions": "all labelled training minutes, before caps and drops",
              "es_family_classes": tab["es_family_classes"], "es_class_classes": tab["es_class_classes"],
              "shl_eval": "same-person day folds", "bootstrap": "users, redraw resamples lacking a class",
              **{f"lgb_{k}": v for k, v in pl.PARAMS.items()}}
    with run(pl.EXP, params, run_name=f"{a.name}_seed{a.seed}", datasets=pl.datasets(pl.feature_tag(cfg))):
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})
        for fn, fig in figs.items():
            mlflow.log_figure(fig, fn)
            plt.close(fig)
        mlflow.log_dict(tab, "tables.json")
    log.info("metrics %s", json.dumps(metrics))
    key = ["es_family_f1", "es_family_f1_lo", "es_family_f1_hi", "es_class_f1", "shl_family_f1", "shl_class_f1",
           "es_family_f1_diff", "es_family_f1_diff_lo", "es_family_f1_diff_hi", "shl_family_f1_diff",
           "shl_class_recall_train", "shl_class_recall_subway", "rounds_mean", "lat_inference_ms_p50",
           "lat_pipeline_ms_p50"]
    print(a.name, {k: round(metrics[k], 4) for k in key if k in metrics}, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
