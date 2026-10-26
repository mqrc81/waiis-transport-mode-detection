"""Step 2: R1 (SHL only) LightGBM on features v1, cross-validated over the 5 User 1 day folds.

Usage: uv run python r1_cv.py [--seed 0]
For each fold k: inner early stopping on fold (k+1) % 5 (training folds only) picks the number of rounds, then
the model is refit on all 4 training folds (pure windows). Evaluated on:
  - SHL fold k, all windows with a non-Null majority label (mixed ones included), per position;
  - ExtraSensory dev users (cross-dataset: never trained on), mean of the 5 fold models' probabilities.
Logs one MLflow run; predictions go to <data>/processed/02_family-baseline_shl-es/r1_<tag>/.
"""
import argparse
import json
import logging
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from tmd.evaluate import bootstrap_users, family_proba, inference_latency, model_size_bytes, plot_cm, scores
from tmd.features import FEATURES_VERSION, NAMES, window_features
from tmd.labels import CLASSES, FAMILIES, to_family
from tmd.paths import PROCESSED
from tmd.tracking import run

HERE = Path(__file__).resolve().parent
EXP = HERE.name
TAG = f"25hz_w5s_s2.5s_feat-{FEATURES_VERSION}"
K = 5
PARAMS = dict(objective="multiclass", num_class=8, learning_rate=0.1, num_leaves=31, min_child_samples=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1)
MAX_ROUNDS, PATIENCE = 1000, 30
log = logging.getLogger()


def class_weights(y):
    c = np.bincount(y, minlength=9)[1:].astype(float)
    w = np.where(c > 0, c.sum() / (8 * np.maximum(c, 1)), 0.0)
    return w[y - 1]


def fit(train: pd.DataFrame, val: pd.DataFrame | None, rounds: int | None, seed: int) -> lgb.Booster:
    p = {**PARAMS, "seed": seed}
    dtr = lgb.Dataset(train[NAMES].to_numpy(), train.y.to_numpy() - 1, weight=class_weights(train.y.to_numpy()),
                      free_raw_data=True)
    if val is not None:
        dva = lgb.Dataset(val[NAMES].to_numpy(), val.y.to_numpy() - 1, weight=class_weights(val.y.to_numpy()),
                          reference=dtr)
        return lgb.train(p, dtr, MAX_ROUNDS, valid_sets=[dva],
                         callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
    return lgb.train(p, dtr, rounds)


def pred(p8):
    return p8.argmax(1) + 1, family_proba(p8).argmax(1) + 1


def flat(prefix: str, s: dict, level: str) -> dict:
    names = CLASSES if level == "class" else FAMILIES
    out = {f"{prefix}_macro_f1": s["macro_f1"], f"{prefix}_acc": s["acc"], f"{prefix}_n": s["n"]}
    out.update({f"{prefix}_recall_{n}": s["recall"][n] for n in names if not np.isnan(s["recall"][n])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(filename=HERE / "logs" / "r1_cv.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    out_dir = PROCESSED / EXP / f"r1_{TAG}_seed{a.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    shl = pd.read_parquet(PROCESSED / f"shl_u1_{TAG}.parquet")
    es = pd.read_parquet(PROCESSED / f"es_dev_{TAG}.parquet")
    log.info("SHL %d windows, ES %d windows", len(shl), len(es))
    oof = np.zeros((len(shl), 8), np.float32)
    es_p = np.zeros((len(es), 8), np.float32)
    fold_rows, models = [], []
    for k in range(K):
        tr = shl[(shl.fold != k) & shl.pure]
        inner = (k + 1) % K
        m0 = fit(tr[tr.fold != inner], tr[tr.fold == inner], None, a.seed)
        rounds = m0.best_iteration
        m = fit(tr, None, rounds, a.seed)
        models.append(m)
        te = (shl.fold == k).to_numpy()
        oof[te] = m.predict(shl.loc[te, NAMES].to_numpy())
        pk = m.predict(es[NAMES].to_numpy()).astype(np.float32)
        es_p += pk / K
        yk = shl.y.to_numpy()[te]
        c8, cf = pred(oof[te])
        ok = (es.reason == "ok").to_numpy()
        e8, ef = pred(pk[ok])
        ye = es.y.to_numpy()[ok]
        row = {"fold": k, "rounds": rounds, "n_trees": m.num_trees(), "model_bytes": model_size_bytes(m),
               "shl_class_f1": scores(yk, c8, "class")["macro_f1"],
               "shl_family_f1": scores(to_family(yk), cf, "family")["macro_f1"],
               "es_class_f1": scores(ye, e8, "class")["macro_f1"],
               "es_family_f1": scores(to_family(ye), ef, "family")["macro_f1"]}
        fold_rows.append(row)
        log.info("fold %s", row)
        print({k2: round(v, 3) if isinstance(v, float) else v for k2, v in row.items()})
        m.save_model(str(out_dir / f"model_fold{k}.txt"))
    folds = pd.DataFrame(fold_rows)

    # Pooled out-of-fold SHL and fold-ensemble ExtraSensory predictions.
    pd.DataFrame(oof, columns=CLASSES).assign(idx=np.arange(len(shl))).to_parquet(out_dir / "shl_oof_proba.parquet")
    pd.DataFrame(es_p, columns=CLASSES).to_parquet(out_dir / "es_dev_proba.parquet")
    metrics, tables, figs = {}, {}, {}
    ys, (s8, sf) = shl.y.to_numpy(), pred(oof)
    for name, sel in [("shl", np.ones(len(shl), bool)), ("shl_pure", shl.pure.to_numpy())] + \
                     [(f"shl_{p.lower()}", (shl.position == p).to_numpy()) for p in ["Hips", "Hand", "Torso", "Bag"]] + \
                     [(f"shl_4phones_hips", ((shl.position == "Hips") & shl.rec.isin(shl[shl.position == "Hand"].rec.unique())).to_numpy())]:
        for lvl, yy, pp in [("class", ys[sel], s8[sel]), ("family", to_family(ys[sel]), sf[sel])]:
            s = scores(yy, pp, lvl)
            metrics.update(flat(f"{name}_{lvl}", s, lvl))
            tables[f"{name}_{lvl}_cm"] = s["cm"].tolist()
            if name == "shl":
                figs[f"cm_shl_{lvl}.png"] = plot_cm(s["cm"], lvl, f"R1 SHL User 1, out-of-fold, {lvl}")
    for c in ["shl_class_f1", "shl_family_f1", "es_class_f1", "es_family_f1"]:
        metrics[f"folds_{c}_mean"], metrics[f"folds_{c}_std"] = folds[c].mean(), folds[c].std()

    ok = (es.reason == "ok").to_numpy()
    e8, ef = pred(es_p)
    ye, users = es.y.to_numpy(), es.user.to_numpy()
    groups = [("es", ok)] + [(f"es_{p}", ok & (es.position == p).to_numpy()) for p in es.position.unique()] + \
             [(f"es_{p}", ok & (es.platform == p).to_numpy()) for p in ["iphone", "android"]]
    for name, sel in groups:
        for lvl, yy, pp in [("class", ye[sel], e8[sel]), ("family", to_family(ye[sel]), ef[sel])]:
            s = scores(yy, pp, lvl)
            metrics.update(flat(f"{name}_{lvl}", s, lvl))
            tables[f"{name}_{lvl}_cm"] = s["cm"].tolist()
            if name == "es":
                figs[f"cm_es_{lvl}.png"] = plot_cm(s["cm"], lvl, f"R1 on ExtraSensory dev users (cross-dataset), {lvl}")
                lo, hi = bootstrap_users(yy, pp, users[sel], lvl)
                metrics[f"es_{lvl}_macro_f1_ci_lo"], metrics[f"es_{lvl}_macro_f1_ci_hi"] = lo, hi
    fam_e = to_family(ye)
    for src, fid in [("road", 4), ("still", 1)]:
        sel = ok & (fam_e == fid)
        metrics[f"es_rail_fp_rate_on_{src}"] = float((ef[sel] == 5).mean())
    metrics["es_train_windows"] = int((ok & (ye == 7)).sum())
    moto = (es.reason == "check:MOTORBIKE").to_numpy()
    tables["es_motorbike_pred_class"] = pd.Series(e8[moto]).map(dict(enumerate(CLASSES, 1))).value_counts().to_dict()
    tables["es_motorbike_pred_family"] = pd.Series(ef[moto]).map(dict(enumerate(FAMILIES, 1))).value_counts().to_dict()
    tables["folds"] = fold_rows

    # Size and latency (single thread, after warm-up): features for one window, model for one window.
    sizes = folds.model_bytes
    metrics.update({"model_bytes_mean": float(sizes.mean()), "n_trees_mean": float(folds.n_trees.mean())})
    metrics.update(inference_latency(lambda x: models[0].predict(x, num_threads=1), window_features))

    params = {"model": "lightgbm", "dataset": "shl_u1", "eval": "shl_u1_dayfolds+es_dev", "setup": "R1",
              "rate_hz": 25, "class_set": "8class+family", "seed": a.seed, "features": FEATURES_VERSION,
              "n_features": len(NAMES), "window_s": 5, "stride_s": 2.5, "train_windows": "pure",
              "class_weight": "balanced", "early_stop": "inner day fold", **{f"lgb_{k}": v for k, v in PARAMS.items()},
              "max_rounds": MAX_ROUNDS, "patience": PATIENCE, "context": "none", "smoothing": "none"}
    with run(EXP, params, run_name=f"R1_lgbm_feat-{FEATURES_VERSION}_seed{a.seed}"):
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})
        for fn, fig in figs.items():
            mlflow.log_figure(fig, fn)
        mlflow.log_dict(tables, "tables.json")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (out_dir / "tables.json").write_text(json.dumps(tables, indent=1, default=int))

    keys = ["shl_class_macro_f1", "shl_family_macro_f1", "folds_shl_class_f1_std", "es_class_macro_f1",
            "es_family_macro_f1", "es_family_macro_f1_ci_lo", "es_family_macro_f1_ci_hi", "model_bytes_mean",
            "lat_features_ms_p50", "lat_model_ms_p50"]
    print({k: round(metrics[k], 3) for k in keys})
    print("motorbike ->", tables["es_motorbike_pred_family"])


if __name__ == "__main__":
    main()
