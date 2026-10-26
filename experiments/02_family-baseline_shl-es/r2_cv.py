"""Step 4 (run before step 3, approved 2026-10-26): R2 = SHL User 1 + ExtraSensory dev users, LightGBM on features
v1, cross-validated with paired folds: fold k holds out SHL day-fold k and ExtraSensory user-fold k.

Usage: uv run python r2_cv.py [--seed 0] [--setup R2|ES]
--setup ES (added 2026-10-26): ExtraSensory-only model, trained on ES user-folds only, same paired evaluation (SHL
day-fold k is then cross-dataset). No R1 reference run for ES.
Training: SHL pure windows + ExtraSensory windows labelled under rules v2 with resting still capped at 20% per user
(motorbike minutes never). Rounds by early stopping on inner fold (k+1) % 5 (SHL days + ES users), then refit.
Also logs an eval-only reference run, R1 fold-matched: the saved R1 fold-k model scored on ES user-fold k, so R1 and
R2 are compared on the same ExtraSensory windows and the same tables.
"""
import argparse
import json
import logging
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from eval_tables import es_label_v2, evaluate
from tmd.evaluate import inference_latency, model_size_bytes
from tmd.features import FEATURES_VERSION, NAMES, window_features
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


def dataset(df, ref=None):
    y = df.y.to_numpy()
    return lgb.Dataset(df[NAMES].to_numpy(), y - 1, weight=class_weights(y), reference=ref, free_raw_data=True)


def common_params(seed):
    return {"model": "lightgbm", "rate_hz": 25, "class_set": "8class+family", "seed": seed,
            "features": FEATURES_VERSION, "n_features": len(NAMES), "window_s": 5, "stride_s": 2.5,
            "es_labels": "v2", "resting_still_max_share": 0.2, "eval": "paired folds: shl day-fold k + es user-fold k",
            "context": "none", "smoothing": "none"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--setup", choices=["R2", "ES"], default="R2")
    a = ap.parse_args()
    use_shl = a.setup == "R2"
    logging.basicConfig(filename=HERE / "logs" / f"{a.setup.lower()}_cv.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    out_dir = PROCESSED / EXP / f"{a.setup.lower()}_{TAG}_seed{a.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    r1_dir = PROCESSED / EXP / f"r1_{TAG}_seed{a.seed}"

    shl = pd.read_parquet(PROCESSED / f"shl_u1_{TAG}.parquet")
    es = es_label_v2(pd.read_parquet(PROCESSED / f"es_dev_{TAG}.parquet"), seed=a.seed)
    es_train_ok = (es.v2_keep & (es.reason == "ok")).to_numpy()
    log.info("SHL %d, ES %d windows, ES trainable (v2 + cap) %d", len(shl), len(es), es_train_ok.sum())
    print(f"SHL {len(shl)} windows; ES {len(es)} windows, {es_train_ok.sum()} under v2 + cap")

    shl_p = np.zeros((len(shl), 8), np.float32)
    es_p = np.zeros((len(es), 8), np.float32)
    es_p_r1 = np.zeros((len(es), 8), np.float32)
    fold_rows, models = [], []
    for k in range(K):
        inner = (k + 1) % K
        s_tr = shl[(shl.fold != k) & shl.pure] if use_shl else shl.iloc[:0]
        e_tr = es[es_train_ok & (es.fold != k).to_numpy()]
        tr = pd.concat([s_tr[NAMES + ["y", "fold"]], e_tr[NAMES + ["y", "fold"]]], ignore_index=True)
        d_in = dataset(tr[tr.fold != inner])
        d_va = dataset(tr[tr.fold == inner], ref=d_in)
        m0 = lgb.train({**PARAMS, "seed": a.seed}, d_in, MAX_ROUNDS, valid_sets=[d_va],
                       callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
        rounds = m0.best_iteration
        m = lgb.train({**PARAMS, "seed": a.seed}, dataset(tr), rounds)
        m.save_model(str(out_dir / f"model_fold{k}.txt"))
        models.append(m)
        ts, te = (shl.fold == k).to_numpy(), (es.fold == k).to_numpy()
        shl_p[ts] = m.predict(shl.loc[ts, NAMES].to_numpy())
        es_p[te] = m.predict(es.loc[te, NAMES].to_numpy())
        if use_shl:
            r1 = lgb.Booster(model_file=str(r1_dir / f"model_fold{k}.txt"))
            es_p_r1[te] = r1.predict(es.loc[te, NAMES].to_numpy())
        row = {"fold": k, "rounds": rounds, "n_trees": m.num_trees(), "model_bytes": model_size_bytes(m),
               "train_shl": len(s_tr), "train_es": len(e_tr)}
        fold_rows.append(row)
        log.info("fold %s", row)
        print(row)
    np.save(out_dir / "shl_oof_proba.npy", shl_p)
    np.save(out_dir / "es_oof_proba.npy", es_p)
    if use_shl:
        np.save(out_dir / "es_oof_proba_r1_foldmatched.npy", es_p_r1)
    folds = pd.DataFrame(fold_rows)

    metrics, tables, figs = evaluate(shl, shl_p, es, es_p, a.setup)
    metrics.update(inference_latency(lambda x: models[0].predict(x, num_threads=1), window_features))
    metrics.update({"model_bytes_mean": float(folds.model_bytes.mean()), "n_trees_mean": float(folds.n_trees.mean()),
                    "rounds_mean": float(folds.rounds.mean())})
    tables["folds"] = fold_rows
    params = {**common_params(a.seed), "dataset": "shl_u1+es_dev" if use_shl else "es_dev", "setup": a.setup,
              "train_windows": "shl pure + es v2 capped" if use_shl else "es v2 capped",
              "class_weight": "balanced", "early_stop": "inner paired fold", **{f"lgb_{k}": v for k, v in PARAMS.items()},
              "max_rounds": MAX_ROUNDS, "patience": PATIENCE}
    name = "R2" if use_shl else "ESonly"
    with run(EXP, params, run_name=f"{name}_lgbm_feat-{FEATURES_VERSION}_seed{a.seed}"):
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})
        for fn, fig in figs.items():
            mlflow.log_figure(fig, fn)
        mlflow.log_dict(tables, "tables.json")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (out_dir / "tables.json").write_text(json.dumps(tables, indent=1, default=int))

    keys = ["shl_class_macro_f1", "shl_family_macro_f1", "es_class_macro_f1", "es_family_macro_f1",
            "es_still_v2_recall_family", "es_still_carried_recall_family", "es_rail_fp_rate_on_still"]
    print(name, {k: round(metrics[k], 3) for k in keys})
    if not use_shl:
        return

    # R1 fold-matched reference on the same ExtraSensory windows (SHL side identical to run c615aedd...)
    m1, t1, f1 = evaluate(None, None, es, es_p_r1, "R1 fold-matched")
    p1 = {**common_params(a.seed), "dataset": "shl_u1", "setup": "R1", "retrained": False,
          "source_run": "c615aedd62da40159a0b65113820aad1", "note": "R1 fold-k model on ES user-fold k"}
    with run(EXP, p1, run_name=f"R1_lgbm_feat-{FEATURES_VERSION}_seed{a.seed}_es-v2-foldmatched"):
        mlflow.log_metrics({k: float(v) for k, v in m1.items() if v == v})
        for fn, fig in f1.items():
            mlflow.log_figure(fig, fn)
        mlflow.log_dict(t1, "tables.json")
    (out_dir / "r1_foldmatched_metrics.json").write_text(json.dumps(m1, indent=1))
    (out_dir / "r1_foldmatched_tables.json").write_text(json.dumps(t1, indent=1, default=int))

    print("R1fm", {k: round(m1[k], 3) for k in keys if k in m1})


if __name__ == "__main__":
    main()
