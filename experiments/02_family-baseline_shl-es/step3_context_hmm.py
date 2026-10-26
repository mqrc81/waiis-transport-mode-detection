"""Step 3 (on R2, decided 2026-10-26): 60 s context features and HMM smoothing, paired-fold CV as in step 4.

Usage: uv run python step3_context_hmm.py [--seed 0]
Configurations, all evaluated on the same windows as R2 (SHL labelled windows, ES labels v2 + cap):
  base      R2 window-only fold models (re-predicted on all windows, needed for the HMM sequences)
  base+fwd  HMM forward filtering (causal) on base;   base+vit  Viterbi (offline) on base
  ctx       LightGBM retrained on window features + 60 s context features (same training rows as R2)
  ctx+fwd / ctx+vit
Context and HMM sequences run over all windows the phone would have produced (feature tables *_all), labelled or not.
Transitions: from SHL training days only (2.5 s steps with real labels; ES labels are per minute).
"""
import argparse
import json
import logging
import time
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from eval_tables import es_label_v2, evaluate
from tmd import hmm
from tmd.context import CTX_NAMES, StreamingContext, context_features, sequence_id, window_time_s
from tmd.evaluate import inference_latency, latency_ms, model_size_bytes
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
R2_RUN = "d668fa2e279d4776a65bc7e8f2f4977a"
log = logging.getLogger()


def class_weights(y):
    c = np.bincount(y, minlength=9)[1:].astype(float)
    return np.where(c > 0, c.sum() / (8 * np.maximum(c, 1)), 0.0)[y - 1]


def dataset(x, y, ref=None):
    return lgb.Dataset(x, y - 1, weight=class_weights(y), reference=ref, free_raw_data=True)


def es_prev_minute_in_context(es: pd.DataFrame, horizon: float = 60.0) -> np.ndarray:
    """True where a window's 60 s context contains a window from an earlier minute of the same user."""
    t, out = window_time_s(es), np.zeros(len(es), bool)
    for u in es.user.unique():
        idx = np.flatnonzero((es.user == u).to_numpy())
        idx = idx[np.argsort(t[idx], kind="stable")]
        lo = np.searchsorted(t[idx], t[idx] - horizon, side="right")
        out[idx] = es.ts.to_numpy()[idx[lo]] < es.ts.to_numpy()[idx]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(filename=HERE / "logs" / "step3.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    out_dir = PROCESSED / EXP / f"step3_{TAG}_seed{a.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    r2_dir = PROCESSED / EXP / f"r2_{TAG}_seed{a.seed}"

    t0 = time.time()
    shl = pd.read_parquet(PROCESSED / f"shl_u1_{TAG}_all.parquet")
    shl = pd.concat([shl, context_features(shl)], axis=1)
    es = pd.read_parquet(PROCESSED / f"es_dev_{TAG}_all.parquet")
    es = pd.concat([es, context_features(es)], axis=1)
    es = es_label_v2(es, seed=a.seed)
    es["prev_minute_ctx"] = es_prev_minute_in_context(es)
    log.info("loaded + context in %.0f s: SHL %d, ES %d", time.time() - t0, len(shl), len(es))
    shl_seq, shl_t = sequence_id(shl), window_time_s(shl)
    es_seq, es_t = sequence_id(es), window_time_s(es)
    shl_lab = (shl.y > 0).to_numpy()
    es_train = (es.v2_keep & (es.reason == "ok")).to_numpy()
    feats = NAMES + CTX_NAMES

    cov = {"es_eval_windows": int(es_train.sum()),
           "es_eval_frac_ctx_prev_minute": float(es.prev_minute_ctx.to_numpy()[es_train].mean()),
           "es_eval_ctx_n_mean": float(es.ctx_n.to_numpy()[es_train].mean()),
           "shl_eval_frac_ctx_full": float((shl.ctx_n.to_numpy()[shl_lab] >= 24).mean()),
           "shl_eval_ctx_n_mean": float(shl.ctx_n.to_numpy()[shl_lab].mean()),
           "es_all_windows": len(es), "shl_all_windows": len(shl)}
    print("coverage", {k: round(v, 3) if isinstance(v, float) else v for k, v in cov.items()})

    P = {c: (np.zeros((len(shl), 8)), np.zeros((len(es), 8))) for c in ["base", "ctx"]}
    trans, fold_rows, ctx_models = [], [], []
    for k in range(K):
        ts, te = (shl.fold == k).to_numpy(), (es.fold == k).to_numpy()
        base = lgb.Booster(model_file=str(r2_dir / f"model_fold{k}.txt"))
        P["base"][0][ts] = base.predict(shl.loc[ts, NAMES].to_numpy())
        P["base"][1][te] = base.predict(es.loc[te, NAMES].to_numpy())
        # ctx model: same training rows as R2 (SHL pure labelled windows, ES v2 + cap), inner fold (k+1) % 5
        inner = (k + 1) % K
        s_rows = (shl.fold != k).to_numpy() & shl.pure.to_numpy() & shl_lab
        e_rows = (es.fold != k).to_numpy() & es_train
        x = np.vstack([shl.loc[s_rows, feats].to_numpy(np.float32), es.loc[e_rows, feats].to_numpy(np.float32)])
        y = np.concatenate([shl.y.to_numpy()[s_rows], es.y.to_numpy()[e_rows]]).astype(int)
        f = np.concatenate([shl.fold.to_numpy()[s_rows], es.fold.to_numpy()[e_rows]])
        d_in = dataset(x[f != inner], y[f != inner])
        d_va = dataset(x[f == inner], y[f == inner], ref=d_in)
        m0 = lgb.train({**PARAMS, "seed": a.seed}, d_in, MAX_ROUNDS, valid_sets=[d_va],
                       callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
        m = lgb.train({**PARAMS, "seed": a.seed}, dataset(x, y), m0.best_iteration)
        del x, d_in, d_va
        m.save_model(str(out_dir / f"model_ctx_fold{k}.txt"))
        ctx_models.append(m)
        P["ctx"][0][ts] = m.predict(shl.loc[ts, feats].to_numpy())
        P["ctx"][1][te] = m.predict(es.loc[te, feats].to_numpy())
        # transitions from SHL training days only
        tr = (shl.fold != k).to_numpy()
        trans.append(hmm.transition_matrix(shl.y.to_numpy()[tr].astype(int), shl_t[tr], shl_seq[tr]))
        row = {"fold": k, "rounds_ctx": m0.best_iteration, "n_trees_ctx": m.num_trees(),
               "model_bytes_ctx": model_size_bytes(m), "train_rows": len(y)}
        fold_rows.append(row)
        log.info("fold %s", row)
        print(row)

    for c in ["base", "ctx"]:
        for kind, fn in [("fwd", hmm.forward), ("vit", hmm.viterbi)]:
            ps, pe = np.zeros_like(P[c][0]), np.zeros_like(P[c][1])
            for k in range(K):
                ts, te = (shl.fold == k).to_numpy(), (es.fold == k).to_numpy()
                ps[ts] = fn(P[c][0][ts], shl_t[ts], shl_seq[ts], trans[k])
                pe[te] = fn(P[c][1][te], es_t[te], es_seq[te], trans[k])
            P[f"{c}+{kind}"] = (ps, pe)
            log.info("%s+%s done", c, kind)
    for name, (ps, pe) in P.items():
        np.save(out_dir / f"shl_proba_{name}.npy", ps.astype(np.float32))
        np.save(out_dir / f"es_proba_{name}.npy", pe.astype(np.float32))
    np.save(out_dir / "transitions.npy", np.stack(trans))

    # Inference time per window (metric c), single thread: features (+ streaming context) + model (+ HMM step).
    a0 = trans[0]
    cur = {"v": np.full(8, 1 / 8)}

    def fwd_step(post):
        v = (cur["v"] @ a0) * post[0]
        cur["v"] = v / v.sum()
        return cur["v"]
    base0 = lgb.Booster(model_file=str(r2_dir / "model_fold0.txt"))
    sc = StreamingContext()
    lat = {
        "base": inference_latency(lambda x: base0.predict(x, num_threads=1), window_features),
        "base+fwd": inference_latency(lambda x: fwd_step(base0.predict(x, num_threads=1)), window_features),
        "ctx": inference_latency(lambda x: ctx_models[0].predict(x, num_threads=1), lambda w: sc(window_features(w))),
        "ctx+fwd": inference_latency(lambda x: fwd_step(ctx_models[0].predict(x, num_threads=1)),
                                     lambda w: sc(window_features(w))),
    }
    lat["base+vit"], lat["ctx+vit"] = lat["base"], lat["ctx"]     # Viterbi is offline: per-window cost not defined
    sizes = {"base": float(np.mean([model_size_bytes(lgb.Booster(model_file=str(r2_dir / f"model_fold{k}.txt")))
                                    for k in range(K)])),
             "ctx": float(np.mean([r["model_bytes_ctx"] for r in fold_rows]))}

    summary = {}
    shl_eval = shl[shl_lab]
    for name, (ps, pe) in P.items():
        metrics, tables, figs = evaluate(shl_eval, ps[shl_lab], es, pe, f"step3 {name}")
        metrics.update(lat[name])
        metrics["model_bytes_mean"] = sizes[name.split("+")[0]]
        metrics.update(cov)
        smooth = {"fwd": "hmm_forward", "vit": "hmm_viterbi"}.get(name.split("+")[-1], "none")
        params = {"model": "lightgbm", "dataset": "shl_u1+es_dev", "setup": "R2", "rate_hz": 25,
                  "class_set": "8class+family", "seed": a.seed, "features": FEATURES_VERSION,
                  "context": "60s mean/std of 65 feats" if name.startswith("ctx") else "none", "smoothing": smooth,
                  "transitions": "SHL training days, 2.5 s steps, +1 smoothing", "es_labels": "v2",
                  "resting_still_max_share": 0.2, "retrained": name == "ctx", "window_s": 5, "stride_s": 2.5,
                  "base_models": R2_RUN if name.startswith("base") else "this step", **{f"lgb_{k}": v for k, v in PARAMS.items()}}
        if name == "ctx":
            tables["folds"] = fold_rows
        with run(EXP, params, run_name=f"step3_{name}_feat-{FEATURES_VERSION}_seed{a.seed}"):
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})
            for fn, fig in figs.items():
                mlflow.log_figure(fig, fn)
            mlflow.log_dict(tables, "tables.json")
        (out_dir / f"metrics_{name}.json").write_text(json.dumps(metrics, indent=1))
        (out_dir / f"tables_{name}.json").write_text(json.dumps(tables, indent=1, default=int))
        summary[name] = {k: round(metrics[k], 3) for k in
                         ["shl_class_macro_f1", "shl_family_macro_f1", "es_class_macro_f1", "es_family_macro_f1",
                          "shl_family_recall_still", "shl_family_recall_rail", "shl_still_to_rail",
                          "es_still_v2_recall_family", "es_family_recall_rail", "es_still_to_rail",
                          "lat_inference_ms_p50"]}
    print(pd.DataFrame(summary).T.to_string())


if __name__ == "__main__":
    main()
