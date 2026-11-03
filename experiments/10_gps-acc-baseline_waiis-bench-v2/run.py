"""One row of the 10 ablation: acc model (+ GPS model), product fusion with the tram rule, HMM forward-backward,
evaluation, shortcut checks, latency, one MLflow run.

    uv run python run.py --row A0 --sources extrasensory shl_user1 --n 8
    uv run python run.py --row A1 --sources extrasensory shl_user1 nor_tmd --gps-from A0 --vs A0
    uv run python run.py --row A2 --sources extrasensory shl_user1 nor_tmd collecty --gps-from A0 --vs A0 A1
    uv run python run.py --row S --sources ... --shl-weight 0.5 --vs A0 <best>         # GPS model retrained
    uv run python run.py --row P --sources ... --positions --gps-from <best> --vs A0 <best>
Log: logs/run_<row>_seed<k>.log; tables also in results/tables/<row>_seed<k>/.
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
import pandas as pd

import pipe
from tmd.evaluate import cm_metrics
from tmd.models import fusion, lgbm, minutes
from tmd.progress import Progress
from tmd.tracking import run

log = logging.getLogger()


def gps_models(name, seed):
    d = pipe.MODELS
    return [lgb.Booster(model_file=str(d / f"{name}_gps_seed{seed}_fold{k}.txt")) for k in range(lgbm.K)], \
        json.loads((d / f"{name}_gps_seed{seed}_folds.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True)
    ap.add_argument("--sources", nargs="+", required=True, choices=pipe.SOURCES)
    ap.add_argument("--n", type=int, default=9, choices=[8, 9])
    ap.add_argument("--shl-weight", type=float, default=1.0)
    ap.add_argument("--positions", action="store_true")
    ap.add_argument("--gps-from", help="reuse this row's GPS model (same GPS training data)")
    ap.add_argument("--vs", nargs="*", default=[])
    ap.add_argument("--hmm-sources", nargs="+", choices=pipe.SOURCES, help="E1 / E2: HMM transitions from these")
    ap.add_argument("--stop-no-tram", action="store_true", help="E1 / E2: tram out of the early-stopping fold")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    row = pipe.Row(a.row, tuple(a.sources), a.n, a.shl_weight, a.positions, a.gps_from, a.vs,
                   tuple(a.hmm_sources) if a.hmm_sources else None, a.stop_no_tram)
    name = f"{row.name}_seed{a.seed}"
    (pipe.HERE / "logs").mkdir(exist_ok=True)
    logging.basicConfig(filename=pipe.HERE / "logs" / f"run_{name}.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    t0 = time.time()
    prm = {"model": "lightgbm acc x gps-v2 lightgbm, product, hmm-fb (06's H1_v2)", "row": row.name, "seed": a.seed,
           "rate_hz": 25, "dataset": "+".join(row.sources) + (" +shl_u1 other positions" if row.positions else ""),
           "class_set": f"{row.n}class (six-class primary, fine, family)", "n_classes": row.n,
           "primary": "es six-class macro-F1, pooled OOF", "guard": "nor_tmd dev bus/car/rail six-class macro-F1",
           "sources": ",".join(row.sources), "shl_weight": row.shl_weight, "positions": row.positions,
           "gps_from": row.gps_from or "", "gps_train": "extrasensory+shl_user1 (hips), covered official minutes",
           "vs": ",".join(row.vs), "tram_rule": "vehicle_mean" if row.n == 9 else "",
           "fusion": "product, acc where no GPS", "smoothing": "fb", "features": pipe.datasets(row)[0].features,
           "acc_lr": pipe.ACC_FIT[0], "gps_lr": pipe.GPS_FIT[0], "benchmark": pipe.BENCH,
           "bootstrap": "users (ES) / installations (NOR-TMD), redraw resamples lacking a class",
           "hmm_transitions": "labelled minutes (fine) of " + ",".join(row.hmm()),
           "hmm_sources": ",".join(row.hmm()), "stop_no_tram": row.stop_no_tram,
           **{f"lgb_{k}": v for k, v in lgbm.PARAMS.items() if k not in ("learning_rate", "num_class")}}
    units = 2 * lgbm.K * (1 + (row.gps_from is None)) + 3
    prog = Progress(pipe.EXP, name, units, log)
    with run(pipe.EXP, prm, run_name=name, datasets=pipe.datasets(row)):
        m, figs, tab = row_run(row, a.seed, prog)
        mlflow.log_metrics({k: float(v) for k, v in m.items() if v == v})
        for fn, fig in figs.items():
            mlflow.log_figure(fig, fn)
            plt.close(fig)
        mlflow.log_dict(pipe.dump(tab), "tables.json")
    log.info("metrics %s", json.dumps(m, default=float))
    prog.step("MLflow")
    key = ["es_six_f1", "es_six_f1_lo", "es_six_f1_hi", "es_fine_f1", "es_family_f1", "es_six_precision_still",
           "es_six_precision_foot", "nor_guard_six_f1", "collecty_six_f1", "shl_six_f1",
           *[k for k in m if k.startswith("es_six_f1_diff_") or k.startswith("nor_guard_six_f1_diff_")],
           "tram_pred_extrasensory_usable_n", "tram_pred_shl_user1_usable_n", "lat_inference_ms_p50",
           "lat_pipeline_ms_p50"]
    print(name, {k: round(m[k], 4) for k in key if k in m}, f"{time.time() - t0:.0f}s", flush=True)


def row_run(row: pipe.Row, seed: int, prog):
    df = pipe.load()
    lab = pipe.labelled(row.hmm(), row.n)
    cov = df.covered.to_numpy()
    es6 = ((df.source == "extrasensory") & df.official).to_numpy()
    y6 = pipe.truth(df.y.to_numpy())["six"]

    def on_fold(kind):
        def f(r, raw_p):
            k = r["fold"]
            sel = es6 & (df.fold.to_numpy() == k) & (cov if kind == "gps" else True)
            pr = pipe.preds(pipe.pad9(raw_p[sel]))["six"]
            f1 = cm_metrics(pipe._cm(y6[sel], pr, "six"), np.isin(np.arange(1, 7), np.unique(y6[sel])))["macro_f1"]
            mlflow.log_metrics({f"fold{k}_{kind}_es_six_f1_raw": f1,
                                **{f"fold{k}_{kind}_{n}": float(r[n]) for n in ("rounds", "n_trees", "model_bytes",
                                                                                "hit_cap", "train_rows")}})
        return f

    tr_a, sw_a = pipe.acc_trainable(df, row)
    acc = pipe.fit(df, pipe.ACC, tr_a, sw_a, row.n, pipe.ACC_FIT, seed, log, f"{row.name} acc",
                   stop_no_tram=row.stop_no_tram, progress=prog, on_fold=on_fold("acc"))
    pipe.save_models(row.name, seed, acc)
    tr_g, sw_g = pipe.gps_trainable(df, row)
    if row.gps_from:
        p_gps = pipe.load_probs(row.gps_from, seed, df, "gps")
        g_models, g_rows = gps_models(row.gps_from, seed)
    else:
        gps = pipe.fit(df, pipe.GPS, tr_g, sw_g, 8, pipe.GPS_FIT, seed, log, f"{row.name} gps", progress=prog,
                       on_fold=on_fold("gps"))
        pipe.save_models(f"{row.name}_gps", seed, gps)
        p_gps, g_models, g_rows = gps["p"], gps["models"], gps["rows"]
    raw = fusion.product(acc["p"], p_gps, cov)
    p = pipe.pad9(minutes.smooth_cv(df, raw, lab, "fb"))
    pipe.save(row.name, seed, df, acc=acc["p"], gps=p_gps, raw=raw, p=p,
              info=json.dumps({"row": vars(row), "seed": seed}))

    bases = {b: pipe.load_probs(b, seed, df) for b in row.vs}
    m, tab, figs = pipe.evaluate(df, p, bases)
    prog.step("evaluation")

    # (d) source identification within class, from the model's out-of-fold probabilities
    ident = pipe.identification(df, {"acc": pipe.pad9(acc["p"]), "final": p}, pipe.id_sample())
    for cls, r in ident.iterrows():
        for c in ("auc_acc", "auc_final", "auc_final_uncovered", "auc_features_09"):
            if r[c] == r[c]:
                m[f"id_{c}_{cls}"] = r[c]
    tab["identification"] = ident
    prog.step("identification")

    # training weight per class and source; HMM: labelled transitions per source, diagonal
    tab["weight_share_acc"] = pipe.weight_share(df, tr_a, sw_a, row.n)
    tab["weight_share_gps"] = pipe.weight_share(df, tr_g, sw_g, 8)
    tab["transitions"] = pipe.transitions(lab)
    a_all, pi_all = minutes.hmm_params(lab[lab.fold >= 0], row.n)
    tab["hmm_diagonal"] = dict(zip(pipe.NAMES["fine"][:row.n], np.round(np.diag(a_all), 4)))
    for s, r in tab["transitions"].iterrows():
        m[f"hmm_transitions_{s}"], m[f"hmm_steps_{s}"] = r.transitions, r.steps
    for c, v in tab["hmm_diagonal"].items():
        m[f"hmm_diag_{c}"] = v
    for kind, rows in (("acc", acc["rows"]), ("gps", g_rows)):
        m[f"{kind}_rounds_mean"] = float(np.mean([r["rounds"] for r in rows]))
        m[f"{kind}_hit_cap_folds"] = float(sum(r["hit_cap"] for r in rows))
    m["acc_model_bytes_mean"], m["gps_model_bytes_mean"] = pipe.model_bytes(acc["models"]), pipe.model_bytes(g_models)
    tab["folds_acc"], tab["folds_gps"] = acc["rows"], g_rows

    # latency: fold-0 models and HMM parameters, single thread
    a0, pi0 = minutes.hmm_params(lab[(lab.fold != 0) & (lab.fold >= 0)], row.n)
    m.update(fusion.latency(fusion.session_pool(df, seed, version=pipe.BENCH), acc["models"][0], g_models[0], a0, pi0,
                            pipe.GPS_VERSION))
    out = pipe.HERE / "results" / "tables" / f"{row.name}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    for k in ("identification", "weight_share_acc", "weight_share_gps", "transitions"):
        pd.DataFrame(tab[k]).to_csv(out / f"{k}.csv")
    pd.DataFrame(tab["recall_by_source"]).to_csv(out / "recall_by_source.csv")
    return m, figs, tab


if __name__ == "__main__":
    main()
