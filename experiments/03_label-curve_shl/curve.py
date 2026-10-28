"""Label curve: LightGBM on k sampled windows per class, acc + gyro vs acc only, validation days only.

Usage: uv run python curve.py --seeds 0 --stage sp2 [--workers 5]
1. Grid (k <= 1000): every setting of GRID per (sensor set, k, seed), early stopping on the validation days. Cached
   in results/grid/<sensor>_k<k>_seed<s>.csv; existing files are not refitted.
2. Choice per (sensor set, k): best mean validation macro-F1 over the seeds given. k = all: 02's parameters, no grid.
3. Final fit per (sensor set, k, seed) with the chosen setting: model to <data>/checkpoints/03_label-curve_shl/<stage>/,
   validation probabilities to <data>/processed/03_label-curve_shl/<stage>/, one MLflow run each.
Test days are never read here. Full log: logs/curve.log.
"""
import argparse
import itertools
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from tmd.evaluate import family_proba, model_size_bytes, scores
from tmd.features import FEATURES_VERSION, NAMES
from tmd.labels import CLASSES
from tmd.paths import CHECKPOINTS, PROCESSED
from tmd.tracking import DataInput, run

HERE = Path(__file__).resolve().parent
EXP = HERE.name
TABLE = PROCESSED / "shl_u1_25hz_w5s_s2.5s_feat-v1_all.parquet"
CACHE = PROCESSED / EXP / "cache"
KS = [1, 2, 5, 10, 50, 200, 1000, "all"]
SENSORS = {"acc_gyro": NAMES, "acc": [n for n in NAMES if not n.startswith("gyro_norm")]}
GRID = [dict(min_data_in_leaf=m, num_leaves=nl, learning_rate=lr)
        for m, nl, lr in itertools.product([1, 5, 20, 50], [7, 31], [0.05, 0.1])]
BASE = dict(objective="multiclass", num_class=8, feature_fraction=0.8, lambda_l2=1.0, min_data_in_bin=1,
            feature_pre_filter=False, verbose=-1)
# 02's R1 parameters (experiments/02_family-baseline_shl-es/r1_cv.py:30), for k = all
PARAMS_02 = dict(objective="multiclass", num_class=8, learning_rate=0.1, num_leaves=31, min_child_samples=50,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1)
MAX_ROUNDS, PATIENCE = 3000, 50
GRID_THREADS, ALL_THREADS = 2, 10
log = logging.getLogger()


def build_cache() -> None:
    """Labelled hips windows of training and validation days: features (npy, for memory-mapped workers) + meta."""
    if (CACHE / "meta.parquet").exists():
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    part = pd.read_csv(HERE / "splits" / "shl_u1_days.csv", dtype={"day": str}).set_index("day").partition
    w = pd.read_parquet(TABLE, filters=[("position", "==", "Hips")])
    w = w[(w.y > 0) & w.day.map(part).isin(["training", "validation"])]
    w = w.assign(partition=w.day.map(part)).sort_values(["rec", "t_start_ms"]).reset_index(drop=True)
    np.save(CACHE / "X.npy", w[NAMES].to_numpy(np.float32))
    w[["day", "rec", "t_start_ms", "y", "pure", "partition"]].to_parquet(CACHE / "meta.parquet")


_X = None


def X() -> np.ndarray:
    global _X
    if _X is None:
        _X = np.load(CACHE / "X.npy", mmap_mode="r")
    return _X


def meta() -> pd.DataFrame:
    return pd.read_parquet(CACHE / "meta.parquet")


def train_rows(m: pd.DataFrame, k, seed: int) -> np.ndarray:
    tr = m[(m.partition == "training") & m.pure]
    if k == "all":
        return tr.index.to_numpy()
    s = pd.read_csv(HERE / "splits" / "samples_k.csv", dtype={"day": str})
    s = s[(s.seed == seed) & (s["rank"] < k)]
    rows = tr.reset_index().merge(s[["rec", "t_start_ms"]], on=["rec", "t_start_ms"])["index"].to_numpy()
    assert len(rows) == 8 * k, (k, seed, len(rows))
    return np.sort(rows)


def fit(rows: np.ndarray, val: np.ndarray, y: np.ndarray, cols: list[int], params: dict, seed: int, threads: int):
    x = X()
    ytr = y[rows] - 1
    w = len(ytr) / (8 * np.bincount(ytr, minlength=8))           # balanced class weights
    d = lgb.Dataset(np.asarray(x[rows][:, cols]), ytr, weight=w[ytr])
    dv = lgb.Dataset(np.asarray(x[val][:, cols]), y[val] - 1, reference=d)
    p = {**params, "seed": seed, "num_threads": threads, "deterministic": True}
    b = lgb.train(p, d, MAX_ROUNDS, valid_sets=[dv], callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
    prob = b.predict(np.asarray(x[val][:, cols]), num_iteration=b.best_iteration, num_threads=threads)
    return b, prob


def val_scores(y: np.ndarray, prob: np.ndarray) -> dict:
    from tmd.labels import to_family
    c = scores(y, prob.argmax(1) + 1, "class")
    f = scores(to_family(y), family_proba(prob).argmax(1) + 1, "family")
    return {"val_macro_f1": c["macro_f1"], "val_macro_f1_family": f["macro_f1"],
            **{f"val_recall_{n}": r for n, r in c["recall"].items()}, "_cm": c["cm"]}


def grid_task(sensor: str, k, seed: int, g: dict) -> dict:
    m = meta()
    rows, val = train_rows(m, k, seed), np.flatnonzero(m.partition == "validation")
    cols = [NAMES.index(n) for n in SENSORS[sensor]]
    b, prob = fit(rows, val, m.y.to_numpy(), cols, {**BASE, **g}, seed, GRID_THREADS)
    s = val_scores(m.y.to_numpy()[val], prob)
    return {**g, "best_iter": b.best_iteration, "val_macro_f1": s["val_macro_f1"],
            "val_macro_f1_family": s["val_macro_f1_family"]}


def grid_path(sensor, k, seed) -> Path:
    return HERE / "results" / "grid" / f"{sensor}_k{k}_seed{seed}.csv"


def run_grid(seeds, workers: int) -> None:
    jobs = [(s, k, seed, g) for seed in seeds for s in SENSORS for k in KS if k != "all"
            for g in GRID if g["min_data_in_leaf"] <= k and not grid_path(s, k, seed).exists()]
    if not jobs:
        return
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(grid_task, *zip(*jobs)))
    df = pd.DataFrame([{"sensor": s, "k": k, "seed": seed, **r} for (s, k, seed, _), r in zip(jobs, res)])
    for (s, k, seed), g in df.groupby(["sensor", "k", "seed"]):
        grid_path(s, k, seed).parent.mkdir(parents=True, exist_ok=True)
        g.to_csv(grid_path(s, k, seed), index=False)
        log.info("grid %s k=%s seed=%s best %.4f", s, k, seed, g.val_macro_f1.max())


def choose(sensor: str, k, seeds) -> tuple[dict, pd.DataFrame]:
    g = pd.concat([pd.read_csv(grid_path(sensor, k, s)) for s in seeds])
    keys = list(GRID[0])
    mean = g.groupby(keys).val_macro_f1.mean().sort_values(ascending=False)
    best = dict(zip(keys, mean.index[0]))
    best = {kk: (int(v) if kk != "learning_rate" else float(v)) for kk, v in best.items()}
    return best, g


def final(sensor: str, k, seed: int, seeds, stage: str) -> dict:
    m = meta()
    y = m.y.to_numpy()
    rows, val = train_rows(m, k, seed), np.flatnonzero(m.partition == "validation")
    cols = [NAMES.index(n) for n in SENSORS[sensor]]
    if k == "all":
        params, grid, threads = PARAMS_02, None, ALL_THREADS
    else:
        best, grid = choose(sensor, k, seeds)
        params, threads = {**BASE, **best}, GRID_THREADS
    b, prob = fit(rows, val, y, cols, params, seed, threads)
    s = val_scores(y[val], prob)
    cm = s.pop("_cm")

    name = f"shl_u1_hips_25hz_w5s_feat-{FEATURES_VERSION}-{sensor}_k{k}_seed{seed}"
    ck = CHECKPOINTS / EXP / stage
    pr = PROCESSED / EXP / stage
    ck.mkdir(parents=True, exist_ok=True)
    pr.mkdir(parents=True, exist_ok=True)
    b.save_model(ck / f"{name}.txt", num_iteration=b.best_iteration)
    out = m.iloc[val][["day", "rec", "t_start_ms", "y", "pure"]].reset_index(drop=True)
    out = pd.concat([out, pd.DataFrame(prob.astype(np.float32), columns=[f"p_{c}" for c in CLASSES])], axis=1)
    out.to_parquet(pr / f"{name}_val_probs.parquet")      # already in (rec, t_start_ms) order

    run_params = {"model": "lightgbm", "dataset": "shl_u1_hips", "rate_hz": 25, "class_set": "shl8", "seed": seed,
                  "k": k, "sensors": sensor, "n_features": len(cols), "stage": stage,
                  "selection": "02 params" if k == "all" else f"grid, mean val macro-F1 over seeds {list(seeds)}",
                  "n_train": len(rows), "n_val": len(val), **{f"lgb_{kk}": v for kk, v in params.items()}}
    samples = "all training-day pure windows" if k == "all" else "splits/samples_k.csv"
    data = [DataInput("shl_user1_03_training", "training", "shl-coarse", f"feat-{FEATURES_VERSION}-{sensor}",
                      extra=(("samples", samples), ("k", str(k)), ("seed", str(seed)))),
            DataInput("shl_user1_03_validation", "evaluation", "shl-coarse", f"feat-{FEATURES_VERSION}-{sensor}",
                      extra=(("use", "early stopping + selection"),))]
    with run(EXP, run_params, run_name=f"{stage}_{sensor}_k{k}_seed{seed}", datasets=data):
        mlflow.log_metrics({**s, "best_iter": b.best_iteration, "model_size_bytes": model_size_bytes(b)})
        mlflow.log_dict({"cm_val_class": cm.tolist()}, "cm_val_class.json")
        if grid is not None:
            mlflow.log_table(grid, "grid_val.json")
    log.info("final %s k=%s seed=%s %s val %.4f fam %.4f iter %d", sensor, k, seed, params, s["val_macro_f1"],
             s["val_macro_f1_family"], b.best_iteration)
    return {"sensor": sensor, "k": k, "seed": seed, "val_macro_f1": s["val_macro_f1"],
            "val_macro_f1_family": s["val_macro_f1_family"], "best_iter": b.best_iteration,
            **({} if k == "all" else best)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--stage", default="sp2")
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    logging.basicConfig(filename=HERE / "logs" / "curve.log", filemode="a", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    log.info("start %s", vars(a))
    build_cache()
    run_grid(a.seeds, a.workers)
    res = [final(s, k, seed, a.seeds, a.stage) for s in SENSORS for k in KS for seed in a.seeds]
    df = pd.DataFrame(res)
    df.to_csv(HERE / "results" / f"curve_{a.stage}.csv", index=False)
    print(df.groupby(["sensor", "k"], sort=False)[["val_macro_f1", "val_macro_f1_family", "best_iter"]]
          .mean().round(3).unstack(0).to_string())


if __name__ == "__main__":
    main()
