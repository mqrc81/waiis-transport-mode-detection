"""Metrics (c) and (d) for finished runs: inference time per window (features + model, single thread, after warm-up,
median and p95) and model size, for every fold model. Adds them to the existing MLflow runs.

Usage: uv run python bench_inference.py
Metrics added: lat_inference_ms_p50 / _p95 (mean over the 5 fold models), the same per fold, model_bytes_mean.
"""
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np

from tmd.evaluate import inference_latency, model_size_bytes
from tmd.features import FEATURES_VERSION, window_features
from tmd.paths import PROCESSED
from tmd.tracking import TRACKING_URI

EXP = Path(__file__).resolve().parent.name
TAG = f"25hz_w5s_s2.5s_feat-{FEATURES_VERSION}"
RUNS = {"c615aedd62da40159a0b65113820aad1": f"r1_{TAG}_seed0",      # R1_lgbm_feat-v1_seed0
        "d668fa2e279d4776a65bc7e8f2f4977a": f"r2_{TAG}_seed0"}      # R2_lgbm_feat-v1_seed0


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    for run_id, d in RUNS.items():
        per = []
        for k in range(5):
            m = lgb.Booster(model_file=str(PROCESSED / EXP / d / f"model_fold{k}.txt"))
            lat = inference_latency(lambda x: m.predict(x, num_threads=1), window_features)
            per.append({**lat, "model_bytes": model_size_bytes(m), "n_trees": m.num_trees()})
        out = {f"{key}_fold{k}": v for k, r in enumerate(per) for key, v in r.items()}
        for key in ["lat_inference_ms_p50", "lat_inference_ms_p95", "model_bytes"]:
            out[f"{key}_mean" if key == "model_bytes" else key] = float(np.mean([r[key] for r in per]))
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(out)
        print(d, {k: round(v, 3) for k, v in out.items() if "fold" not in k},
              "per fold p50", [round(r["lat_inference_ms_p50"], 3) for r in per],
              "trees", [r["n_trees"] for r in per])


if __name__ == "__main__":
    main()
