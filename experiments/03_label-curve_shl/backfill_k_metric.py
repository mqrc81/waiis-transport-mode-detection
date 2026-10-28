"""Add the numeric metric k_per_class (= n_train / 8) to the existing runs of experiment 03 (2026-10-28).

The MLflow UI plots params as text, and k = "all" has no position on an axis. Only `MlflowClient.log_metric` is
called: params, other metrics, tags and artifacts are not touched. Runs that already have it are skipped. Dry run by
default; `--apply` writes.

    uv run python backfill_k_metric.py            # table of run -> k_per_class, writes nothing
    uv run python backfill_k_metric.py --apply
"""
import argparse
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from tmd.tracking import TRACKING_URI

EXP = Path(__file__).resolve().parent.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    mlflow.set_tracking_uri(TRACKING_URI)
    d = mlflow.search_runs(experiment_names=[EXP])
    if "metrics.k_per_class" in d:
        d = d[d["metrics.k_per_class"].isna()]
    d = d.assign(k_per_class=d["params.n_train"].astype(int) / 8)
    print(d[["tags.mlflow.runName", "params.k", "k_per_class"]].sort_values("k_per_class").to_string(index=False))
    if a.apply:
        c = MlflowClient()
        for rid, k in zip(d.run_id, d.k_per_class):
            c.log_metric(rid, "k_per_class", k)
        print(f"logged k_per_class on {len(d)} runs")


if __name__ == "__main__":
    main()
