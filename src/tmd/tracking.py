"""Minimal MLflow tracking shared by all experiments.

Usage in an experiment script:

    import mlflow
    from tmd.tracking import run

    with run("02_family-baseline_shl-es", params):
        mlflow.log_metrics({"macro_f1": f1, "size_mb": size, "latency_ms_p50": p50})
        mlflow.log_figure(cm_fig, "confusion_matrix.png")

Export an experiment's runs (the CSV that goes in git):

    uv run python -m tmd.tracking export 02_family-baseline_shl-es
"""
from __future__ import annotations

import argparse
import subprocess
from contextlib import contextmanager
from pathlib import Path

import os

import mlflow
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]       # src/tmd/tracking.py -> repo root
load_dotenv(_ROOT / ".env")                       # by path, so scripts can run from any folder
REPO = Path(os.getenv("REPO", _ROOT))
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{REPO / 'mlflow.db'}")  # git-ignored
ARTIFACTS = Path(os.getenv("MLFLOW_ARTIFACTS", REPO / "mlartifacts"))               # git-ignored


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    return out.stdout.strip()


def _use_experiment(name: str) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(ARTIFACTS / name).as_uri())
    mlflow.set_experiment(name)


@contextmanager
def run(experiment: str, params: dict, run_name: str | None = None):
    """Start a run tagged with the git commit and whether the tree was dirty."""
    _use_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as active:
        mlflow.set_tags({
            "git_commit": _git("rev-parse", "--short", "HEAD") or "none",
            "dirty": str(bool(_git("status", "--porcelain"))).lower(),
        })
        mlflow.log_params(params)
        yield active


def export(experiment: str, out: Path | None = None) -> Path:
    """Write one row per run (params, metrics, commit, dirty) to the experiment's results/."""
    _use_experiment(experiment)
    df = mlflow.search_runs(experiment_names=[experiment], order_by=["start_time ASC"])
    keep = ["run_id", "start_time", "tags.git_commit", "tags.dirty"]
    keep += [c for c in df.columns if c.startswith(("params.", "metrics."))]
    df = df[keep].rename(columns=lambda c: c.split(".", 1)[-1])
    out = out or REPO / "experiments" / experiment / "results" / "runs.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="export an experiment's runs to CSV")
    e.add_argument("experiment")
    e.add_argument("--out", type=Path)
    a = p.parse_args()
    print(export(a.experiment, a.out))
