"""Minimal MLflow tracking shared by all experiments.

Usage in an experiment script:

    import mlflow
    from tmd.tracking import run

    from tmd.tracking import DataInput, run

    data = [DataInput("shl_user1_dev", "training", label_rules="shl-coarse", features="feat-v1"),
            DataInput("extrasensory_dev", "evaluation", label_rules="v2", features="feat-v1")]
    with run("02_family-baseline_shl-es", params, datasets=data):
        mlflow.log_metrics({"macro_f1": f1, "size_mb": size, "latency_ms_p50": p50})
        mlflow.log_figure(cm_fig, "confusion_matrix.png")

Each DataInput shows in MLflow's Dataset column as name (digest of the split file) and context. Label rules,
features, rate and window are input tags (per run). Only metadata is logged, never the data.

Label-budget runs (a param `k`, labelled windows per class) also get the metric `k_per_class`, so the MLflow UI can
plot against k as a number. A non-numeric k (e.g. "all") needs the number in params as `k_per_class`.

Export an experiment's runs (the CSV that goes in git):

    uv run python -m tmd.tracking export 02_family-baseline_shl-es
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import mlflow
from mlflow.data.dataset_source_registry import get_registered_sources
from mlflow.data.meta_dataset import MetaDataset
from mlflow.entities import DatasetInput, InputTag
from mlflow.utils.mlflow_tags import MLFLOW_DATASET_CONTEXT

from tmd.paths import MLARTIFACTS as ARTIFACTS
from tmd.paths import MLFLOW_DB, PROJECT

TRACKING_URI = f"sqlite:///{MLFLOW_DB}"


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=PROJECT, capture_output=True, text=True)
    return out.stdout.strip()


def _use_experiment(name: str) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(ARTIFACTS / name).as_uri())
    mlflow.set_experiment(name)


# Dataset part -> split file (repo-relative). A new split gets a new name here.
SPLITS = {
    "shl_user1_dev": "experiments/02_family-baseline_shl-es/splits/shl_days.csv",  # 02's day folds
    "extrasensory_dev": "experiments/02_family-baseline_shl-es/splits/es_users.csv",  # 02's user folds
    "shl_user23_test": "splits/reserved_test.json",
    "extrasensory_test": "splits/reserved_test.json",
    "shl_user1_03_training": "experiments/03_label-curve_shl/splits/shl_u1_days.csv",     # 03's day split
    "shl_user1_03_validation": "experiments/03_label-curve_shl/splits/shl_u1_days.csv",
    "shl_user1_03_test": "experiments/03_label-curve_shl/splits/shl_u1_days.csv",
}
CONTEXTS = ("training", "evaluation")
# Registered dynamically by MLflow, so it can't be imported by name.
_LocalSource = next(s for s in get_registered_sources() if s.__name__ == "LocalArtifactDatasetSource")


@dataclass(frozen=True)
class DataInput:
    """One dataset part a run used. Metadata only: the data never go into MLflow."""
    name: str  # dataset and part, a key of SPLITS
    context: str  # "training" or "evaluation"
    label_rules: str  # e.g. "v2" (ExtraSensory), "shl-coarse"
    features: str  # e.g. "feat-v1"
    rate_hz: int = 25
    window_s: float = 5
    stride_s: float = 2.5
    oof_eval: bool = False  # training data also scored out-of-fold (CV); MLflow keeps one input per dataset
    extra: tuple[tuple[str, str], ...] = ()  # further input tags, e.g. (("trained_in_run", run_id),)

    @property
    def split(self) -> str:
        return SPLITS[self.name]

    def dataset(self) -> MetaDataset:
        digest = hashlib.sha256((PROJECT / self.split).read_bytes()).hexdigest()[:8]
        return MetaDataset(_LocalSource(self.split), name=self.name, digest=digest)

    def tags(self) -> dict[str, str]:
        t = {"split": self.split, "label_rules": self.label_rules, "features": self.features,
             "rate_hz": str(self.rate_hz), "window_s": str(self.window_s), "stride_s": str(self.stride_s)}
        if self.oof_eval:
            t["eval"] = "out-of-fold"
        return {**t, **dict(self.extra)}

    def entity(self) -> DatasetInput:
        """The input as MlflowClient.log_inputs takes it (what mlflow.log_input builds internally)."""
        tags = [InputTag(MLFLOW_DATASET_CONTEXT, self.context)] + [InputTag(k, v) for k, v in self.tags().items()]
        return DatasetInput(dataset=self.dataset()._to_mlflow_entity(), tags=tags)


def check_inputs(datasets: list[DataInput]) -> None:
    names = [d.name for d in datasets]
    if len(set(names)) != len(names):
        raise ValueError(f"dataset logged twice in one run (MLflow keeps only the first): {names}")
    for d in datasets:
        if d.name not in SPLITS or d.context not in CONTEXTS:
            raise ValueError(f"unknown dataset or context: {d}")


@contextmanager
def run(experiment: str, params: dict, run_name: str | None = None, *, datasets: list[DataInput]):
    """Start a run tagged with the git commit and whether the tree was dirty, with its dataset inputs.

    `datasets` is required; pass [] only for runs that use no data (smoke tests).
    """
    check_inputs(datasets)
    k_num = k_per_class(params)
    _use_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as active:
        mlflow.set_tags({
            "git_commit": _git("rev-parse", "--short", "HEAD") or "none",
            "dirty": str(bool(_git("status", "--porcelain"))).lower(),
        })
        mlflow.log_params({p: v for p, v in params.items() if p != "k_per_class"})   # metric only (export columns)
        if k_num is not None:
            mlflow.log_metric("k_per_class", k_num)
        for d in datasets:
            mlflow.log_input(d.dataset(), context=d.context, tags=d.tags())
        yield active


def k_per_class(params: dict) -> float | None:
    """Labelled windows per class as a number: params["k_per_class"], else a numeric params["k"], else None
    (no k). A non-numeric k without k_per_class is an error, so label curves can always be plotted in MLflow."""
    if "k_per_class" in params:
        return float(params["k_per_class"])
    if "k" not in params:
        return None
    try:
        return float(params["k"])
    except (TypeError, ValueError):
        raise ValueError(f"k = {params['k']!r} is not a number: pass k_per_class in params") from None


def export(experiment: str, out: Path | None = None) -> Path:
    """Write one row per run (params, metrics, commit, dirty) to the experiment's results/."""
    _use_experiment(experiment)
    df = mlflow.search_runs(experiment_names=[experiment], order_by=["start_time ASC"])
    keep = ["run_id", "start_time", "tags.git_commit", "tags.dirty"]
    keep += [c for c in df.columns if c.startswith(("params.", "metrics."))]
    df = df[keep].rename(columns=lambda c: c.split(".", 1)[-1])
    out = out or PROJECT / "experiments" / experiment / "results" / "runs.csv"
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
