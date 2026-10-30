"""Minimal MLflow tracking shared by all experiments.

Usage in an experiment script:

    import mlflow
    from tmd.tracking import run

    from tmd.tracking import DataInput, run

    data = [DataInput("shl_user1_dev", "training", label_rules="shl-coarse", features="feat-v1"),
            DataInput("extrasensory_dev_v2", "evaluation", label_rules="bench-v1", features="feat-v1")]
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
import fnmatch
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
    "shl_user1_dev": "experiments/04_frozen-benchmark_es-shl/splits/shl_days.csv",  # = 02's file, same sha256
    "extrasensory_dev": "experiments/02_family-baseline_shl-es/splits/es_users.csv",  # legacy (02 only)
    "shl_user23_test": "experiments/02_family-baseline_shl-es/splits/reserved_test.json",     # legacy (02 only)
    "extrasensory_test": "experiments/02_family-baseline_shl-es/splits/reserved_test.json",   # legacy (02 only)
    "extrasensory_dev_v2": "experiments/04_frozen-benchmark_es-shl/splits/es_users_v2.csv",  # approved 2026-10-28
    "extrasensory_test_v2": "splits/reserved_test_v2.json",                                  # the reference
    "shl_user23_test_v2": "splits/reserved_test_v2.json",
    "shl_user1_03_training": "experiments/03_label-curve_shl/splits/shl_u1_days.csv",     # legacy (03 only)
    "shl_user1_03_validation": "experiments/03_label-curve_shl/splits/shl_u1_days.csv",
    "shl_user1_03_test": "experiments/03_label-curve_shl/splits/shl_u1_days.csv",
}
# Names of superseded splits, kept only so the records of experiments 02 and 03 resolve. A new run must use a
# current name: extrasensory_dev_v2, extrasensory_test_v2, shl_user23_test_v2, shl_user1_dev (waiis_bench_v1).
LEGACY = {"extrasensory_dev", "extrasensory_test", "shl_user23_test",                  # 02's ExtraSensory split
          "shl_user1_03_training", "shl_user1_03_validation", "shl_user1_03_test"}     # 03's day split
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


# Changes that don't make a run's code or inputs differ from its commit (fnmatch patterns on repo-relative paths;
# `*` also matches "/"). DIRTY_KEEP wins over DIRTY_IGNORE.
DIRTY_IGNORE = (
    "experiments/*/logs/*",       # run logs written while the run is going (since 05)
    "experiments/*/results/*",    # run outputs: tables, CSVs, figures, runs.csv exports (since 06)
    "log/*",                      # the daily logs
    "*.md",                       # documentation (READMEs, STATUS, DECISIONS, CLAUDE.md)
)
DIRTY_KEEP = (
    "experiments/04_frozen-benchmark_es-shl/results/manifests/*",   # read by tmd.bench.verify(): an input
)


def changed_paths() -> list[str]:
    """Repo-relative paths with uncommitted changes, untracked files listed one by one (renames: both paths)."""
    out = subprocess.run(["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"], cwd=PROJECT,
                         capture_output=True, text=True).stdout
    tok, paths, i = out.split("\0"), [], 0
    while i < len(tok):
        if len(tok[i]) > 3:
            paths.append(tok[i][3:])
            if tok[i][0] in "RC":                 # a rename / copy is followed by its original path
                i += 1
                paths.append(tok[i])
        i += 1
    return paths


def counts_as_dirty(path: str) -> bool:
    if any(fnmatch.fnmatchcase(path, k) for k in DIRTY_KEEP):
        return True
    return not any(fnmatch.fnmatchcase(path, ig) for ig in DIRTY_IGNORE)


def is_dirty() -> bool:
    """Uncommitted changes in the repo other than DIRTY_IGNORE (logs, run outputs, daily logs, docs), DIRTY_KEEP
    excepted."""
    return any(counts_as_dirty(x) for x in changed_paths())


@contextmanager
def run(experiment: str, params: dict, run_name: str | None = None, *, datasets: list[DataInput]):
    """Start a run tagged with the git commit and whether the tree was dirty, with its dataset inputs.

    `datasets` is required; pass [] only for runs that use no data (smoke tests). Legacy split names
    (LEGACY: superseded splits of experiments 02 and 03) are refused.
    """
    legacy = sorted({d.name for d in datasets} & LEGACY)
    if legacy:
        raise ValueError(f"legacy split name(s) {legacy}: superseded split, kept only for experiments 02 / 03; "
                         "use extrasensory_dev_v2, extrasensory_test_v2, shl_user23_test_v2 or shl_user1_dev")
    check_inputs(datasets)
    k_num = k_per_class(params)
    _use_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as active:
        mlflow.set_tags({
            "git_commit": _git("rev-parse", "--short", "HEAD") or "none",
            "dirty": str(is_dirty()).lower(),
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
