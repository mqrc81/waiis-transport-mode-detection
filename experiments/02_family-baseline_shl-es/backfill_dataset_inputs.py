"""Add MLflow dataset inputs to the existing runs of experiment 02 (metadata only).

Only `MlflowClient.log_inputs` is called: metrics, params, tags and artifacts are not touched. Runs that already have
inputs are skipped. Dry run by default; `--apply` writes.

    uv run python backfill_dataset_inputs.py            # table of run -> inputs, writes nothing
    uv run python backfill_dataset_inputs.py --apply
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess

import mlflow
from mlflow import MlflowClient

from tmd.tracking import REPO, TRACKING_URI, DataInput, check_inputs

EXP = "02_family-baseline_shl-es"
R1 = "c615aedd62da40159a0b65113820aad1"
R2 = "d668fa2e279d4776a65bc7e8f2f4977a"
CTX = "3a47f096d01a4b078c4590fd73870fe9"
F1 = "feat-v1"
STEP3 = (("windows", "all, incl. unlabelled (context/HMM)"),)


def shl(context, oof=False, features=F1, extra=()):
    return DataInput("shl_user1_dev", context, label_rules="shl-coarse", features=features, oof_eval=oof, extra=extra)


def es(context, rules, oof=False, features=F1, extra=()):
    return DataInput("extrasensory_dev", context, label_rules=rules, features=features, oof_eval=oof, extra=extra)


def by_run(trained_in: str) -> tuple[tuple[str, str], ...]:
    return (("trained_in_run", trained_in),)


# Run name -> inputs. From each run's params (dataset, setup, es_labels, features, base_models / source_run).
# ES label rules "v1" = the step-1 rules (es_labels param absent).
PLAN = {
    "R1_lgbm_feat-v1_seed0": [shl("training", oof=True), es("evaluation", "v1")],
    # Rescoring runs: no training in the run; ExtraSensory is evaluation only. SHL is listed because the scored
    # models were trained on it, in run R1.
    "R1_lgbm_feat-v1_seed0_es-labels-v2": [shl("training", extra=by_run(R1)), es("evaluation", "v2")],
    "R1_lgbm_feat-v1_seed0_es-v2-foldmatched": [shl("training", extra=by_run(R1)), es("evaluation", "v2")],
    "R2_lgbm_feat-v1_seed0": [shl("training", oof=True), es("training", "v2", oof=True)],
    "ESonly_lgbm_feat-v1_seed0": [es("training", "v2", oof=True), shl("evaluation")],
}
for cfg in ("base", "base+fwd", "base+vit"):
    PLAN[f"step3_{cfg}_feat-v1_seed0"] = [shl("training", oof=True, extra=by_run(R2) + STEP3),
                                          es("training", "v2", oof=True, extra=by_run(R2) + STEP3)]
PLAN["step3_ctx_feat-v1_seed0"] = [shl("training", oof=True, features="feat-v1+ctx60s", extra=STEP3),
                                   es("training", "v2", oof=True, features="feat-v1+ctx60s", extra=STEP3)]
for cfg in ("ctx+fwd", "ctx+vit"):
    PLAN[f"step3_{cfg}_feat-v1_seed0"] = [
        shl("training", oof=True, features="feat-v1+ctx60s", extra=by_run(CTX) + STEP3),
        es("training", "v2", oof=True, features="feat-v1+ctx60s", extra=by_run(CTX) + STEP3)]


def split_unchanged(path: str, commit: str) -> bool:
    """The split file at the run's commit equals today's (the digest describes what the run used)."""
    old = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=REPO, capture_output=True).stdout
    return hashlib.sha256(old).digest() == hashlib.sha256((REPO / path).read_bytes()).digest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true")
    a = p.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    runs = client.search_runs([mlflow.get_experiment_by_name(EXP).experiment_id], order_by=["start_time ASC"])
    names = {r.info.run_name for r in runs}
    if names != set(PLAN):
        raise SystemExit(f"runs without a plan: {names - set(PLAN)}; plan without a run: {set(PLAN) - names}")

    for r in runs:
        inputs = PLAN[r.info.run_name]
        check_inputs(inputs)
        commit = r.data.tags["git_commit"]
        for d in inputs:
            if not split_unchanged(d.split, commit):
                raise SystemExit(f"{d.split} changed since {commit} ({r.info.run_name})")
        done = len(r.inputs.dataset_inputs)
        status = f"skip, has {done} inputs" if done else ("logged" if a.apply else "would log")
        print(f"{r.info.run_id[:8]}  {r.info.run_name:42s}  {commit}  {status}")
        for d in inputs:
            ds = d.dataset()
            more = "  ".join(f"{k}={v}" for k, v in d.tags().items() if k not in ("split", "rate_hz", "window_s",
                                                                                "stride_s"))
            print(f"    {ds.name:17s} {ds.digest}  {d.context:10s}  {more}")
        if a.apply and not done:
            client.log_inputs(r.info.run_id, datasets=[d.entity() for d in inputs])
    print("rate 25 Hz, window 5 s, stride 2.5 s and the split path are tagged on every input."
          + ("" if a.apply else "\nDry run: nothing written."))


if __name__ == "__main__":
    main()
