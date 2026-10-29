"""Markdown tables of the ablation from MLflow (or, with --csv, from the exported results/runs.csv).

    uv run python tables.py [--csv]
"""
from __future__ import annotations

import argparse

import mlflow
import pandas as pd
import pipeline as pl

from tmd.tracking import TRACKING_URI

ROWS = ["A_esonly", "A", "B", "C", "D", "Dm", "E", "F_forward", "F_fb", "F_viterbi", "X_Dm_fb", "X_Dm_viterbi"]


def f(v, d=3):
    return "–" if pd.isna(v) else f"{v:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--seed", default="0")
    a = ap.parse_args()
    if a.csv:
        r = pd.read_csv(pl.HERE / "results" / "runs.csv")
    else:
        mlflow.set_tracking_uri(TRACKING_URI)
        r = mlflow.search_runs(experiment_names=[pl.EXP])
        r = r.rename(columns=lambda c: c.split(".", 1)[-1])
    r = r[(r["model"].astype(str).str.startswith("lightgbm")) & (r["model"] != "lightgbm-check")
          & (r["seed"].astype(str) == a.seed)]
    r = r.sort_values("start_time").drop_duplicates("row", keep="last").set_index("row")
    print("| row | vs | ES family F1 [CI] | Δ family vs [paired CI] | ES 8-class F1 [CI] | SHL family | Δ SHL | "
          "SHL 8-class | SHL train / subway recall | trees (mean) | single window p50 / p95 ms | pipeline p50 / p95 ms "
          "| size MB |")
    print("|" + "---|" * 13)
    for row in [x for x in ROWS if x in r.index]:
        m = r.loc[row]
        dlt = "–" if pd.isna(m.get("es_family_f1_diff")) else \
            f"{m.es_family_f1_diff:+.3f} [{m.es_family_f1_diff_lo:+.3f}, {m.es_family_f1_diff_hi:+.3f}]"
        print(f"| {row} | {m['vs'] or '–'} | {f(m.es_family_f1)} [{f(m.es_family_f1_lo)}–{f(m.es_family_f1_hi)}] | {dlt} "
              f"| {f(m.es_class_f1)} [{f(m.es_class_f1_lo)}–{f(m.es_class_f1_hi)}] | {f(m.shl_family_f1)} "
              f"| {'–' if pd.isna(m.get('shl_family_f1_diff')) else f'{m.shl_family_f1_diff:+.3f}'} "
              f"| {f(m.shl_class_f1)} | {f(m.shl_class_recall_train, 2)} / {f(m.shl_class_recall_subway, 2)} "
              f"| {f(m.n_trees_mean, 0)} | {f(m.lat_inference_ms_p50, 1)} / {f(m.lat_inference_ms_p95, 1)} "
              f"| {f(m.lat_pipeline_ms_p50, 2)} / {f(m.lat_pipeline_ms_p95, 2)} | {f(m.model_bytes_mean / 1e6, 1)} |")
    print()
    print("| row | ES family F1 per fold (min–max) | ES family F1 iPhone / Android | ES per-class recall: still walk run "
          "bike car bus train |")
    print("|---|---|---|---|")
    for row in [x for x in ROWS if x in r.index]:
        m = r.loc[row]
        rec = " ".join(f(m[f"es_class_recall_{c}"], 2) for c in ["still", "walk", "run", "bike", "car", "bus", "train"])
        print(f"| {row} | {f(m.es_family_f1_fold_min)}–{f(m.es_family_f1_fold_max)} "
              f"| {f(m.es_family_f1_iphone)} / {f(m.es_family_f1_android)} | {rec} |")


if __name__ == "__main__":
    main()
