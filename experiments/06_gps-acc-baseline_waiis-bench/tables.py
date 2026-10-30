"""Markdown tables of experiment 06 from MLflow (or, with --csv, from the exported results/runs.csv).

    uv run python tables.py [--csv] [--part ablation checks final test]
Clean runs only (dirty = false); the latest run per (row, seed).
"""
from __future__ import annotations

import argparse

import mlflow
import numpy as np
import pandas as pd
import pipe

from tmd.tracking import TRACKING_URI

ROWS = ["A", "A2", "G", "G_esonly", "H1", "H1_esgps", "H2", "H2_esgps", "I"]


def f(v, d=3):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def ci(m, k, d=3):
    return f"{f(m.get(k), d)} [{f(m.get(k + '_lo'), d)}–{f(m.get(k + '_hi'), d)}]"


def load(csv):
    if csv:
        r = pd.read_csv(pipe.HERE / "results" / "runs.csv")
    else:
        mlflow.set_tracking_uri(TRACKING_URI)
        r = mlflow.search_runs(experiment_names=[pipe.EXP])
        r = r.rename(columns=lambda c: c.split(".", 1)[-1] if c.startswith(("metrics.", "params.", "tags.git",
                                                                             "tags.dirty")) else c)
    r = r[r.dirty.astype(str).str.lower() == "false"]
    return r.sort_values("start_time").drop_duplicates(["row", "seed"], keep="last")


def ablation(r, seed="0"):
    a = r[r.seed.astype(str) == seed].set_index("row")
    print("| row | vs | ES 8-class F1 [CI] | Δ 8-class [paired CI] | ES family F1 [CI] | Δ family | ES 8-class covered / "
          "uncovered | SHL 8-class | Δ SHL 8-class | SHL train / subway | single window p50 / p95 ms | pipeline p50 / "
          "p95 ms | acc / GPS model MB |")
    print("|" + "---|" * 13)
    for row in [x for x in ROWS if x in a.index]:
        m = a.loc[row]
        d8 = "–" if pd.isna(m.get("es_class_f1_diff")) else \
            f"{m.es_class_f1_diff:+.3f} [{m.es_class_f1_diff_lo:+.3f}, {m.es_class_f1_diff_hi:+.3f}]"
        df_ = "–" if pd.isna(m.get("es_family_f1_diff")) else f"{m.es_family_f1_diff:+.3f}"
        ds = "–" if pd.isna(m.get("shl_class_f1_diff")) else f"{m.shl_class_f1_diff:+.3f}"
        mb = f"{f(m.get('acc_model_bytes_mean', np.nan) / 1e6, 1)} / {f(m.get('gps_model_bytes_mean', np.nan) / 1e6, 2)}"
        print(f"| {row} | {m['vs'] if isinstance(m['vs'], str) and m['vs'] else '–'} | {ci(m, 'es_class_f1')} | {d8} "
              f"| {ci(m, 'es_family_f1')} | {df_} | {f(m.es_class_f1_covered)} / {f(m.get('es_class_f1_uncovered'))} "
              f"| {f(m.shl_class_f1)} | {ds} | {f(m.shl_class_recall_train, 2)} / {f(m.shl_class_recall_subway, 2)} "
              f"| {f(m.lat_inference_ms_p50, 1)} / {f(m.lat_inference_ms_p95, 1)} "
              f"| {f(m.lat_pipeline_ms_p50, 2)} / {f(m.lat_pipeline_ms_p95, 2)} | {mb} |")
    print()
    print("| row | ES recall bus [CI] | car [CI] | train [CI] | run [CI] | still recall quiet / not quiet (ES) | "
          "(SHL) | ES 8-class iPhone / Android | trees (mean rounds) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for row in [x for x in ROWS if x in a.index]:
        m = a.loc[row]
        rc = " | ".join(ci(m, f"es_class_recall_{c}", 2) for c in ("bus", "car", "train", "run"))
        print(f"| {row} | {rc} | {f(m.es_still_recall_quiet, 3)} / {f(m.es_still_recall_not_quiet, 3)} "
              f"| {f(m.shl_still_recall_quiet, 3)} / {f(m.shl_still_recall_not_quiet, 3)} "
              f"| {f(m.es_class_f1_iphone)} / {f(m.es_class_f1_android)} | {f(m.rounds_mean, 1)} |")
    g = a[a.index.isin(["G", "G_esonly"])]
    if len(g):
        print()
        print("| GPS only (covered minutes) | ES 8-class | ES family | SHL 8-class | acc row A on the same minutes: ES "
              "8-class / family / SHL 8-class |")
        print("|---|---|---|---|---|")
        for row, m in g.iterrows():
            print(f"| {row} | {ci(m, 'es_class_f1')} | {ci(m, 'es_family_f1')} | {f(m.shl_class_f1)} "
                  f"| {f(m.acc_row_on_covered_es_class_f1)} / {f(m.acc_row_on_covered_es_family_f1)} "
                  f"/ {f(m.acc_row_on_covered_shl_class_f1)} |")


def checks(r):
    c = r[r.model == "lightgbm-check"].set_index("check")
    if "gps" in c.index:
        m = c.loc["gps"]
        print("| within class | dataset AUC: GPS / acc | platform AUC (ES): GPS / acc |")
        print("|---|---|---|")
        for k in ("walk", "car"):
            print(f"| {k} | {f(m[f'auc_dataset_{k}_gps'])} / {f(m[f'auc_dataset_{k}_acc'])} "
                  f"| {f(m[f'auc_platform_{k}_gps'])} / {f(m[f'auc_platform_{k}_acc'])} |")
        print(f"\ncheck 1 pass: {bool(m.check_pass)}\n")
    if "missing" in c.index:
        m = c.loc["missing"]
        print("| missingness pattern | dataset AUC | platform AUC |")
        print("|---|---|---|")
        for k in ("walk", "car", "all"):
            print(f"| {k} | {f(m[f'auc_dataset_{k}_missing'])} | {f(m[f'auc_platform_{k}_missing'])} |")
        print(f"\ncheck 2 pass (walk, car <= 0.60): {bool(m.check_pass)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--part", nargs="+", default=["ablation", "checks"])
    a = ap.parse_args()
    r = load(a.csv)
    for p in a.part:
        print(f"\n### {p}\n")
        {"ablation": ablation, "checks": checks}[p](r)


if __name__ == "__main__":
    main()
