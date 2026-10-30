"""Markdown tables of experiment 06 from MLflow (or, with --csv, from the exported results/runs.csv).

    uv run python tables.py [--csv] [--part ablation checks final test]
Clean runs only (dirty = false), except the test runs (see load); the latest run per (row, seed).
"""
from __future__ import annotations

import argparse

import mlflow
import numpy as np
import pandas as pd
import pipe

from tmd.tracking import TRACKING_URI

ROWS = ["A", "A2", "G", "G_esonly", "H1", "H1_esgps", "H2", "H2_esgps", "I", "G_v2", "H1_v2"]


def f(v, d=3):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


f_ = f        # final() uses a local frame named f


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
    # clean runs only, except the test runs (scored once; tagged dirty by test.py's own results/gps_coverage_test.csv)
    r = r[(r.dirty.astype(str).str.lower() == "false") | (r.row == "test")]
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
    c = r[r.model == "lightgbm-check"].set_index("row")
    rows = [(n, v) for n, v in (("check_gps", "gps-v1"), ("check_gps_gps-v2", "gps-v2")) if n in c.index]
    if rows:
        print("| check 1, within class | " + " | ".join(f"dataset AUC {v}" for _, v in rows) + " | dataset AUC acc | "
              + " | ".join(f"platform AUC {v}" for _, v in rows) + " | platform AUC acc |")
        print("|---|" + "---|" * (2 * len(rows) + 2))
        for k in ("walk", "car"):
            m0 = c.loc[rows[0][0]]
            print(f"| {k} | " + " | ".join(f(c.loc[n][f"auc_dataset_{k}_gps"]) for n, _ in rows)
                  + f" | {f(m0[f'auc_dataset_{k}_acc'])} | "
                  + " | ".join(f(c.loc[n][f"auc_platform_{k}_gps"]) for n, _ in rows)
                  + f" | {f(m0[f'auc_platform_{k}_acc'])} |")
        print("\ncheck 1 pass: " + ", ".join(f"{v} {bool(c.loc[n].check_pass)}" for n, v in rows) + "\n")
    if "check_missing" in c.index:
        m = c.loc["check_missing"]
        print("| missingness pattern (gps-v1) | dataset AUC | platform AUC |")
        print("|---|---|---|")
        for k in ("walk", "car", "all"):
            print(f"| {k} | {f(m[f'auc_dataset_{k}_missing'])} | {f(m[f'auc_platform_{k}_missing'])} |")
        print(f"\ncheck 2 pass (walk, car <= 0.60): {bool(m.check_pass)}")


def final(r):
    """A and H1_v2 (the final configuration) on dev, seeds 0-2."""
    f = r[r.row.isin(["A", "H1_v2"])].copy()
    f["seed"] = f.seed.astype(str)
    print("| row | seed | ES 8-class F1 [CI] | gain vs A [paired CI] | ES family F1 [CI] | 8-class covered / uncovered "
          "| iPhone / Android | SHL 8-class | SHL train / subway | commit |")
    print("|" + "---|" * 10)
    for row in ("A", "H1_v2"):
        g = f[f.row == row].sort_values("seed")
        for _, m in g.iterrows():
            gain = "–" if pd.isna(m.get("es_class_f1_diff")) else \
                f"{m.es_class_f1_diff:+.3f} [{m.es_class_f1_diff_lo:+.3f}, {m.es_class_f1_diff_hi:+.3f}]"
            print(f"| {row} | {m.seed} | {ci(m, 'es_class_f1')} | {gain} | {ci(m, 'es_family_f1')} "
                  f"| {f_(m.es_class_f1_covered)} / {f_(m.es_class_f1_uncovered)} | {f_(m.es_class_f1_iphone)} / "
                  f"{f_(m.es_class_f1_android)} | {f_(m.shl_class_f1)} | {f_(m.shl_class_recall_train, 2)} / "
                  f"{f_(m.shl_class_recall_subway, 2)} | {m.git_commit} |")
        gm = "" if row == "A" else f"{g.es_class_f1_diff.mean():+.3f}"
        print(f"| {row} | mean | {f_(g.es_class_f1.mean())} | {gm} | {f_(g.es_family_f1.mean())} | | | "
              f"{f_(g.shl_class_f1.mean())} | | |")
    print()
    print("| row | seed | bus [CI] | car [CI] | train [CI] | run [CI] | still recall quiet / not quiet |")
    print("|---|---|---|---|---|---|---|")
    for row in ("A", "H1_v2"):
        for _, m in f[f.row == row].sort_values("seed").iterrows():
            rc = " | ".join(ci(m, f"es_class_recall_{c}", 2) for c in ("bus", "car", "train", "run"))
            print(f"| {row} | {m.seed} | {rc} | {f_(m.es_still_recall_quiet)} / {f_(m.es_still_recall_not_quiet)} |")


def test(r):
    t = r[r.row == "test"].copy()
    if t.empty:
        print("(no test runs)")
        return
    t["seed"] = t.seed.astype(str)
    t = t.sort_values("seed")

    def line(label, key, gain=False):
        if key not in t.columns:
            return
        vals = [(f"{m[key]:+.3f} [{m[key + '_lo']:+.3f}, {m[key + '_hi']:+.3f}]" if gain else ci(m, key))
                if key + "_lo" in t.columns else f_(m[key]) for _, m in t.iterrows()]
        mean = t[key].mean()
        print(f"| {label} | **{mean:+.3f}** | " if gain else f"| {label} | **{f_(mean)}** | ", end="")
        print(" | ".join(vals) + " |")

    print("| test, ExtraSensory 12 users | mean of 3 seeds | seed 0 | seed 1 | seed 2 |")
    print("|---|---|---|---|---|")
    for scope in ("all", "iphone", "android"):
        line(f"{scope}: 8-class F1 (H1_v2)", f"test_es_{scope}_class_f1")
        line(f"{scope}: 8-class F1, A (05)", f"test_es_{scope}_A_class_f1")
        line(f"{scope}: 8-class gain over A", f"test_es_{scope}_class_gain_vs_A", gain=True)
        line(f"{scope}: family F1 (H1_v2)", f"test_es_{scope}_family_f1")
        line(f"{scope}: family F1, A", f"test_es_{scope}_A_family_f1")
        line(f"{scope}: family gain over A", f"test_es_{scope}_family_gain_vs_A", gain=True)
        line(f"{scope}: 8-class F1, minutes with GPS", f"test_es_{scope}_class_f1_covered")
        line(f"{scope}: 8-class F1, minutes without GPS", f"test_es_{scope}_class_f1_uncovered")
        line(f"{scope}: A, minutes with GPS", f"test_es_{scope}_A_class_f1_covered")
        line(f"{scope}: A, minutes without GPS", f"test_es_{scope}_A_class_f1_uncovered")
    for c in ("still", "walk", "run", "bike", "car", "bus", "train"):
        line(f"all: recall {c} (H1_v2)", f"test_es_all_class_recall_{c}")
        line(f"all: recall {c}, A", f"test_es_all_A_class_recall_{c}")
    for nm, k in (("rail recall before HMM", "test_es_all_raw_family_recall_rail"),
                  ("rail recall after HMM", "test_es_all_family_recall_rail"),
                  ("train recall before HMM", "test_es_all_raw_class_recall_train"),
                  ("train recall after HMM", "test_es_all_class_recall_train")):
        line(f"all: {nm}", k)
    print()
    print("| SHL Users 2–3 (no CI) | mean | seed 0 | seed 1 | seed 2 |")
    print("|---|---|---|---|---|")
    for nm, k in (("8-class F1 (H1_v2)", "test_shl23_class_f1"), ("8-class F1, A", "test_shl23_A_class_f1"),
                  ("8-class gain over A", "test_shl23_class_gain_vs_A"), ("family F1", "test_shl23_family_f1"),
                  ("family F1, A", "test_shl23_A_family_f1"), ("8-class before HMM", "test_shl23_raw_class_f1")) + \
            tuple((f"recall {c}", f"test_shl23_class_recall_{c}") for c in
                  ("still", "walk", "run", "bike", "car", "bus", "train", "subway")) + \
            (("train recall, A", "test_shl23_A_class_recall_train"), ("subway recall, A", "test_shl23_A_class_recall_subway")):
        if k in t.columns:
            print(f"| {nm} | {f_(t[k].mean())} | " + " | ".join(f_(v) for v in t[k]) + " |")
    print()
    print("| seed | acc / GPS model MB | GPS rounds | single window p50 / p95 ms | pipeline p50 / p95 ms | CI redraw (all / Android / iPhone, 8-class) |")
    print("|---|---|---|---|---|---|")
    for _, m in t.iterrows():
        print(f"| {m.seed} | {m.acc_model_bytes / 1e6:.2f} / {m.gps_model_bytes / 1e6:.2f} | {f_(m.gps_rounds, 0)} "
              f"| {f_(m.lat_inference_ms_p50, 1)} / {f_(m.lat_inference_ms_p95, 1)} | {f_(m.lat_pipeline_ms_p50, 2)} / "
              f"{f_(m.lat_pipeline_ms_p95, 2)} | {f_(m.test_es_all_class_ci_redraw_rate)} / "
              f"{f_(m.test_es_android_class_ci_redraw_rate)} / {f_(m.test_es_iphone_class_ci_redraw_rate)} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--part", nargs="+", default=["ablation", "checks", "final", "test"])
    a = ap.parse_args()
    r = load(a.csv)
    for p in a.part:
        print(f"\n### {p}\n")
        {"ablation": ablation, "checks": checks, "final": final, "test": test}[p](r)


if __name__ == "__main__":
    main()

