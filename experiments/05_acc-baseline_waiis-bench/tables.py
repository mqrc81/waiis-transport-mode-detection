"""Markdown tables of experiment 05 from MLflow (or, with --csv, from the exported results/runs.csv).

    uv run python tables.py [--csv] [--part ablation final test]
Runs are selected by params (the export has no run names). Ablation: the rescored rows (param rescore_of; fixed
bootstrap). Final: clean B / F_fb runs, seeds 0-2. Test: row = test, seeds 0-2.
"""
from __future__ import annotations

import argparse

import mlflow
import numpy as np
import pandas as pd
import pipeline as pl

from tmd.tracking import TRACKING_URI

ROWS = ["A_esonly", "A", "B", "C", "D", "Dm", "E", "F_forward", "F_fb", "F_viterbi", "X_Dm_fb", "X_Dm_viterbi"]
CLS = ["still", "walk", "run", "bike", "car", "bus", "train", "subway"]


def f(v, d=3):
    return "–" if v is None or pd.isna(v) else f"{v:.{d}f}"


def ci(m, k, d=3):
    return f"{f(m.get(k), d)} [{f(m.get(k + '_lo'), d)}–{f(m.get(k + '_hi'), d)}]"


def load(csv: bool) -> pd.DataFrame:
    if csv:
        r = pd.read_csv(pl.HERE / "results" / "runs.csv")
    else:
        mlflow.set_tracking_uri(TRACKING_URI)
        r = mlflow.search_runs(experiment_names=[pl.EXP])
        r = r.rename(columns=lambda c: c.split(".", 1)[-1] if c.startswith(("metrics.", "params.", "tags.git", "tags.dirty")) else c)
    return r


def ablation(r):
    a = r[r.rescore_of.notna()].sort_values("start_time").drop_duplicates("row", keep="last")
    a = a.set_index("row")
    print("| row | vs | ES family F1 [CI] | Δ family vs [paired CI] | ES 8-class F1 [CI] | SHL family | Δ SHL | "
          "SHL 8-class | SHL train / subway recall | trees (mean) | single window p50 / p95 ms | pipeline p50 / p95 ms "
          "| size MB |")
    print("|" + "---|" * 13)
    for row in [x for x in ROWS if x in a.index]:
        m = a.loc[row]
        dlt = "–" if pd.isna(m.get("es_family_f1_diff")) else \
            f"{m.es_family_f1_diff:+.3f} [{m.es_family_f1_diff_lo:+.3f}, {m.es_family_f1_diff_hi:+.3f}]"
        sd = "–" if pd.isna(m.get("shl_family_f1_diff")) else f"{m.shl_family_f1_diff:+.3f}"
        print(f"| {row} | {m['vs'] if isinstance(m['vs'], str) and m['vs'] else '–'} | {ci(m, 'es_family_f1')} | {dlt} "
              f"| {ci(m, 'es_class_f1')} | {f(m.shl_family_f1)} | {sd} | {f(m.shl_class_f1)} "
              f"| {f(m.shl_class_recall_train, 2)} / {f(m.shl_class_recall_subway, 2)} | {f(m.n_trees_mean, 0)} "
              f"| {f(m.lat_inference_ms_p50, 1)} / {f(m.lat_inference_ms_p95, 1)} "
              f"| {f(m.lat_pipeline_ms_p50, 2)} / {f(m.lat_pipeline_ms_p95, 2)} | {f(m.model_bytes_mean / 1e6, 1)} |")
    print()
    print("| row | ES family F1 per fold (min–max) | ES family F1 iPhone / Android | ES recall still walk run bike car "
          "bus train | ES rail recall | CI redraw rate (family / class) |")
    print("|---|---|---|---|---|---|")
    for row in [x for x in ROWS if x in a.index]:
        m = a.loc[row]
        rec = " ".join(f(m[f"es_class_recall_{c}"], 2) for c in CLS[:7])
        print(f"| {row} | {f(m.es_family_f1_fold_min)}–{f(m.es_family_f1_fold_max)} "
              f"| {f(m.es_family_f1_iphone)} / {f(m.es_family_f1_android)} | {rec} | {f(m.es_family_recall_rail, 2)} "
              f"| {f(m.es_family_ci_redraw_rate)} / {f(m.es_class_ci_redraw_rate)} |")


def final(r):
    fr = r[(r.row.isin(["B", "F_fb"])) & (r.dirty.astype(str).str.lower() == "false")
           & r.rescore_of.isna()].sort_values("start_time")
    fr = fr.drop_duplicates(["row", "seed"], keep="last")
    print("| row | seed | ES family F1 [CI] | ES 8-class F1 [CI] | ES car / bus / train recall [CI] | ES rail recall "
          "| SHL family | SHL 8-class | SHL train / subway | trees (mean) | commit |")
    print("|" + "---|" * 11)
    for row in ["B", "F_fb"]:
        for _, m in fr[fr.row == row].sort_values("seed").iterrows():
            cbt = " / ".join(ci(m, f"es_class_recall_{c}", 2) for c in ("car", "bus", "train"))
            print(f"| {row} | {m.seed} | {ci(m, 'es_family_f1')} | {ci(m, 'es_class_f1')} | {cbt} "
                  f"| {f(m.es_family_recall_rail, 2)} | {f(m.shl_family_f1)} | {f(m.shl_class_f1)} "
                  f"| {f(m.shl_class_recall_train, 2)} / {f(m.shl_class_recall_subway, 2)} | {f(m.n_trees_mean, 0)} "
                  f"| {m.git_commit} |")
        s = fr[fr.row == row]
        print(f"| {row} | mean (range) | {f(s.es_family_f1.mean())} ({f(s.es_family_f1.min())}–{f(s.es_family_f1.max())}) "
              f"| {f(s.es_class_f1.mean())} ({f(s.es_class_f1.min())}–{f(s.es_class_f1.max())}) | | "
              f"| {f(s.shl_family_f1.mean())} | {f(s.shl_class_f1.mean())} | | | |")


def test(r):
    t = r[r.row == "test"].sort_values("seed")
    if t.empty:
        print("(no test runs)")
        return
    def line(label, key):
        vals = [ci(m, key) for _, m in t.iterrows()]
        mean = np.mean([m[key] for _, m in t.iterrows()])
        print(f"| {label} | **{f(mean)}** | " + " | ".join(vals) + " |")
    print("| test (B + HMM fb) | mean of 3 seeds | seed 0 [CI] | seed 1 [CI] | seed 2 [CI] |")
    print("|---|---|---|---|---|")
    for scope in ("all", "iphone", "android"):
        for lvl in ("family", "class"):
            line(f"ES {scope} {lvl} macro-F1", f"test_es_{scope}_{lvl}_f1")
            line(f"ES {scope} {lvl} macro-F1, before HMM", f"test_es_{scope}_raw_{lvl}_f1")
    for scope in ("all", "android"):
        line(f"ES {scope} rail recall, before HMM", f"test_es_{scope}_raw_family_recall_rail")
        line(f"ES {scope} rail recall, after HMM", f"test_es_{scope}_family_recall_rail")
        line(f"ES {scope} train recall, before HMM", f"test_es_{scope}_raw_class_recall_train")
        line(f"ES {scope} train recall, after HMM", f"test_es_{scope}_class_recall_train")
    for c in CLS[:7]:
        line(f"ES all recall {c}", f"test_es_all_class_recall_{c}")
    print()
    print("| SHL Users 2–3 (no CI) | mean | seed 0 | seed 1 | seed 2 |")
    print("|---|---|---|---|---|")
    keys = [("family macro-F1", "test_shl23_family_f1"), ("8-class macro-F1", "test_shl23_class_f1"),
            ("family macro-F1, before HMM", "test_shl23_raw_family_f1"),
            ("8-class macro-F1, before HMM", "test_shl23_raw_class_f1")]
    keys += [(f"recall {c}", f"test_shl23_class_recall_{c}") for c in CLS]
    keys += [("rail recall, before / after HMM", None)]
    for label, key in keys:
        if key is None:
            vals = [f"{f(m.test_shl23_raw_family_recall_rail, 2)} / {f(m.test_shl23_family_recall_rail, 2)}"
                    for _, m in t.iterrows()]
            print(f"| {label} | | " + " | ".join(vals) + " |")
            continue
        vals = [f(m[key]) for _, m in t.iterrows()]
        print(f"| {label} | {f(t[key].mean())} | " + " | ".join(vals) + " |")
    print()
    print("| seed | trees | model MB | single window p50 / p95 ms | pipeline p50 / p95 ms | CI redraw rate ES all family / iPhone family |")
    print("|---|---|---|---|---|---|")
    for _, m in t.iterrows():
        print(f"| {m.seed} | {f(m.n_trees, 0)} | {f(m.model_bytes / 1e6, 2)} | {f(m.lat_inference_ms_p50, 1)} / "
              f"{f(m.lat_inference_ms_p95, 1)} | {f(m.lat_pipeline_ms_p50, 2)} / {f(m.lat_pipeline_ms_p95, 2)} "
              f"| {f(m.test_es_all_family_ci_redraw_rate)} / {f(m.test_es_iphone_family_ci_redraw_rate)} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--part", nargs="+", default=["ablation", "final", "test"])
    a = ap.parse_args()
    r = load(a.csv)
    for p in a.part:
        print(f"\n### {p}\n")
        {"ablation": ablation, "final": final, "test": test}[p](r)


if __name__ == "__main__":
    main()
