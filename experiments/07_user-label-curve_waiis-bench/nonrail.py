"""Experiment 07 (review): macro-F1 over the 6 non-rail classes (still, walk, run, bike, car, bus) for both curves,
from each run's logged 8-class confusion matrix (MLflow artifact tables.json), and how much of SHL User 1's advantage
is train. No new runs.

Per-class F1 comes from the full 8 × 8 matrix (rail predictions on non-rail minutes count against precision); the
6-class score is their mean. Cross-checked against the per-class F1 logged in results/runs.csv.

    uv run python nonrail.py        # results/nonrail.csv, results/nonrail.md
"""
from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from tables import CLS7, ORDER, cell, load, summary
from tmd.labels import CLASSES
from tmd.tracking import TRACKING_URI

HERE = Path(__file__).resolve().parent
SIX = ["still", "walk", "run", "bike", "car", "bus"]


def f1_from_cm(cm: np.ndarray) -> dict[str, float]:
    tp, pred, true = np.diag(cm), cm.sum(0), cm.sum(1)
    f = np.where(pred + true > 0, 2 * tp / np.maximum(pred + true, 1), np.nan)
    return dict(zip(CLASSES, f))


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    r = load(HERE / "results" / "runs.csv")
    r = r[r.curve.isin(["main", "noshl"])]
    uri = mlflow.search_runs(experiment_names=[HERE.name]).set_index("run_id").artifact_uri
    rows = []
    for _, x in r.iterrows():
        cm = np.array(mlflow.artifacts.load_dict(f"{uri[x.run_id]}/tables.json")["cm_class"])
        f = f1_from_cm(cm)
        for c in CLS7:                                     # cross-check with the logged per-class F1
            assert abs(f[c] - x[f"es_class_f1_{c}"]) < 1e-9, (x.run_id, c)
        assert abs(np.mean([f[c] for c in CLS7]) - x.es_class_f1) < 1e-9
        rows.append({"curve": x.curve, "n": x.n, "draw": x.draw, "f1_6": float(np.mean([f[c] for c in SIX])),
                     "f1_7": x.es_class_f1, "f1_train": f["train"]})
    t = pd.DataFrame(rows)
    t.to_csv(HERE / "results" / "nonrail.csv", index=False)
    main_, noshl = t[t.curve == "main"], t[t.curve == "noshl"]
    L = ["**Macro-F1 over the 6 non-rail classes** (still, walk, run, bike, car, bus; from the logged confusion "
         "matrices; median [range] over draws), against the 8-class score (7 classes)", "",
         "| n | 6-class, with SHL User 1 | 6-class, without | gap 6-class | 8-class, with | 8-class, without | gap "
         "8-class | train F1 with / without | share of the 8-class gap from train |", "|---|---|---|---|---|---|---|---|---|"]
    for n in ORDER:
        a = main_[main_.n == n]
        b = noshl[noshl.n == n]
        if b.empty:
            L.append(f"| {n} | {cell(summary(a, 'f1_6').loc[n])} | – | – | {cell(summary(a, 'f1_7').loc[n])} | – | – "
                     f"| {a.f1_train.median():.3f} / – | – |")
            continue
        g6, g7 = a.f1_6.median() - b.f1_6.median(), a.f1_7.median() - b.f1_7.median()
        gt = a.f1_train.median() - b.f1_train.median()
        share = gt / 7 / g7                                # 8-class = (6 · F1_6 + F1_train) / 7
        L.append(f"| {n} | {cell(summary(a, 'f1_6').loc[n])} | {cell(summary(b, 'f1_6').loc[n])} | {g6:+.3f} | "
                 f"{cell(summary(a, 'f1_7').loc[n])} | {cell(summary(b, 'f1_7').loc[n])} | {g7:+.3f} | "
                 f"{a.f1_train.median():.3f} / {b.f1_train.median():.3f} | {share:.0%} |")
    L += ["", "Gaps are differences of medians. 8-class = (6 × 6-class + train F1) / 7, so the train share of the "
          "8-class gap = (train F1 gap / 7) / 8-class gap; the rest is the 6 non-rail classes.", ""]
    md = "\n".join(L) + "\n"
    (HERE / "results" / "nonrail.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
