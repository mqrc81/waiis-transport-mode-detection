"""Step 2 follow-up: ExtraSensory label rules v2 (sleeping dropped, foot/bike not carried dropped, resting-phone still
capped at 20% per user) and R1 rescored on the ExtraSensory dev users with them. No retraining: R1 never trains on
ExtraSensory, and v2 only removes minutes.

Usage: uv run python rescore_es_labels.py [--seed 0]
Logs an eval-only MLflow run; writes results/audit/es_label_rules_v2_counts.csv (all 60 users, labels only).
"""
import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from tmd import extrasensory as es
from tmd.evaluate import bootstrap_users, family_proba, plot_cm, scores
from tmd.features import FEATURES_VERSION
from tmd.labels import CLASSES, FAMILIES, to_family
from tmd.paths import PROCESSED
from tmd.tracking import run

HERE = Path(__file__).resolve().parent
EXP = HERE.name
TAG = f"25hz_w5s_s2.5s_feat-{FEATURES_VERSION}"
SOURCE_RUN = "c615aedd62da40159a0b65113820aad1"     # R1_lgbm_feat-v1_seed0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    # Counts under v1 vs v2, all users (labels only).
    mins = es.minutes_table(es.uuids())
    keep = es.cap_resting_still(mins[mins.reason == "ok"], seed=a.seed)
    ok = mins[mins.reason == "ok"].assign(capped_out=~keep)
    counts = pd.DataFrame({
        "v2_ok": ok.groupby("cls").size(),
        "v2_after_cap": ok[~ok.capped_out].groupby("cls").size(),
        "drop_sleeping": mins[mins.reason == "drop:SLEEPING"].groupby("cls").size(),
        "drop_not_carried": mins[mins.reason == "drop:not_carried"].groupby("cls").size(),
        "resting_share_before_cap": ok.groupby("cls").resting.mean(),
        "resting_share_after_cap": ok[~ok.capped_out].groupby("cls").resting.mean(),
    }).fillna(0)
    counts.to_csv(HERE / "results" / "audit" / "es_label_rules_v2_counts.csv", float_format="%.3f")
    print(counts.round(3).to_string())

    # Rescore R1 on the dev windows that survive v2.
    win = pd.read_parquet(PROCESSED / f"es_dev_{TAG}.parquet", columns=["user", "ts", "platform", "position", "y", "reason"])
    proba = pd.read_parquet(PROCESSED / EXP / f"r1_{TAG}_seed{a.seed}" / "es_dev_proba.parquet").to_numpy()
    v2 = ok.assign(keep=~ok.capped_out)[["user", "ts", "resting", "keep"]]
    w = win.merge(v2, on=["user", "ts"], how="left", validate="many_to_one")
    in_v2 = w.keep.fillna(False).to_numpy(bool) & (w.reason == "ok").to_numpy()
    resting_still_all = (w.resting.fillna(False).to_numpy(bool) & (w.y == 1).to_numpy() & w.keep.notna().to_numpy())
    p8, pf = proba.argmax(1) + 1, family_proba(proba).argmax(1) + 1
    y, users = w.y.to_numpy(), w.user.to_numpy()

    metrics, figs, tables = {}, {}, {}
    groups = [("es", in_v2), ("es_carried_still_only", in_v2 & ~((y == 1) & resting_still_all)),
              ("es_resting_still_uncapped", resting_still_all)] + \
             [(f"es_{p}", in_v2 & (w.position == p).to_numpy()) for p in ["in_pocket", "in_hand", "in_bag", "on_table", "none"]] + \
             [(f"es_{p}", in_v2 & (w.platform == p).to_numpy()) for p in ["iphone", "android"]]
    for name, sel in groups:
        for lvl, yy, pp in [("class", y[sel], p8[sel]), ("family", to_family(y[sel]), pf[sel])]:
            s = scores(yy, pp, lvl)
            names = CLASSES if lvl == "class" else FAMILIES
            metrics.update({f"{name}_{lvl}_macro_f1": s["macro_f1"], f"{name}_{lvl}_n": s["n"]})
            metrics.update({f"{name}_{lvl}_recall_{n}": s["recall"][n] for n in names if s["recall"][n] == s["recall"][n]})
            tables[f"{name}_{lvl}_cm"] = s["cm"].tolist()
            if name == "es":
                figs[f"cm_es_v2_{lvl}.png"] = plot_cm(s["cm"], lvl, f"R1 on ES dev, label rules v2, {lvl}")
                metrics[f"es_{lvl}_macro_f1_ci_lo"], metrics[f"es_{lvl}_macro_f1_ci_hi"] = \
                    bootstrap_users(yy, pp, users[sel], lvl)
    fam = to_family(y)
    for src, fid in [("road", 4), ("still", 1)]:
        sel = in_v2 & (fam == fid)
        metrics[f"es_rail_fp_rate_on_{src}"] = float((pf[sel] == 5).mean())
    metrics["es_still_resting_share"] = float(resting_still_all[in_v2 & (y == 1)].mean())

    params = {"model": "lightgbm", "dataset": "shl_u1", "eval": "es_dev_labels_v2", "setup": "R1", "rate_hz": 25,
              "class_set": "8class+family", "seed": a.seed, "features": FEATURES_VERSION, "es_labels": "v2",
              "resting_still_max_share": es.RESTING_STILL_MAX_SHARE, "source_run": SOURCE_RUN, "retrained": False}
    with run(EXP, params, run_name=f"R1_lgbm_feat-{FEATURES_VERSION}_seed{a.seed}_es-labels-v2"):
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})
        for fn, fig in figs.items():
            mlflow.log_figure(fig, fn)
        mlflow.log_dict(tables, "tables.json")
    keys = ["es_class_macro_f1", "es_family_macro_f1", "es_family_macro_f1_ci_lo", "es_family_macro_f1_ci_hi",
            "es_family_recall_still", "es_rail_fp_rate_on_still", "es_rail_fp_rate_on_road", "es_still_resting_share",
            "es_carried_still_only_family_macro_f1", "es_resting_still_uncapped_family_recall_still"]
    print({k: round(metrics[k], 3) for k in keys})


if __name__ == "__main__":
    main()
