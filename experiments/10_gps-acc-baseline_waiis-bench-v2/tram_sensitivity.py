"""Tram rule sensitivity (descriptive only; it can't replace the approved vehicle-mean rule): a row's saved acc and
GPS probabilities recombined with the uniform (1/8) and bus GPS tram factors, then the row's HMM. One MLflow run per
rule, compared with the row itself (vehicle mean).

    uv run python tram_sensitivity.py --row A2 --sources extrasensory shl_user1 nor_tmd collecty
Log: logs/tram_sensitivity_<row>.log
"""
from __future__ import annotations

import argparse
import json
import logging

import matplotlib.pyplot as plt
import mlflow
import numpy as np

import pipe
from tmd.models import fusion, minutes
from tmd.tracking import run

log = logging.getLogger()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True)
    ap.add_argument("--sources", nargs="+", required=True, choices=pipe.SOURCES)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(filename=pipe.HERE / "logs" / f"tram_sensitivity_{a.row}.log", filemode="w",
                        level=logging.INFO, format="%(asctime)s %(message)s")
    df = pipe.load()
    lab = pipe.labelled(a.sources)
    acc, gps = pipe.load_probs(a.row, a.seed, df, "acc"), pipe.load_probs(a.row, a.seed, df, "gps")
    base = pipe.load_probs(a.row, a.seed, df)
    assert acc.shape[1] == 9
    row = pipe.Row(a.row, tuple(a.sources))
    for rule in ("uniform", "bus"):
        p = minutes.smooth_cv(df, fusion.product(acc, gps, df.covered.to_numpy(), rule), lab, "fb")
        m, tab, figs = pipe.evaluate(df, p, {a.row: base})
        prm = {"model": "row's models, GPS tram factor replaced (descriptive)", "row": f"{a.row}_tram-{rule}",
               "seed": a.seed, "rate_hz": 25, "dataset": "+".join(a.sources), "class_set": "9class",
               "tram_rule": rule, "retrained": False, "vs": a.row}
        with run(pipe.EXP, prm, run_name=f"{a.row}_tram-{rule}_seed{a.seed}", datasets=pipe.datasets(row)):
            mlflow.log_metrics({k: float(v) for k, v in m.items() if v == v})
            for fn, fig in figs.items():
                plt.close(fig)
            mlflow.log_dict(pipe.dump(tab), "tables.json")
        log.info("%s %s", rule, json.dumps(m, default=float))
        keys = ["es_six_f1", f"es_six_f1_diff_{a.row}", "nor_guard_six_f1", "tram_pred_extrasensory_usable_n",
                "tram_pred_shl_user1_usable_n", "tram_pred_extrasensory_official_share", "nor_fine_recall_tram"]
        print(rule, {k: round(m[k], 4) for k in keys if k in m}, flush=True)


if __name__ == "__main__":
    main()
