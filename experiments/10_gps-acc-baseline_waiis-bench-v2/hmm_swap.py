"""Descriptive (stop-point 2 decisions, item 5; no retraining): A1's fused probabilities smoothed with A0's transition
matrix (labelled minutes of ExtraSensory + SHL User 1, 9 states; tram's row and column are pseudo-counts only), to
separate the HMM's share of A1's ExtraSensory drop. One MLflow run, A1_hmmA0_seed0, compared with A0 and A1.

    uv run python hmm_swap.py           # logs/hmm_swap.log
"""
from __future__ import annotations

import json
import logging

import matplotlib.pyplot as plt
import mlflow

import pipe
from tmd.models import minutes
from tmd.tracking import run

log = logging.getLogger()


def main():
    logging.basicConfig(filename=pipe.HERE / "logs" / "hmm_swap.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    df = pipe.load()
    raw = pipe.load_probs("A1", 0, df, "raw")
    p = minutes.smooth_cv(df, raw, pipe.labelled(("extrasensory", "shl_user1")), "fb")
    m, tab, figs = pipe.evaluate(df, p, {"A0": pipe.load_probs("A0", 0, df), "A1": pipe.load_probs("A1", 0, df)})
    row = pipe.Row("A1_hmmA0", ("extrasensory", "shl_user1", "nor_tmd"))
    prm = {"model": "A1's models, HMM transitions of A0 (descriptive)", "row": "A1_hmmA0", "seed": 0, "rate_hz": 25,
           "dataset": "extrasensory+shl_user1+nor_tmd", "class_set": "9class", "retrained": False,
           "hmm_sources": "extrasensory,shl_user1", "vs": "A0,A1"}
    with run(pipe.EXP, prm, run_name="A1_hmmA0_seed0", datasets=pipe.datasets(row)):
        mlflow.log_metrics({k: float(v) for k, v in m.items() if v == v})
        for fn, fig in figs.items():
            plt.close(fig)
        mlflow.log_dict(pipe.dump(tab), "tables.json")
    log.info("%s", json.dumps(m, default=float))
    keys = ["es_six_f1", "es_six_f1_diff_A0", "es_six_f1_diff_A0_lo", "es_six_f1_diff_A0_hi", "es_six_f1_diff_A1",
            "es_six_f1_diff_A1_lo", "es_six_f1_diff_A1_hi", "nor_guard_six_f1", "nor_guard_six_f1_diff_A1",
            "nor_guard_six_f1_diff_A1_lo", "nor_guard_six_f1_diff_A1_hi", "es_six_precision_rail"]
    print({k: round(m[k], 4) for k in keys if k in m})


if __name__ == "__main__":
    main()
