"""Descriptive, no retraining (stop-point 2 approval, item 4): each row's acc model alone + the row's HMM (no GPS
fusion), from the saved seed-0 out-of-fold probabilities, against the row's fused output. Fusion gain = fused −
acc-only, paired user-bootstrap CI, on all ExtraSensory dev minutes and on GPS-covered ones; walk → still minutes
with and without fusion. One MLflow run per row (`<row>_acconly_seed0`).

    uv run python fusion_gain.py            # results/tables/fusion_gain.csv; logs/fusion_gain.log
"""
from __future__ import annotations

import json
import logging

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

import pipe
from tmd.models import minutes
from tmd.tracking import run

ES, SHL, NOR, COL = "extrasensory", "shl_user1", "nor_tmd", "collecty"
ROWS = {"A0": ((ES, SHL), (ES, SHL), 8), "A1": ((ES, SHL, NOR), (ES, SHL, NOR), 9),
        "A2": ((ES, SHL, COL), (ES, SHL, COL), 9), "E1": ((ES, SHL, NOR), (ES, SHL), 9),
        "E2": ((ES, SHL, COL), (ES, SHL, COL), 8)}          # row: (training sources, HMM sources, classes)
log = logging.getLogger()


def gain(m, k):
    """fused − acc-only from the acc-only run's 'diff vs fused' (acc − fused): sign and CI flipped."""
    d = f"{k}_diff_fused"
    return -m[d], -m[d + "_hi"], -m[d + "_lo"]


def main():
    logging.basicConfig(filename=pipe.HERE / "logs" / "fusion_gain.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    df = pipe.load()
    es = ((df.source == ES) & df.official).to_numpy()
    cov = df.covered.to_numpy()
    y = df.y.to_numpy()
    out = []
    for r, (srcs, hmm_srcs, n) in ROWS.items():
        p_acc = pipe.pad9(minutes.smooth_cv(df, pipe.load_probs(r, 0, df, "acc"), pipe.labelled(hmm_srcs, n), "fb"))
        p_fused = pipe.load_probs(r, 0, df)
        m, tab, figs = pipe.evaluate(df, p_acc, {"fused": p_fused})
        sel = es & cov                                     # ExtraSensory dev, GPS-covered minutes only
        tr, pa, pf = pipe.truth(y), pipe.preds(p_acc), pipe.preds(p_fused)
        for lvl in ("six", "fine"):
            pipe.block(m, tab, figs, "es_cov", tr[lvl][sel], pa[lvl][sel], lvl, df.user.to_numpy()[sel],
                       {"fused": pf[lvl][sel]}, fig=False)
        walk = es & (y == 2)
        for nm, pr in (("acconly", pa), ("fused", pf)):
            m[f"es_walk_to_still_{nm}"] = float((pr["fine"][walk] == 1).sum())
            m[f"es_walk_to_still_{nm}_covered"] = float((pr["fine"][walk & cov] == 1).sum())
        row = pipe.Row(f"{r}_acconly", srcs, n, hmm_sources=hmm_srcs)
        prm = {"model": "row's acc model + HMM, no GPS fusion (descriptive)", "row": f"{r}_acconly", "seed": 0,
               "rate_hz": 25, "dataset": "+".join(srcs), "class_set": f"{n}class", "retrained": False,
               "hmm_sources": ",".join(hmm_srcs), "vs": f"{r} (fused)"}
        with run(pipe.EXP, prm, run_name=f"{r}_acconly_seed0", datasets=pipe.datasets(row)):
            mlflow.log_metrics({k: float(v) for k, v in m.items() if v == v})
            for fig in figs.values():
                plt.close(fig)
            mlflow.log_dict(pipe.dump(tab), "tables.json")
        log.info("%s %s", r, json.dumps(m, default=float))
        rec = {"row": r}
        for k, nm in (("es_six_f1", "es_six"), ("es_fine_f1", "es_9"), ("es_cov_six_f1", "es_six_covered"),
                      ("es_cov_fine_f1", "es_9_covered")):
            g, lo, hi = gain(m, k)
            rec.update({f"{nm}_acconly": m[k], f"{nm}_gain": g, f"{nm}_gain_lo": lo, f"{nm}_gain_hi": hi})
        rec.update({"nor_guard_acconly": m["nor_guard_six_f1"], "nor_guard_gain": -m["nor_guard_six_f1_diff_fused"],
                    "shl_six_acconly": m["shl_six_f1"], "shl_six_gain": -m["shl_six_f1_diff_fused"],
                    "walk_to_still_acconly": m["es_walk_to_still_acconly"],
                    "walk_to_still_fused": m["es_walk_to_still_fused"], "walk_minutes": int(walk.sum())})
        for c in ("still", "foot", "bike", "car", "bus", "rail"):
            rec[f"prec_{c}_acconly"] = m.get(f"es_six_precision_{c}", np.nan)
            rec[f"rec_{c}_acconly"] = m.get(f"es_six_recall_{c}", np.nan)
        out.append(rec)
        print(r, {k: round(v, 4) for k, v in rec.items() if k != "row" and "prec" not in k and "rec_" not in k},
              flush=True)
    t = pd.DataFrame(out).set_index("row")
    t.round(4).to_csv(pipe.HERE / "results" / "tables" / "fusion_gain.csv")


if __name__ == "__main__":
    main()
