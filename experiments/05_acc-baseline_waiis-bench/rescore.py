"""Rescore saved ablation rows with the fixed user bootstrap (resamples lacking a class are redrawn).

Reads the out-of-fold probabilities saved by the original (dirty, commit 5852704) runs, kept in oof/dirty_5852704/,
recomputes every development metric, copies the metrics that don't depend on the bootstrap (latency, size, rounds)
from the original MLflow run, and logs one run per row: <row>_seed0_ci-redraw.

    uv run python rescore.py
"""
from __future__ import annotations

import json
import logging

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pipeline as pl

from tmd.tracking import TRACKING_URI, run

ROWS = {"A": None, "A_esonly": "A", "B": "A", "C": "B", "D": "B", "Dm": "B", "E": "B", "F_forward": "B",
        "F_fb": "B", "F_viterbi": "B", "X_Dm_fb": "B", "X_Dm_viterbi": "B"}
SRC = pl.OOF / "dirty_5852704"
COPY = ("model_bytes_mean", "n_trees_mean", "rounds_mean", "rounds_min", "rounds_max", "n_features",
        "lat_inference_ms_p50", "lat_inference_ms_p95", "lat_features_ms_p50", "lat_features_ms_p95",
        "lat_model_ms_p50", "lat_model_ms_p95", "lat_pipeline_ms_p50", "lat_pipeline_ms_p95", "lat_pipeline_sessions")
log = logging.getLogger()


def main():
    logging.basicConfig(filename=pl.HERE / "logs" / "rescore.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    mlflow.set_tracking_uri(TRACKING_URI)
    orig = mlflow.search_runs(experiment_names=[pl.EXP], filter_string="tags.git_commit = '5852704'")
    df = pl.load(pl.Cfg("keys"))
    oof = {}
    for row in ROWS:
        z = np.load(SRC / f"{row}_seed0.npz")
        assert (z["user"] == df.user.to_numpy(str)).all() and (z["ts"] == df.ts.to_numpy()).all()
        oof[row] = (z["p"], json.loads(str(z["cfg"])))
    for row, vs in ROWS.items():
        o = orig[orig["tags.mlflow.runName"] == f"{row}_seed0"]
        assert len(o) == 1, row
        o = o.iloc[0]
        m, figs, tab = pl.evaluate(df, oof[row][0], oof[vs][0] if vs else None)
        m.update({k: float(o[f"metrics.{k}"]) for k in COPY})
        params = {k[len("params."):]: v for k, v in o.items() if k.startswith("params.") and v is not None}
        params.update({"bootstrap": "users, redraw resamples lacking a class", "rescore_of": o.run_id,
                       "rescore_copied": "latency, size, rounds from rescore_of"})
        cfg = pl.Cfg(**oof[row][1])
        with run(pl.EXP, params, run_name=f"{row}_seed0_ci-redraw", datasets=pl.datasets(pl.feature_tag(cfg))):
            mlflow.log_metrics({k: float(v) for k, v in m.items() if v == v})
            for fn, fig in figs.items():
                mlflow.log_figure(fig, fn)
                plt.close(fig)
            mlflow.log_dict(tab, "tables.json")
        log.info("%s %s", row, json.dumps(m))
        print(row, {k: round(m[k], 4) for k in ("es_family_f1", "es_family_f1_lo", "es_family_f1_hi",
                                                "es_family_ci_redraw_rate") if k in m}, flush=True)


if __name__ == "__main__":
    main()
