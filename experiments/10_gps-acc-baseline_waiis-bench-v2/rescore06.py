"""Descriptive row R06: 06's final configuration as-is (H1_v2, trained on waiis_bench_v1), its saved out-of-fold
probabilities rescored on v2's labels (ExtraSensory dev, SHL User 1). No retraining. Seeds 0-2, one MLflow run each;
the seed spread of the six-class score is the context for the 0.01 thresholds (stop-point 1 answers, item 7).

    uv run python rescore06.py          # logs/rescore06.log
"""
from __future__ import annotations

import json
import logging

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

import pipe
from tmd.models import fusion
from tmd.paths import PROCESSED
from tmd.tracking import DataInput, run

P06 = PROCESSED / "06_gps-acc-baseline_waiis-bench" / "oof"
log = logging.getLogger()


def main():
    logging.basicConfig(filename=pipe.HERE / "logs" / "rescore06.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    df = pipe.load()
    v1 = fusion.load_frame("dev", "gps-v2")[["source", "user", "ts", "y", "official"]]
    hips = (df.source != pipe.OTHER).to_numpy()
    pos = pd.Series(np.flatnonzero(hips), index=pd.MultiIndex.from_frame(df.loc[hips, ["source", "user", "ts"]]))
    ix = pos.reindex(pd.MultiIndex.from_frame(v1[["source", "user", "ts"]])).to_numpy()
    assert not np.isnan(ix.astype(float)).any()
    ix = ix.astype(int)
    rows = []
    for seed in (0, 1, 2):
        z = np.load(P06 / f"H1_v2_seed{seed}.npz", allow_pickle=True)
        assert (z["user"] == v1.user.to_numpy(str)).all() and (z["ts"] == v1.ts.to_numpy()).all()
        p = np.full((len(df), 9), np.nan)
        p[ix] = pipe.pad9(z["p"])
        m, tab, figs = {}, {}, {}
        pr, tr = pipe.preds(np.nan_to_num(p)), pipe.truth(df.y.to_numpy())
        src, off = df.source.to_numpy(), df.official.to_numpy()
        sel = (src == "extrasensory") & off
        for lvl in pipe.LEVELS:
            pipe.block(m, tab, figs, "es", tr[lvl][sel], pr[lvl][sel], lvl, df.user.to_numpy()[sel])
        sel = (src == "shl_user1") & off
        for lvl in pipe.LEVELS:
            pipe.block(m, tab, figs, "shl", tr[lvl][sel], pr[lvl][sel], lvl, ci=False, fig=False)
        # the same probabilities on v1's labels (09's numbers; v1's official set)
        y1 = v1.y.to_numpy()
        es1 = ((v1.source == "extrasensory") & v1.official).to_numpy()
        m1 = {}
        pipe.block(m1, {}, {}, "es_v1labels", pipe.truth(y1[es1])["six"], pipe.preds(pipe.pad9(z["p"])[es1])["six"],
                   "six", ci=False, fig=False)
        m.update(m1)
        kw = {"label_rules": "bench-v2 (9 fine classes)", "features": "acc-v1_whole_sub5s+gps-v2_w2min",
              "rate_hz": 25, "window_s": 15, "stride_s": 60, "oof_eval": True,
              "extra": (("trained_in", "06_gps-acc-baseline_waiis-bench H1_v2 (waiis_bench_v1)"),)}
        prm = {"model": "06 H1_v2 as-is, rescored", "row": "R06", "seed": seed, "rate_hz": 25,
               "dataset": "extrasensory+shl_user1", "class_set": "8class (six-class primary, fine, family)",
               "retrained": False, "benchmark": "waiis_bench_v2 labels; models from waiis_bench_v1"}
        with run(pipe.EXP, prm, run_name=f"R06_seed{seed}",
                 datasets=[DataInput("extrasensory_dev_v2", "evaluation", **kw),
                           DataInput("shl_user1_dev", "evaluation", **kw)]):
            mlflow.log_metrics({k: float(v) for k, v in m.items() if v == v})
            for fn, fig in figs.items():
                mlflow.log_figure(fig, fn)
                plt.close(fig)
            mlflow.log_dict(pipe.dump(tab), "tables.json")
        log.info("seed %s %s", seed, json.dumps(m, default=float))
        rows.append({"seed": seed, "es_six_v2labels": m["es_six_f1"], "es_six_lo": m["es_six_f1_lo"],
                     "es_six_hi": m["es_six_f1_hi"], "es_six_v1labels": m["es_v1labels_six_f1"],
                     "es_fine": m["es_fine_f1"], "es_family": m["es_family_f1"], "shl_six": m["shl_six_f1"]})
    t = pd.DataFrame(rows).set_index("seed")
    t.loc["range"] = t.max() - t.min()
    out = pipe.HERE / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    t.round(4).to_csv(out / "rescore06.csv")
    log.info("\n%s", t.round(4).to_string())
    print(t.round(4).to_string())


if __name__ == "__main__":
    main()
