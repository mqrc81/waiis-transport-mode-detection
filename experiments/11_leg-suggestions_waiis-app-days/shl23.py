"""Stop-point 3, once: SHL Users 2-3 (reserved; all positions), leg metrics with the approved rules, descriptive.

Models as saved (10's final E2 / E1, seed 0; prediction only). Two GPS inputs: 10's per-minute GPS (as in 10's test)
and the approved WAIIS rule C1 on the 1 Hz track thinned to 20 s. First checks that the saved models reproduce 10's
saved test probabilities (per-minute GPS) for these rows. Legs: E2 s_veh 0, s_other 3, g 10, l_min 2, c 0.5; E1 raw
legs, c 0.5. Truth: every labelled minute (v2 fine labels, mixed minutes by majority), top level.

    uv run python shl23.py          # log: logs/shl23.log; MLflow one run per model
"""
import json
import logging

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

import dev
from tmd import bench
from tmd import gps as tgps
from tmd import legs as L
from tmd import suggest as S
from tmd.features import gps as gf
from tmd.labels import FINE_ID
from tmd.models import fusion, minutes
from tmd.tracking import DataInput, run

log = logging.getLogger()
TABLES = dev.HERE / "results" / "tables"


def frames() -> pd.DataFrame:
    d = []
    for name, src in (("shl_user23", "shl_user23"), ("shl_user23_other", "shl_user23_other")):
        a = pd.read_parquet(dev.P10 / f"{name}_test_25hz_w15s_acc-v1-B.parquet")
        g = pd.read_parquet(dev.P10 / f"{name}_test_gps-v2_w2min.parquet")
        assert (a.user.to_numpy() == g.user.to_numpy()).all() and (a.ts.to_numpy() == g.ts.to_numpy()).all()
        a = pd.concat([a, g[dev.GPS]], axis=1).assign(source=src)
        if "position" not in a:
            a["position"] = "hips"
        a["session_int"] = a.session.astype(int)
        if src == "shl_user23_other":
            a["session"] = a.session.astype(str) + "/" + a.position          # one HMM sequence per position (10)
        d.append(a)
    return d


def truth(df: pd.DataFrame) -> np.ndarray:
    lab = bench.read("shl_user23", "labels")
    c = lab.cls_fine.astype(str).where(lab.cls_fine.astype(str) != "", lab.majority_cls.astype(str))
    c = c.where(~(lab.label_excluded | lab.run_user_flagged), "")
    lab = lab.assign(y8=c.map(FINE_ID).fillna(0).astype(int))[["user", "ts", "y8"]]
    y = df[["user", "ts"]].merge(lab, on=["user", "ts"], how="left", validate="m:1").y8.fillna(0).astype(int)
    return L.fine_to_top(y.to_numpy())


def c1_features(df_hips: pd.DataFrame) -> pd.DataFrame:
    """gps-v2 from the 1 Hz track thinned to 20 s, rule C1, per (user, ts) of the hips frame."""
    full = bench.shl_gps_1hz("shl_user23", include_reserved=True).sort_values(["user", "rec", "time"])
    full = full.reset_index(drop=True)
    keep = [g.index.to_numpy()[tgps.thin(g.time.to_numpy(), np.full(len(g), 20.0))]
            for _, g in full.groupby(["user", "rec"], sort=False)]
    tr = full.loc[np.concatenate(keep)]
    tr = tr[tr.accuracy <= 500].merge(bench.read("shl_user23", "labels", columns=["user", "ts", "session"]),
                                      on=["user", "ts"], how="inner")
    keys = df_hips[["user", "session_int", "ts"]].rename(columns={"session_int": "session"})
    x = gf.features(keys, tgps.nearest_per_minute(tr), dev.GPS_VERSION)
    log.info("SHL 2-3 1 Hz %d -> 20 s %d fixes; C1 covered hips minutes %.3f", len(full), len(tr),
             gf.covered(x).mean())
    return pd.concat([df_hips[["user", "ts"]].reset_index(drop=True), pd.DataFrame(x, columns=dev.GPS)], axis=1)


def probs(spec: S.Spec, df: pd.DataFrame, x_gps: np.ndarray) -> np.ndarray:
    acc, gps = lgb.Booster(model_file=str(spec.acc)), lgb.Booster(model_file=str(spec.gps))
    cov = gf.covered(x_gps)
    raw = fusion.product(acc.predict(df[dev.ACC].to_numpy(np.float32)), gps.predict(x_gps), cov)
    h = np.load(spec.hmm)
    return dev.pad9(minutes.smooth(df, raw, h["a"], h["pi"], "fb")), cov


def legs_metrics(df: pd.DataFrame, p: np.ndarray, rules: L.Rules) -> dict:
    seq = minutes.seq_ids(df)
    order = np.lexsort((df.ts.to_numpy(), seq))
    q = L.top_probs(p[order])
    s, t = seq[order], df.ts.to_numpy(np.int64)[order]
    tr = truth(df)[order]
    day = df.day.astype(str).to_numpy()[order]
    lg = L.build(s, t, q, rules)
    sc = L.score(lg, L.truth_legs(s, t, tr), len(q), q, day)
    pm = sc.per_mode.set_index("mode")
    m = dict(sc.metrics)
    for md in ("rail", "bus", "car", "still", "walking"):
        if md in pm.index:
            m[f"leg_recall_{md}"] = pm.leg_recall[md]
            m[f"time_recall_{md}"] = pm.time_recall[md]
    return m


def main():
    logging.basicConfig(filename=dev.HERE / "logs" / "shl23.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    hips, other = frames()
    c1 = c1_features(hips)
    rows = []
    for mid, spec in S.SPECS.items():
        z = np.load(dev.OOF10 / f"test_{mid}_seed0.npz", allow_pickle=True)
        met = {}
        for df in (hips, other):
            src = df.source.iat[0]
            sel = z["source"] == src
            assert (z["user"][sel] == df.user.to_numpy(str)).all() and (z["ts"][sel] == df.ts.to_numpy()).all()
            p_min, _ = probs(spec, df, df[dev.GPS].to_numpy(np.float32))
            diff = float(np.abs(p_min - z["p"][sel]).max())
            log.info("%s %s per-minute GPS reproduces 10's saved test p: max |diff| %.2e", mid, src, diff)
            assert diff < 1e-5, diff
            xg = df[["user", "ts"]].merge(c1, on=["user", "ts"], how="left")[dev.GPS].to_numpy(np.float32)
            p_c1, cov = probs(spec, df, xg)
            for gname, p in (("minute", p_min), ("c1", p_c1)):
                for pos, g in df.groupby("position"):
                    idx = g.index.to_numpy()
                    m = legs_metrics(g.reset_index(drop=True), p[idx], spec.rules)
                    rows.append({"model": mid, "gps": gname, "position": pos, **m})
                    for k in ("leg_f1", "edits_per_h", "time_acc", "wrong_unflagged_time", "boundary_median_min",
                              "leg_recall_rail", "time_recall_rail", "flagged_leg_share"):
                        if k in m:
                            met[f"{gname}_{pos}_{k}"] = float(m[k])
        prm = {"model": spec.id, "model_version": spec.version, "dataset": "shl_user23 (reserved; all positions)",
               "rate_hz": 25, "class_set": "top level", "seed": 0, "leg_rules": spec.rules.name(),
               "gps_inputs": "10's per-minute; C1 at 20 s", "step": "shl23_once"}
        ds = [DataInput("shl_user23_test_v3", "evaluation", "bench-v2 (9 fine classes) -> tool top level",
                        "acc-v1-B + gps-v2 (per minute; C1 20 s)", window_s=15, stride_s=60)]
        with run(dev.EXP, prm, run_name=f"shl23_{mid}", datasets=ds):
            mlflow.log_metrics({k: v for k, v in met.items() if v == v})
    r = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    r.to_csv(TABLES / "shl23_legs.csv", index=False)
    cols = ["model", "gps", "position", "hours", "truth_legs", "leg_f1", "edits_per_h", "time_acc",
            "wrong_unflagged_time", "boundary_median_min", "boundary_p90_min", "leg_recall_rail", "time_recall_rail",
            "flagged_leg_share"]
    print(r[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
