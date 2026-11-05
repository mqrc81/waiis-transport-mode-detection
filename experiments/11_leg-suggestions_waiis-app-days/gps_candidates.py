"""Stop-point 2, step 1: how WAIIS's ~20 s fixes become E2's GPS input. SHL User 1's 1 Hz track thinned to a fixed
20 s cadence per recording; candidate rules give one fix (or position) per minute, or every fix (C5); gps-v2
features; E2's GPS model out of fold (A0_gps fold models), fused with E2's saved out-of-fold acc probabilities, HMM
per fold. E1 (same GPS model) is scored alongside, descriptive. References: 10's per-minute GPS (SHL's training rule
on the 1 Hz track) and no GPS.

Choice (11's Plan): highest six-class macro-F1 on SHL User 1 official minutes after fusion + HMM, seed 0; within
0.005 of the best, more GPS-covered minutes, then the simpler rule (C1 < C3 < C2 < C4).

    uv run python gps_candidates.py           # log: logs/gps_candidates.log; MLflow one run per rule and seed
"""
import json
import logging

import mlflow
import numpy as np
import pandas as pd
from tmd import bench
from tmd.evaluate import cm_metrics, six_pred
from tmd.features import gps as gf
from tmd.labels import to_score6
from tmd.progress import Progress
from tmd.tracking import DataInput, run

import dev

log = logging.getLogger()
CADENCE_S = 20.0
RULES = ["C1", "C2", "C3", "C4", "C5", "ref_minute_1hz", "no_gps"]
SIMPLE = {"C1": 0, "C3": 1, "C2": 2, "C4": 3, "C5": 4}
DESC = {"C1": "fix nearest the minute centre, within the minute",
        "C2": "best-accuracy fix within +-7.5 s of the minute centre (ties: nearest the centre)",
        "C3": "best-accuracy fix within the minute (ties: nearest the centre)",
        "C4": "median lat / lon of the minute's fixes, median time",
        "C5": "every fix, each in its clock minute",
        "ref_minute_1hz": "10's per-minute GPS (best accuracy within +-7.5 s on the 1 Hz track; training rule)",
        "no_gps": "acc model alone + HMM"}


def thinned_track(keys: pd.DataFrame) -> pd.DataFrame:
    """SHL User 1's 1 Hz fixes thinned per recording to a fixed 20 s cadence, accuracy > 500 m dropped, with the
    session of their minute. Moved to `tmd.models.e10.shl_thinned_track` in 12 (2026-11-05)."""
    return dev.shl_thinned_track(keys)


def per_minute(tr: pd.DataFrame, rule: str) -> pd.DataFrame:
    """One row per (user, session, ts) with time, lat, lon by the rule; C5: every fix."""
    tr = tr.assign(dc=np.abs(tr.time - (tr.ts + 30.0)))
    if rule == "C5":
        return tr[["user", "session", "ts", "time", "lat", "lon"]]
    if rule == "C1":
        pick = tr.sort_values(["user", "session", "ts", "dc"])
    elif rule == "C2":
        pick = tr[tr.dc <= 7.5].sort_values(["user", "session", "ts", "accuracy", "dc"])
    elif rule == "C3":
        pick = tr.sort_values(["user", "session", "ts", "accuracy", "dc"])
    elif rule == "C4":
        return tr.groupby(["user", "session", "ts"], as_index=False).agg(
            time=("time", "median"), lat=("lat", "median"), lon=("lon", "median"))
    return pick.drop_duplicates(["user", "session", "ts"])[["user", "session", "ts", "time", "lat", "lon"]]


def gps_features(df: pd.DataFrame, tr: pd.DataFrame | None, rule: str) -> np.ndarray:
    if rule == "no_gps":
        return np.full((len(df), len(dev.GPS)), np.nan, np.float32)
    if rule == "ref_minute_1hz":
        return df[dev.GPS].to_numpy(np.float32)
    return gf.features(df[["user", "session", "ts"]], per_minute(tr, rule), dev.GPS_VERSION)


def six(y, pr):
    cm = np.zeros((6, 6), np.int64)
    np.add.at(cm, (to_score6(y) - 1, pr - 1), 1)
    return cm_metrics(cm, np.isin(np.arange(1, 7), np.unique(to_score6(y))))


def main():
    logging.basicConfig(filename=dev.HERE / "logs" / "gps_candidates.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    prog = Progress(dev.EXP, "gps_candidates", len(RULES) * len(dev.SEEDS), log)
    df = dev.frame("shl_user1")
    off, y, fold = df.official.to_numpy(), df.y.to_numpy(), df.fold.to_numpy()
    ld = dev.leg_data(df)
    labs = {n: dev.labelled(m.hmm_sources, m.n) for n, m in dev.MODELS.items()}
    tr = thinned_track(df)
    # sanity: 10's per-minute features from the bench table equal its saved ones
    ref = gf.features(df[["user", "session", "ts"]], bench.read("shl_user1", "gps_per_minute"), dev.GPS_VERSION)
    same = np.allclose(ref, df[dev.GPS].to_numpy(np.float32), equal_nan=True, atol=1e-5)
    log.info("10's SHL User 1 GPS features recomputed equal: %s", same)
    assert same
    dev.OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for rule in RULES:
        x = gps_features(df, tr, rule)
        cov = gf.covered(x)
        np.save(dev.OUT / f"shl_user1_gps_{rule}.npy", x)
        for seed in dev.SEEDS:
            p_gps = dev.predict_oof(dev.fold_models("A0_gps", seed), x, fold)
            m = {"covered_official": float(cov[off].mean()), "covered_all": float(cov.mean())}
            g = six(y[off & cov], six_pred(p_gps[off & cov]))
            m["gps_alone_six_f1_covered"] = g["macro_f1"]
            for name, mdl in dev.MODELS.items():
                pre = "" if name == "E2" else "e1_"
                p_acc = dev.saved(name, seed, df, "acc")
                raw, p = dev.fuse_smooth(mdl, df, p_acc, p_gps, cov, labs[name])
                s = six(y[off], six_pred(p[off]))
                m[f"{pre}six_f1"] = s["macro_f1"]
                for i, c in enumerate(["still", "foot", "bike", "car", "bus", "rail"]):
                    m[f"{pre}six_recall_{c}"] = float(s["recall"][i])
                m[f"{pre}six_f1_before_hmm"] = six(y[off], six_pred(dev.pad9(raw)[off]))["macro_f1"]
                _, sc = dev.leg_eval(ld, p)
                for k in ("leg_f1", "time_acc", "edits_per_h", "boundary_median_min", "boundary_p90_min",
                          "wrong_unflagged_time", "extra_legs_per_h", "missed_legs_per_h"):
                    m[f"{pre}rawlegs_{k}"] = sc.metrics[k]
                np.save(dev.OUT / f"shl_user1_{name}_seed{seed}_{rule}_p.npy", p.astype(np.float32))
            prm = {"model": "E2 (10, saved fold models) + E1 descriptive", "dataset": "shl_user1 (hips, day folds)",
                   "rate_hz": 25, "class_set": "8class (six-class primary)", "seed": seed, "gps_rule": rule,
                   "gps_rule_desc": DESC[rule], "gps_cadence_s": CADENCE_S if rule.startswith("C") else "",
                   "gps_features": dev.GPS_VERSION, "gps_model": "A0_gps (10)", "step": "gps_candidates"}
            ds = [DataInput("shl_user1_dev", "evaluation", "bench-v2 (9 fine classes)",
                            f"acc-v1-B + {dev.GPS_VERSION} ({rule})", window_s=15, stride_s=60, oof_eval=True)]
            with run(dev.EXP, prm, run_name=f"gps_{rule}_seed{seed}", datasets=ds):
                mlflow.log_metrics({k: float(v) for k, v in m.items() if v == v})
            rows.append({"rule": rule, "seed": seed, **m})
            log.info("%s seed %d %s", rule, seed, json.dumps({k: round(v, 4) for k, v in m.items()}))
            prog.step(f"{rule} seed {seed}")
    res = pd.DataFrame(rows)
    out = dev.HERE / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "gps_candidates.csv", index=False)
    s0 = res[res.seed == 0].set_index("rule")
    spread = res.groupby("rule").six_f1.agg(["min", "max"])
    cand = s0.loc[[r for r in RULES if r.startswith("C")]]
    best = cand.six_f1.max()
    near = cand[cand.six_f1 >= best - 0.005].copy()
    near["simple"] = near.index.map(SIMPLE)
    chosen = near.sort_values(["covered_official", "simple"], ascending=[False, True]).index[0]
    cols = ["six_f1", "e1_six_f1", "gps_alone_six_f1_covered", "covered_official", "rawlegs_edits_per_h",
            "rawlegs_leg_f1"]
    print(s0[cols].join(spread.rename(columns={"min": "six_f1_seed_min", "max": "six_f1_seed_max"})).round(4).to_string())
    print(f"chosen: {chosen} (best six-class F1 {best:.4f}; within 0.005: {list(near.index)})")
    log.info("chosen %s", chosen)
    (out / "gps_choice.json").write_text(json.dumps({"chosen": chosen, "best_six_f1_seed0": best,
                                                     "within_0.005": list(near.index)}, indent=1))


if __name__ == "__main__":
    main()
