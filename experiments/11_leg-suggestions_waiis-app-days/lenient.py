"""Stop-point 3 (replaces the manual leg review, 2026-11-05): automatic, lenient comparison of the leg suggestions
with the diaries. Diary modes are taken as correct, diary times as approximate. Descriptive; no rule or threshold
changes.

- Core minutes of a diary leg: drop 3 min at each end if its times are marked GT (both) or GT start, 5 min otherwise;
  a leg too short for that keeps its middle minute. Only core minutes are scored, so boundary error never counts.
- Scored minutes: the accelerometer minute windows of the day (the model minutes in the output files), the same set
  for every model. A minute's predicted mode = the mode of the model's leg containing the minute centre (unknown is
  never right); for the package (`fusion_late__hmm_viterbi`), its leg at that centre (none: not right).
- Per diary leg and model: right = the majority of its core minutes with data have the diary's top-level mode;
  found = at least one does; no data = no core minute has accelerometer data (an app recording gap); wrong = data
  but not right.
- Extra vehicle legs: predicted bike / car / bus / rail legs (model legs, not GPS-only) that overlap no diary vehicle
  leg, even with each diary leg widened by 10 min on each side.
- Walking was not tracked from 2026-10-31 on: the listed walking legs are scored; walking or still predictions are
  never counted as extra.

    uv run python lenient.py        # log: logs/lenient.log; MLflow one run per model (lenient_<model>)
"""
import json
import logging

import mlflow
import numpy as np
import pandas as pd

import dev
import review as R
from tmd.readers import waiis
from tmd.tracking import DataInput, run

log = logging.getLogger()
VEHICLE = {"bike", "car", "bus", "rail"}
OTHER = {"walking", "running", "still"}
TRIM_GT_S, TRIM_S, WIDEN_S = 180.0, 300.0, 600.0
MODELS = ("E2", "E1", "package")


def core(r) -> tuple[float, float, bool]:
    """(lo, hi, middle-minute only) of a diary leg's core."""
    trim = TRIM_GT_S if r.gt in ("both", "start") else TRIM_S
    if r.end - r.start > 2 * trim:
        return r.start + trim, r.end - trim, False
    mid = (r.start + r.end) / 2
    return mid, mid, True


def mode_at(legs: pd.DataFrame, t: np.ndarray) -> np.ndarray:
    out = np.full(len(t), "none", dtype=object)
    for a, b, m in zip(legs.start, legs.end, legs["mode"]):
        out[(t >= a) & (t < b)] = m
    return out


def main():
    logging.basicConfig(filename=dev.HERE / "logs" / "lenient.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    rows, extras = [], []
    for day in waiis.days():
        d = R.diary(day.date)
        base = R.OUT / f"user={day.user}" / f"date={day.date}"
        docs = {m: json.loads((base / f"legs_{m}.json").read_text()) for m in ("E2", "E1")}
        ts = np.array([x["t_ms"] for x in docs["E2"]["minutes"]], float) / 1000      # accelerometer minutes
        assert np.array_equal(ts, np.array([x["t_ms"] for x in docs["E1"]["minutes"]], float) / 1000)
        centre = ts + 30
        legs = {m: pd.DataFrame([{"start": x["start_ms"] / 1000, "end": x["end_ms"] / 1000, "mode": x["mode"],
                                  "source": "model" if x["model_id"] == m else x["model_id"]}
                                 for x in docs[m]["legs"]]) for m in ("E2", "E1")}
        legs["package"] = R.package_legs(day.path)[["start", "end", "mode", "source"]].assign(source="model")
        pred = {m: mode_at(legs[m], centre) for m in MODELS}
        for r in d.itertuples():
            lo, hi, mid_only = core(r)
            sel = ((ts <= lo) & (ts + 60 > lo)) if mid_only else ((centre >= lo) & (centre <= hi))
            for m in MODELS:
                p = pred[m][sel]
                n_ok = int((p == r.mode).sum())
                n = int(sel.sum())
                status = "no data" if n == 0 else "right" if n_ok > n / 2 else "wrong"
                rows.append({"date": day.date, "diary_id": r.diary_id, "start_local": r.start_local,
                             "end_local": r.end_local, "mode": r.mode, "sub_mode": r.sub_mode, "gt": r.gt,
                             "core_min": round((hi - lo) / 60, 1) if not mid_only else 1.0, "model": m,
                             "minutes_with_data": n, "minutes_right": n_ok, "status": status,
                             "found": n_ok > 0,
                             "predicted": pd.Series(p).value_counts().to_dict() if n else {}})
        dv = d[d["mode"].isin(VEHICLE)]
        for m in MODELS:
            lv = legs[m][(legs[m].source == "model") & legs[m]["mode"].isin(VEHICLE)]
            for a, b, md in zip(lv.start, lv.end, lv["mode"]):
                hit = ((dv.start - WIDEN_S < b) & (dv.end + WIDEN_S > a)).any()
                if not hit:
                    extras.append({"date": day.date, "model": m, "mode": md,
                                   "start_local": pd.to_datetime(a, unit="s", utc=True).tz_convert(R.TZ)
                                   .strftime("%H:%M"),
                                   "end_local": pd.to_datetime(b, unit="s", utc=True).tz_convert(R.TZ)
                                   .strftime("%H:%M"), "minutes": round((b - a) / 60, 1)})
    x = pd.DataFrame(rows)
    ex = pd.DataFrame(extras, columns=["date", "model", "mode", "start_local", "end_local", "minutes"])
    out = R.OUT / "review"
    x.to_csv(out / "lenient_per_leg.csv", index=False)
    ex.to_csv(out / "lenient_extra_vehicle_legs.csv", index=False)

    summ = {}
    for m in MODELS:
        met = {}
        for grp, modes in (("vehicle", VEHICLE), ("other", OTHER)):
            g = x[(x.model == m) & x["mode"].isin(modes)]
            dat = g[g.status != "no data"]
            met.update({f"{grp}_legs": len(g), f"{grp}_right": int((g.status == "right").sum()),
                        f"{grp}_found": int(g.found.sum()), f"{grp}_wrong": int((g.status == "wrong").sum()),
                        f"{grp}_no_data": int((g.status == "no data").sum()),
                        f"{grp}_core_minutes_with_data": int(dat.minutes_with_data.sum()),
                        f"{grp}_core_minute_accuracy": float(dat.minutes_right.sum() / max(dat.minutes_with_data.sum(), 1))})
            for md in sorted(modes):
                gm = g[g["mode"] == md]
                if len(gm):
                    met[f"{md}_legs"], met[f"{md}_right"] = len(gm), int((gm.status == "right").sum())
                    met[f"{md}_no_data"] = int((gm.status == "no data").sum())
        met["extra_vehicle_legs"] = int((ex.model == m).sum())
        summ[m] = met
        prm = {"model": m if m != "package" else "package fusion_late__hmm_viterbi (descriptive)",
               "model_version": {"E2": "10:test_E2_seed0", "E1": "10:test_E1_seed0", "package": "WAIIS package"}[m],
               "dataset": "waiis_app_days", "rate_hz": 25, "class_set": "top level", "seed": 0,
               "step": "lenient_diary_comparison", "core_trim_s": f"GT/GT start {TRIM_GT_S:g}, else {TRIM_S:g}",
               "extra_widen_s": WIDEN_S, "minutes_scored": "accelerometer minute windows (same set for all)"}
        ds = [DataInput("waiis_app_days", "evaluation", "diary modes, core minutes only (lenient)",
                        "leg suggestions (output files)", window_s=15, stride_s=60)]
        with run(dev.EXP, prm, run_name=f"lenient_{m}", datasets=ds):
            mlflow.log_metrics({k: float(v) for k, v in met.items()})
        log.info("%s %s", m, json.dumps(met))

    # paired E1 vs E2 on vehicle legs with data
    v = x[x["mode"].isin(VEHICLE)].pivot_table(index=["date", "diary_id", "start_local", "end_local", "mode",
                                                       "sub_mode"], columns="model", values="status", aggfunc="first")
    v = v[(v.E1 != "no data") & (v.E2 != "no data")].reset_index()
    v["pair"] = np.select([(v.E1 == "right") & (v.E2 == "right"), (v.E1 == "right"), (v.E2 == "right")],
                          ["both right", "only E1", "only E2"], "neither")
    v.to_csv(out / "lenient_paired_vehicle.csv", index=False)
    s = pd.DataFrame(summ).T
    tabs = dev.HERE / "results" / "tables"
    tabs.mkdir(parents=True, exist_ok=True)
    s.to_csv(tabs / "lenient_summary.csv")
    print(s[[c for c in s.columns if c.startswith(("vehicle", "other", "extra"))]].to_string())
    print("\nper mode:", s[[c for c in s.columns if c.split("_")[0] in VEHICLE | OTHER]].to_string())
    print("\npaired (vehicle legs with data in both):", v.pair.value_counts().to_dict())
    print(v[["date", "start_local", "end_local", "mode", "sub_mode", "E2", "E1", "package", "pair"]].to_string(index=False))
    print("\nextra vehicle legs:")
    print(ex.to_string(index=False))


if __name__ == "__main__":
    main()
