"""11's lenient diary comparison for every model on the three WAIIS days: the engine's E1 + R1 (tmd.suggest.SPECS),
E2 (11's setting, no gap rule) and every model in the day folders' predictions/ (the package; `point_quality_rules_v1`
has no modes). Descriptive. Scored minutes as 11: the accelerometer minute windows of the day (same set for all
models), core minutes of each diary leg, mode of the model's leg at the minute centre. Extra for the legs without
accelerometer data: whether a model shows rail at any clock-minute centre of the leg's core (GPS-only models can).

    uv run python lenient_all.py     # MLflow one run per model; personal per-leg table under <processed>/12_.../waiis/
"""
import dataclasses
import sys

import mlflow
import numpy as np
import pandas as pd

from tmd import legs as L
from tmd import suggest as S
from tmd.models.e10 import PROCESSED
from tmd.readers import waiis
from tmd.tracking import DataInput, run

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "11_leg-suggestions_waiis-app-days"))
import lenient as LN  # noqa: E402
import review as R  # noqa: E402

EXP = "12_gps-gap-rail_shl-waiis"
OUT = PROCESSED / EXP / "waiis" / "tables"
PKG = ["fusion_late__hmm_viterbi", "inceptiontime_acc__hmm_viterbi", "inceptiontime_acc_25hz__hmm_viterbi",
       "hart_acc__hmm_viterbi", "lightgbm_gps__hmm_viterbi", "rules_speed__majority_window"]
OURS = {"E1 + R1": S.SPECS["E1"], "E2": dataclasses.replace(S.SPECS["E2"], gap=None)}


def main():
    rows, extras = [], []
    for day in waiis.days():
        dia = R.diary(day.date)
        d_in = S.day_input(day)
        ts = d_in.minutes.ts.to_numpy(float)
        centre = ts + 30
        legs = {}
        for name, spec in OURS.items():
            lg = S.suggest(spec, d_in)
            legs[name] = lg.assign(mode=["unknown" if m < 0 else L.MODES[m] for m in lg["mode"]])
        for p in PKG:
            legs[p] = R.package_legs(day.path, p)
        dveh = dia[dia["mode"].isin(LN.VEHICLE)]
        for name, lg in legs.items():
            pred = LN.mode_at(lg[["start", "end", "mode"]], centre)
            for r in dia.itertuples():
                lo, hi, mid_only = LN.core(r)
                sel = ((ts <= lo) & (ts + 60 > lo)) if mid_only else ((centre >= lo) & (centre <= hi))
                p = pred[sel]
                n, n_ok = int(sel.sum()), int((p == r.mode).sum())
                cc = np.arange(np.floor(lo / 60) * 60, hi + 1, 60) + 30
                cc = cc[(cc >= lo - 30) & (cc <= hi + 30)]
                any_mode = set(LN.mode_at(lg[["start", "end", "mode"]], cc)) if len(cc) else set()
                rows.append({"date": day.date, "diary_id": r.diary_id, "mode": r.mode, "sub_mode": r.sub_mode,
                             "model": name, "minutes_with_data": n, "minutes_right": n_ok,
                             "status": "no data" if n == 0 else "right" if n_ok > n / 2 else "wrong",
                             "mode_shown_anywhere_in_core": r.mode in any_mode})
            src = lg.source.to_numpy() if "source" in lg else np.full(len(lg), "package")
            for a, b, m, s in zip(lg.start, lg.end, lg["mode"], src):
                if s in ("model", "package") and m in LN.VEHICLE and not (
                        (dveh.start - LN.WIDEN_S < b) & (dveh.end + LN.WIDEN_S > a)).any():
                    extras.append({"date": day.date, "model": name, "mode": m, "minutes": round((b - a) / 60, 1)})
    x, ex = pd.DataFrame(rows), pd.DataFrame(extras, columns=["date", "model", "mode", "minutes"])
    OUT.mkdir(parents=True, exist_ok=True)
    x.to_csv(OUT / "lenient_all_models_per_leg.csv", index=False)
    summ = {}
    for name in list(OURS) + PKG:
        g = x[x.model == name]
        v, o = g[g["mode"].isin(LN.VEHICLE)], g[g["mode"].isin(LN.OTHER)]
        vd, od = v[v.status != "no data"], o[o.status != "no data"]
        nd = v[(v.status == "no data") & (v["mode"] == "rail")]
        m = {"vehicle_right": int((v.status == "right").sum()), "vehicle_wrong": int((v.status == "wrong").sum()),
             "vehicle_no_data": int((v.status == "no data").sum()),
             "vehicle_core_minute_accuracy": vd.minutes_right.sum() / max(vd.minutes_with_data.sum(), 1)}
        for md in ("bike", "car", "bus", "rail"):
            m[f"{md}_right"] = int(((v["mode"] == md) & (v.status == "right")).sum())
        m["rail_no_data_shown_as_rail"] = int(nd.mode_shown_anywhere_in_core.sum())
        m.update({"other_right": int((o.status == "right").sum()), "other_wrong": int((o.status == "wrong").sum()),
                  "other_no_data": int((o.status == "no data").sum()),
                  "other_core_minute_accuracy": od.minutes_right.sum() / max(od.minutes_with_data.sum(), 1),
                  "extra_vehicle_legs": int((ex.model == name).sum())})
        summ[name] = m
        prm = {"model": name if name in OURS else f"package {name} (descriptive)",
               "model_version": OURS[name].version if name in OURS else "WAIIS package frozen_run",
               "dataset": "waiis_app_days", "rate_hz": 25, "class_set": "top level", "seed": 0,
               "gap_rule": OURS[name].gap.name() if name in OURS and OURS[name].gap else "off",
               "step": "lenient_all_models", "extra_widen_s": LN.WIDEN_S,
               "core_trim_s": f"GT/GT start {LN.TRIM_GT_S:g}, else {LN.TRIM_S:g}"}
        ds = [DataInput("waiis_app_days", "evaluation", "diary modes, core minutes only (lenient, 11)",
                        "leg suggestions / package predictions", window_s=15, stride_s=60)]
        with run(EXP, prm, run_name=f"lenient_all_{name.replace(' + ', '_').replace(' ', '_')}", datasets=ds):
            mlflow.log_metrics({k: float(v) for k, v in m.items()})
    s = pd.DataFrame(summ).T
    pd.set_option("display.width", 250)
    print(s.round(2).to_string())


if __name__ == "__main__":
    main()
