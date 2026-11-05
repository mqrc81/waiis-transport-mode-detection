"""Stop-point 3: E2 and E1 on the three WAIIS app days with the approved input handling and leg rules; one output file
per model and day (`tmd.legs_adapter`), review sheets against the diaries, tram check, the package's own
predictions beside them. Single thread; run time per day measured. Personal outputs under
<processed>/11_.../waiis/ (never in git); this script prints counts only.

    uv run python waiis_days.py          # log: logs/waiis_days.log; MLflow one run per day and model
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import logging  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import dev  # noqa: E402
import review as R  # noqa: E402
from tmd import legs as L  # noqa: E402
from tmd import legs_adapter as A  # noqa: E402
from tmd import suggest as S  # noqa: E402
from tmd.progress import Progress  # noqa: E402
from tmd.readers import waiis  # noqa: E402
from tmd.tracking import DataInput, run  # noqa: E402

log = logging.getLogger()
REVIEW = R.OUT / "review"


def named(lg: pd.DataFrame) -> pd.DataFrame:
    out = lg.copy()
    out["mode"] = ["unknown" if m < 0 else L.MODES[m] for m in lg["mode"]]
    out["conf"] = lg.p1
    return out


def local(t):
    return pd.to_datetime(np.asarray(t, float), unit="s", utc=True).tz_convert(R.TZ).strftime("%H:%M")


def main():
    logging.basicConfig(filename=dev.HERE / "logs" / "waiis_days.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    days = waiis.days()
    prog = Progress(dev.EXP, "waiis_days", len(days) * len(S.SPECS), log)
    REVIEW.mkdir(parents=True, exist_ok=True)
    summ, drows_all, tram_all = [], [], []
    for day in days:
        dv = waiis.device(day.user, day.date)
        t0 = time.perf_counter()
        d_in = S.day_input(day)
        t_input = time.perf_counter() - t0
        dia = R.diary(day.date)
        lanes = {"diary": dia.assign(hint=dia.sub_mode, conf=np.nan, source="diary")}
        leg_tables, per_leg = {}, []
        for mid, spec in S.SPECS.items():
            t1 = time.perf_counter()
            lg = S.suggest(spec, d_in)
            t_model = time.perf_counter() - t1
            doc = A.document(lg, day.user, day.date, spec.id, spec.version, f"{S.RULES_VERSION} {spec.rules.name()}",
                             {"platform": dv.platform, "phone_model": dv.phone_model, "axis_convention":
                              dv.axis_convention, "gps_rule": "C1 (nearest the minute centre)",
                              "acc_rate_hz": 25}, commit)
            A.write(doc, R.OUT / f"user={day.user}" / f"date={day.date}" / f"legs_{spec.id}.json")
            nl = named(lg)
            leg_tables[mid] = nl
            lanes[mid] = nl
            dm, lm = R.match(dia, nl, day.date, mid)
            tc = R.tram_check(dia, lg.attrs["minutes"], nl, mid)
            drows_all.append(dm.assign(date=day.date))
            if len(tc):
                tram_all.append(tc.assign(date=day.date))
            src = nl.source.to_numpy()
            mdl = nl[src == "model"]
            n_min = len(d_in.minutes)
            m = {"minutes": n_min, "minutes_gps_covered_share": float(d_in.minutes.covered.mean()) if n_min else 0,
                 "legs_model": len(mdl), "legs_gps_only_still": int((src == "rule:gps_only_still").sum()),
                 "legs_gps_only_moving": int((src == "rule:gps_only_moving").sum()),
                 "legs_unknown_low_confidence": int((mdl["mode"] == "unknown").sum()),
                 "flagged_leg_share": float((mdl["mode"] == "unknown").mean()) if len(mdl) else 0,
                 "diary_legs": len(dia), "diary_matched": int(dm.matched.sum()),
                 "diary_missed": int((~dm.matched).sum()), "extra_legs": int((lm.status == "extra").sum()),
                 "not_in_diary_walking_untracked": int(lm.status.str.startswith("not in diary").sum()),
                 "diary_same_mode_share_mean": float(dm.same_mode_share.mean()),
                 "diary_model_data_share_mean": float(dm.model_data_share.mean()),
                 "run_seconds_input": t_input, "run_seconds_model": t_model,
                 "run_seconds_total": t_input + t_model,
                 "run_ms_per_minute_window": 1000 * (t_input + t_model) / max(n_min, 1),
                 "acc_hours": d_in.n_samples / 24.9 / 3600}
            summ.append({"date": day.date, "model": mid, **m})
            prm = {"model": spec.id, "model_version": spec.version, "dataset": "waiis_app_days", "rate_hz": 25,
                   "class_set": "top level (still, walking, running, bike, car, bus, rail)", "seed": 0,
                   "date": day.date, "leg_rules": spec.rules.name(), "gps_rule": "C1",
                   "platform": dv.platform, "step": "waiis_days", "threads": 1}
            ds = [DataInput("waiis_app_days", "evaluation", "diary (rough times; review only)",
                            f"acc-v1-B + gps-v2 (C1, 20 s)", window_s=15, stride_s=60)]
            with run(dev.EXP, prm, run_name=f"waiis_{day.date}_{mid}", datasets=ds):
                mlflow.log_metrics({k: float(v) for k, v in m.items()})
            log.info("%s %s %s", day.date, mid, json.dumps({k: round(v, 3) for k, v in m.items()}))
            for i, r in nl.reset_index(drop=True).iterrows():
                per_leg.append({"model": mid, "leg": i, "start_local": local([r.start])[0],
                                "end_local": local([r.end])[0], "minutes": round((r.end - r.start) / 60, 1),
                                "mode": r["mode"], "top1": "unknown" if r.top1 < 0 else L.MODES[r.top1],
                                "confidence": round(r.p1, 3) if r.p1 == r.p1 else "",
                                "second": "" if r.top2 < 0 else L.MODES[r.top2],
                                "second_p": round(r.p2, 3) if r.p2 == r.p2 else "", "hint": r.hint,
                                "hint_share": round(r.hint_share, 2) if r.hint_share == r.hint_share else "",
                                "coverage": r.coverage, "gps_covered_share": round(r.gps_covered_share, 2),
                                "flags": ";".join(f["code"] for f in r.flag_reasons),
                                "status": lm.status.iat[i],
                                "same_mode_diary_share": round(lm.same_mode_diary_share.iat[i], 2),
                                "review": "", "review_note": ""})
            prog.step(f"{day.date} {mid}")
        pk = R.package_legs(day.path)
        lanes["package"] = pk.assign(conf=np.nan)
        dpk, lpk = R.match(dia, pk, day.date, "package")
        drows_all.append(dpk.assign(date=day.date))
        summ.append({"date": day.date, "model": "package", "diary_legs": len(dia),
                     "diary_matched": int(dpk.matched.sum()), "diary_missed": int((~dpk.matched).sum()),
                     "extra_legs": int((lpk.status == "extra").sum()),
                     "diary_same_mode_share_mean": float(dpk.same_mode_share.mean())})
        g = waiis.fix_states(waiis.read_gps(day))
        g = g[g.accuracy_m <= waiis.COARSE_M]
        mins = d_in.minutes.assign(kind=np.where(d_in.minutes.covered, "acc+gps", "acc"))
        R.sheet(REVIEW / f"{day.date}_sheet.png", day.date, lanes, mins, g,
                f"WAIIS {day.date}: diary vs E2 / E1 leg suggestions vs the package's {R.PKG_MODEL} "
                "(labels: mode, hint, confidence; hatched = GPS-only or unknown)")
        pd.DataFrame(per_leg).to_csv(REVIEW / f"{day.date}_legs_review.csv", index=False)
    s = pd.DataFrame(summ)
    s.to_csv(REVIEW / "summary.csv", index=False)
    pd.concat(drows_all).to_csv(REVIEW / "diary_matching.csv", index=False)
    if tram_all:
        pd.concat(tram_all).to_csv(REVIEW / "tram_check.csv", index=False)
    cols = ["date", "model", "diary_legs", "diary_matched", "diary_missed", "extra_legs", "legs_model",
            "legs_gps_only_still", "legs_gps_only_moving", "legs_unknown_low_confidence", "run_seconds_total",
            "minutes"]
    print(s[[c for c in cols if c in s]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
