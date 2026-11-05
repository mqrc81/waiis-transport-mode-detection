"""Stop-point 3, once, with the approved rule (`tmd.suggest.GAP_RULE`: gap >= 5 min, speed across 2-25 m/s, k = 2 on
train and subway; R1 GPS-only gap legs), each part with the rule off and on. Descriptive; no choice is made here.

- shl23: SHL Users 2-3 (reserved; all positions), 10's final E1 / E2 (seed 0, prediction only), C1 at 20 s as 11's
  shl23.py, gaps on the same 20 s track per recording. Leg metrics as dev (dev_k.evaluate).
- es_test: ExtraSensory test users (reserved): false rail time (the guard), 10's saved test raw probabilities.
- waiis: the three WAIIS app days through `tmd.suggest` (rule on = SPECS; off = gap None), 11's lenient diary
  comparison per model and variant, rail legs separately, gap legs (`underground_rail`), extra rail legs, run time
  per day (single thread). Output files (rule on) and tables under <processed>/12_.../waiis/ (personal, never in
  git); this script prints counts only.

    uv run python sp3.py shl23|es_test|waiis      # logs/sp3_<part>.log; MLflow one run per model (and variant)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import dataclasses
import json
import logging
import subprocess
import sys
import time

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

import dev_k as K
from tmd import bench
from tmd import gps as tgps
from tmd import legs as L
from tmd import legs_adapter as A
from tmd import suggest as S
from tmd.features import gps as gf
from tmd.labels import FINE_ID
from tmd.models import e10 as E
from tmd.models import fusion, minutes
from tmd.readers import waiis
from tmd.tracking import DataInput, run

sys.path.insert(0, str(K.HERE.parent / "11_leg-suggestions_waiis-app-days"))
import lenient as LN
import review as R
import shl23 as S11

log = logging.getLogger()
GAP = S.GAP_RULE
OUT = E.PROCESSED / K.EXP / "waiis"
VARIANTS = ("off", "on")


def setup(part: str) -> None:
    logging.basicConfig(filename=K.HERE / "logs" / f"sp3_{part}.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    K.TABLES.mkdir(parents=True, exist_ok=True)


def fine_names(source: str, df: pd.DataFrame) -> np.ndarray:
    """Fine truth name per frame row, reserved users included (v2 rules: cls_fine, else majority; excluded labels
    and flagged run users unlabelled; excluded users unlabelled)."""
    lab = bench.read(source, "labels")
    c = lab.cls_fine.astype(str)
    if "majority_cls" in lab:
        c = c.where(c != "", lab.majority_cls.astype(str))
    c = c.where(~(lab.label_excluded | lab.run_user_flagged | lab.user_excluded), "")
    lab = lab.assign(y8=c.map(FINE_ID).fillna(0).astype(int))[["user", "ts", "y8"]]
    y = df[["user", "ts"]].merge(lab, on=["user", "ts"], how="left", validate="m:1").y8.fillna(0).astype(int)
    return K.NAMES[y.to_numpy()]


def part_of(name: str, df: pd.DataFrame, fine: np.ndarray, covered: np.ndarray, mg: pd.DataFrame) -> K.Part:
    seq = minutes.seq_ids(df)
    order = np.lexsort((df.ts.to_numpy(), seq))
    ud = df.user.astype(str) + "/" + pd.to_datetime(df.ts, unit="s").dt.date.astype(str)
    day = df.day.astype(str).where(df.day.astype(str) != "", ud).to_numpy() if "day" in df else ud.to_numpy()
    return K.Part(name, df, order, seq[order], df.ts.to_numpy(np.int64)[order], fine[order], day[order], covered,
                  mg.gap_dt.to_numpy(), mg.gap_v.to_numpy())


def smooth(spec: S.Spec, df: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    h = np.load(spec.hmm)
    return E.pad9(minutes.smooth(df, raw, h["a"], h["pi"], "fb"))


# ---------------------------------------------------------------- SHL Users 2-3

def shl23() -> None:
    setup("shl23")
    hips, other = S11.frames()
    c1 = S11.c1_features(hips)
    tr = E.shl_thinned_track(source="shl_user23", include_reserved=True)
    g = tgps.gaps(tr, ["rec"])
    rec = bench.read("shl_user23", "labels", columns=["user", "ts", "rec"])
    log.info("SHL Users 2-3: gaps %d, moving %d", len(g), GAP.moving(g.dt, g.v_across).sum())
    rows, frames = [], {}
    for nm, df in (("hips", hips), ("other", other)):
        df = df.reset_index(drop=True)
        xg = df[["user", "ts"]].merge(c1, on=["user", "ts"], how="left")[E.GPS].to_numpy(np.float32)
        k = df[["user", "ts"]].merge(rec, on=["user", "ts"], how="left", validate="m:1")
        mg = tgps.minute_gaps(k, g, ["rec"])
        frames[nm] = (df, xg, part_of(nm, df, fine_names("shl_user23", df), gf.covered(xg), mg))
    for mid, spec in S.SPECS.items():
        acc, gps = lgb.Booster(model_file=str(spec.acc)), lgb.Booster(model_file=str(spec.gps))
        met = {}
        for var in VARIANTS:
            ps, igs = {}, {}
            for nm, (df, xg, pt) in frames.items():
                raw = fusion.product(acc.predict(df[E.ACC].to_numpy(np.float32), num_threads=1),
                                     gps.predict(xg, num_threads=1), pt.covered)
                ig = GAP.moving(pt.gap_dt, pt.gap_v) if var == "on" else np.zeros(len(df), bool)
                ps[nm] = smooth(spec, df, L.boost_rail(raw, ig, GAP.k) if var == "on" else raw)
                igs[nm] = ig
            res = {"pooled": K.evaluate([(frames[n][2], ps[n], igs[n]) for n in frames], spec.rules),
                   "hips": K.evaluate([(frames["hips"][2], ps["hips"], igs["hips"])], spec.rules)}
            pos = frames["other"][0].position.to_numpy()
            for p in np.unique(pos):
                sel = pos == p
                res[p] = K.evaluate([(K.sub_part(frames["other"][2], sel, p), ps["other"][sel], igs["other"][sel])],
                                    spec.rules)
            for part, m in res.items():
                rows.append({"model": mid, "rule": var, "position": part, **{k: m[k] for k in K.KEYS if k in m}})
            met.update({f"{var}_{a}": b for a, b in K.flat(res).items()})
        prm = {"model": mid, "model_version": spec.version, "dataset": "shl_user23 (reserved; all positions)",
               "rate_hz": 25, "class_set": "top level", "seed": 0, "leg_rules": spec.rules.name(),
               "gap_rule": GAP.name(), "boost_columns": "train, subway", "gps_rule": "C1 (20 s)", "step": "sp3_shl23"}
        ds = [DataInput("shl_user23_test_v3", "evaluation", "bench-v2 (9 fine classes) -> tool top level",
                        "acc-v1-B + gps-v2 (C1, 20 s); gap rule off / on", window_s=15, stride_s=60)]
        with run(K.EXP, prm, run_name=f"sp3_shl23_{mid}", datasets=ds):
            mlflow.log_metrics({a: float(b) for a, b in met.items() if b == b})
    t = pd.DataFrame(rows)
    t.to_csv(K.TABLES / "sp3_shl23.csv", index=False)
    cols = ["model", "rule", "position", "edits_per_h", "time_acc", "leg_f1", "subway_truth_legs", "subway_leg_recall",
            "rail_leg_recall", "rail_leg_precision", "extra_rail_legs_per_h", "false_rail_car", "false_rail_bus",
            "false_rail_still", "in_gap_minutes", "in_gap_subway_minutes"]
    print(t[cols].round(4).to_string(index=False))


# ---------------------------------------------------------------- ExtraSensory test (guard)

def es_test() -> None:
    setup("es_test")
    a = pd.read_parquet(E.P10 / "extrasensory_test_25hz_w15s_acc-v1-B.parquet")
    gx = pd.read_parquet(E.P10 / "extrasensory_test_gps-v2_w2min.parquet")
    assert (a.user.to_numpy() == gx.user.to_numpy()).all() and (a.ts.to_numpy() == gx.ts.to_numpy()).all()
    df = a.assign(source="extrasensory").reset_index(drop=True)
    cov = gf.covered(gx[E.GPS].to_numpy(np.float32))
    fx = bench.read("extrasensory", "gps_per_minute", columns=["user", "session", "time", "lat", "lon", "accuracy"])
    fx = fx[fx.user.isin(df.user.unique()) & (fx.accuracy <= E.COARSE_M)]
    g = tgps.gaps(fx, ["user", "session"])
    mg = tgps.minute_gaps(df[["user", "session", "ts"]], g, ["user", "session"])
    pt = part_of("es_test", df, fine_names("extrasensory", df), cov, mg)
    plat = bench.read("extrasensory", "labels", columns=["user", "ts", "platform"])
    plat = df[["user", "ts"]].merge(plat, on=["user", "ts"], how="left").platform.to_numpy()
    log.info("ExtraSensory test: %d users, %d minutes, gaps %d, moving %d", df.user.nunique(), len(df), len(g),
             GAP.moving(g.dt, g.v_across).sum())
    rows = []
    for mid, spec in S.SPECS.items():
        z = np.load(E.OOF10 / f"test_{mid}_seed0.npz", allow_pickle=True)
        sel = z["source"] == "extrasensory"
        assert (z["user"][sel] == df.user.to_numpy(str)).all() and (z["ts"][sel] == df.ts.to_numpy()).all()
        raw = z["raw"][sel]
        d = float(np.abs(smooth(spec, df, raw) - E.pad9(z["p"][sel])).max())
        log.info("%s: HMM on 10's saved test raw reproduces 10's saved test p: max |diff| %.2e", mid, d)
        assert d < 1e-6, d
        met = {}
        for var in VARIANTS:
            ig = GAP.moving(pt.gap_dt, pt.gap_v) if var == "on" else np.zeros(len(df), bool)
            p = smooth(spec, df, L.boost_rail(raw, ig, GAP.k) if var == "on" else raw)
            res = {"all": K.evaluate([(pt, p, ig)], spec.rules)}
            for pf in ("iphone", "android"):
                s = plat == pf
                res[pf] = K.evaluate([(K.sub_part(pt, s, pf), p[s], ig[s])], spec.rules)
            for part, m in res.items():
                rows.append({"model": mid, "rule": var, "part": part, **{k: m[k] for k in K.KEYS if k in m}})
            met.update({f"{var}_{a}": b for a, b in K.flat(res).items()})
        prm = {"model": mid, "model_version": spec.version, "dataset": "extrasensory test (12 reserved users)",
               "rate_hz": 25, "class_set": "top level", "seed": 0, "leg_rules": spec.rules.name(),
               "gap_rule": GAP.name(), "boost_columns": "train, subway", "gps_rule": "per minute",
               "step": "sp3_es_test_guard"}
        ds = [DataInput("extrasensory_test_v3", "evaluation", "bench-v2 (9 fine classes) -> tool top level",
                        "acc-v1-B + gps-v2 (per minute); gap rule off / on", window_s=15, stride_s=60)]
        with run(K.EXP, prm, run_name=f"sp3_es_test_{mid}", datasets=ds):
            mlflow.log_metrics({a: float(b) for a, b in met.items() if b == b})
    t = pd.DataFrame(rows)
    t.to_csv(K.TABLES / "sp3_es_test.csv", index=False)
    cols = ["model", "rule", "part", "false_rail_nonrail", "false_rail_min", "false_rail_still", "false_rail_walk",
            "false_rail_car", "false_rail_bus", "extra_rail_legs_per_h", "edits_per_h", "in_gap_minutes", "hours"]
    print(t[cols].round(5).to_string(index=False))


# ---------------------------------------------------------------- WAIIS days

def local(t: float) -> str:
    return pd.to_datetime(t, unit="s", utc=True).tz_convert(R.TZ).strftime("%H:%M")


def overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def waiis_days() -> None:
    setup("waiis")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    per_leg, rail_rows, extras, runtime, same11 = [], [], [], [], []
    for day in waiis.days():
        dv = waiis.device(day.user, day.date)
        dia = R.diary(day.date)
        t0 = time.perf_counter()
        d_in = S.day_input(day)
        t_input = time.perf_counter() - t0
        ts = d_in.minutes.ts.to_numpy(float)
        centre = ts + 30
        n_mg = int(GAP.moving(d_in.gaps.dt, d_in.gaps.v_across).sum()) if len(d_in.gaps) else 0
        log.info("%s: minutes %d, fixes %d, gaps %d, moving %d, model minutes in a moving gap %d", day.date,
                 len(d_in.minutes), len(d_in.fixes), len(d_in.gaps), n_mg, int(S.in_gap(S.SPECS["E1"], d_in).sum()))
        for mid, spec_on in S.SPECS.items():
            for var in VARIANTS:
                spec = spec_on if var == "on" else dataclasses.replace(spec_on, gap=None)
                t1 = time.perf_counter()
                lg = S.suggest(spec, d_in)
                t_model = time.perf_counter() - t1
                runtime.append({"date": day.date, "model": mid, "rule": var, "seconds_input": t_input,
                                "seconds_model": t_model, "seconds_total": t_input + t_model,
                                "minutes": len(d_in.minutes)})
                if var == "on":
                    doc = A.document(lg, day.user, day.date, spec.id, spec.version,
                                     f"{S.RULES_VERSION} {spec.rules.name()} {GAP.name()}",
                                     {"platform": dv.platform, "phone_model": dv.phone_model,
                                      "axis_convention": dv.axis_convention, "gps_rule": "C1 (nearest the minute centre)",
                                      "gap_rule": GAP.name(), "acc_rate_hz": 25}, commit)
                    A.write(doc, OUT / f"user={day.user}" / f"date={day.date}" / f"legs_{mid}.json")
                else:   # rule off must give 11's legs
                    old = A.read_legs(R.OUT / f"user={day.user}" / f"date={day.date}" / f"legs_{mid}.json")
                    same = len(old) == len(lg) and np.allclose(old.start, lg.start) and np.allclose(old.end, lg.end)
                    same11.append({"date": day.date, "model": mid, "same_as_11": bool(same)})
                legs = lg.assign(mode=["unknown" if m < 0 else L.MODES[m] for m in lg["mode"]])
                pred = LN.mode_at(legs[["start", "end", "mode"]], centre)
                gap_legs = legs[legs.source == "rule:gps_gap_moving"]
                for r in dia.itertuples():
                    lo, hi, mid_only = LN.core(r)
                    sel = ((ts <= lo) & (ts + 60 > lo)) if mid_only else ((centre >= lo) & (centre <= hi))
                    p = pred[sel]
                    n, n_ok = int(sel.sum()), int((p == r.mode).sum())
                    status = "no data" if n == 0 else "right" if n_ok > n / 2 else "wrong"
                    gcov = sum(overlap(r.start, r.end, a, b) for a, b in zip(gap_legs.start, gap_legs.end))
                    per_leg.append({"date": day.date, "diary_id": r.diary_id, "start_local": r.start_local,
                                    "end_local": r.end_local, "mode": r.mode, "sub_mode": r.sub_mode, "gt": r.gt,
                                    "model": mid, "rule": var, "minutes_with_data": n, "minutes_right": n_ok,
                                    "status": status, "found": n_ok > 0,
                                    "gap_leg_share": gcov / max(r.end - r.start, 1.0)})
                    if r.mode == "rail":
                        ml = legs[(legs.source == "model")]
                        ov = [(overlap(r.start, r.end, a, b), m, h) for a, b, m, h in
                              zip(ml.start, ml.end, ml["mode"], ml.hint)]
                        ov = sorted([o for o in ov if o[0] > 0], reverse=True)
                        rail_rows.append({"date": day.date, "diary_id": r.diary_id, "start_local": r.start_local,
                                          "end_local": r.end_local, "sub_mode": r.sub_mode, "model": mid, "rule": var,
                                          "status": status, "minutes_with_data": n, "minutes_right": n_ok,
                                          "gap_leg_share": round(gcov / max(r.end - r.start, 1.0), 2),
                                          "model_legs": "; ".join(f"{m}{'/' + h if h else ''} {o / 60:.0f}m"
                                                                  for o, m, h in ov[:3])})
                # extra legs: model vehicle legs and gap legs that overlap no diary leg of their kind (widened)
                dveh = dia[dia["mode"].isin(LN.VEHICLE)]
                drail = dia[dia["mode"] == "rail"]
                for r in legs.itertuples():
                    kind = "gap_leg" if r.source == "rule:gps_gap_moving" else "model" if r.source == "model" else ""
                    if not kind or (kind == "model" and r.mode not in LN.VEHICLE):
                        continue
                    ref = drail if kind == "gap_leg" else dveh
                    hit = ((ref.start - LN.WIDEN_S < r.end) & (ref.end + LN.WIDEN_S > r.start)).any()
                    if not hit:
                        extras.append({"date": day.date, "model": mid, "rule": var, "kind": kind, "mode": r.mode,
                                       "start_local": local(r.start), "end_local": local(r.end),
                                       "minutes": round((r.end - r.start) / 60, 1)})
    x = pd.DataFrame(per_leg)
    ex = pd.DataFrame(extras, columns=["date", "model", "rule", "kind", "mode", "start_local", "end_local", "minutes"])
    rr, rt, sm = pd.DataFrame(rail_rows), pd.DataFrame(runtime), pd.DataFrame(same11)
    tabs = OUT / "tables"
    tabs.mkdir(parents=True, exist_ok=True)
    x.to_csv(tabs / "lenient_per_leg.csv", index=False)
    ex.to_csv(tabs / "extra_legs.csv", index=False)
    rr.to_csv(tabs / "rail_legs.csv", index=False)
    rt.to_csv(K.TABLES / "sp3_waiis_runtime.csv", index=False)
    print("rule off reproduces 11's legs:", sm.same_as_11.all(), sm.to_dict("records") if not sm.same_as_11.all() else "")

    summ = {}
    for mid in S.SPECS:
        for var in VARIANTS:
            g = x[(x.model == mid) & (x.rule == var)]
            met = {}
            for grp, modes in (("vehicle", LN.VEHICLE), ("other", LN.OTHER)):
                gg = g[g["mode"].isin(modes)]
                dat = gg[gg.status != "no data"]
                met.update({f"{grp}_legs": len(gg), f"{grp}_right": int((gg.status == "right").sum()),
                            f"{grp}_found": int(gg.found.sum()), f"{grp}_wrong": int((gg.status == "wrong").sum()),
                            f"{grp}_no_data": int((gg.status == "no data").sum()),
                            f"{grp}_core_minute_accuracy": float(dat.minutes_right.sum() /
                                                                 max(dat.minutes_with_data.sum(), 1))})
            for md in sorted(LN.VEHICLE):
                gm = g[g["mode"] == md]
                if len(gm):
                    met[f"{md}_legs"], met[f"{md}_right"] = len(gm), int((gm.status == "right").sum())
                    met[f"{md}_no_data"] = int((gm.status == "no data").sum())
            rail = g[g["mode"] == "rail"]
            nod = rail[rail.status == "no data"]
            met["rail_no_data_with_gap_hint"] = int((nod.gap_leg_share > 0).sum())
            met["rail_no_data_gap_leg_share_mean"] = float(nod.gap_leg_share.mean()) if len(nod) else 0.0
            met["rail_with_gap_hint"] = int((rail.gap_leg_share > 0).sum())
            e = ex[(ex.model == mid) & (ex.rule == var)]
            met["extra_vehicle_legs"] = int((e.kind == "model").sum())
            met["extra_rail_legs"] = int(((e.kind == "model") & (e["mode"] == "rail")).sum())
            met["extra_gap_legs"] = int((e.kind == "gap_leg").sum())
            r = rt[(rt.model == mid) & (rt.rule == var)]
            met["run_seconds_per_day_mean"] = float(r.seconds_total.mean())
            met["run_seconds_per_day_max"] = float(r.seconds_total.max())
            summ[(mid, var)] = met
            prm = {"model": mid, "model_version": S.SPECS[mid].version, "dataset": "waiis_app_days", "rate_hz": 25,
                   "class_set": "top level", "seed": 0, "rule": var, "gap_rule": GAP.name() if var == "on" else "off",
                   "leg_rules": S.SPECS[mid].rules.name(), "step": "sp3_waiis_lenient", "threads": 1,
                   "core_trim_s": f"GT/GT start {LN.TRIM_GT_S:g}, else {LN.TRIM_S:g}", "extra_widen_s": LN.WIDEN_S}
            ds = [DataInput("waiis_app_days", "evaluation", "diary modes, core minutes only (lenient, 11)",
                            "leg suggestions (tmd.suggest)", window_s=15, stride_s=60)]
            with run(K.EXP, prm, run_name=f"sp3_waiis_{mid}_{var}", datasets=ds):
                mlflow.log_metrics({k: float(v) for k, v in met.items()})
            log.info("%s %s %s", mid, var, json.dumps(met))
    s = pd.DataFrame(summ).T
    s.index.names = ["model", "rule"]
    s.to_csv(K.TABLES / "sp3_waiis_summary.csv")       # counts only
    pd.set_option("display.width", 250)
    print(s.to_string())
    v = x[x["mode"].isin(LN.VEHICLE)].pivot_table(index=["date", "diary_id", "mode", "sub_mode"],
                                                  columns=["model", "rule"], values="status", aggfunc="first")
    print("\nvehicle legs, status by model and rule:\n", v.to_string())
    print("\nrail legs:\n", rr[["date", "diary_id", "sub_mode", "model", "rule", "status", "gap_leg_share",
                                "model_legs"]].to_string(index=False))
    print("\nextra legs:\n", ex.to_string(index=False))
    print("\nrun time per day [s]:\n", rt.groupby(["model", "rule"]).seconds_total.agg(["mean", "max"]).round(3))


if __name__ == "__main__":
    {"shl23": shl23, "es_test": es_test, "waiis": waiis_days}[sys.argv[1]]()
