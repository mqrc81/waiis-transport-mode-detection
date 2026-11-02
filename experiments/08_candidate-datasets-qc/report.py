"""All check tables (README: Plan, checks 1-9; Results) from the window and inventory tables.

    uv run python report.py

Reads <data>/processed/{collecty,nor_tmd}_windows_25hz_w15s/, results/tables/mobilitapp_*.csv and the v1 benchmark
(official examples, for comparison and for the still-energy threshold: dev users only). Writes
results/tables/<name>.csv and results/tables/report.md (every table, in README order); stdout gets a short summary.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from tmd import bench
from tmd import candidates as q
from tmd import collecty, nor_tmd
from tmd.labels import CLASSES
from tmd.paths import PROCESSED

import mobilitapp                    # this folder: not used beyond 08

HERE = Path(__file__).resolve().parent
TABLES = HERE / "results" / "tables"
VEHICLES = {"car", "bus", "train", "subway"}
REST_STD = 0.1                 # m/s2: "phone at rest" windows for the |acc| level check (no still label in 2 sets)
MD = []


def md(df: pd.DataFrame, name: str, title: str, index: bool = True) -> None:
    """Save df as CSV and append it to report.md."""
    df.to_csv(TABLES / f"{name}.csv", index=index)
    d = df.reset_index() if index else df
    fmt = lambda v: (f"{v:,.3g}" if abs(v) < 1 and v != 0 else f"{v:,.1f}" if v % 1 else f"{int(v):,}") \
        if isinstance(v, (float, np.floating)) and np.isfinite(v) else ("" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(fmt(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |"
              for r in d.itertuples(index=False)]
    MD.append(f"### {title}\n\n`{name}.csv`\n\n" + "\n".join(lines) + "\n")


def load(ds: str) -> dict:
    p = PROCESSED / f"{ds}_windows_25hz_w15s"
    t = {k: pd.read_parquet(p / f"{k}.parquet") for k in ("minutes", "tiles", "timing", "legs", "clipping")}
    t["acc"] = pd.read_parquet(p / "accelerometer.parquet")
    return t


def mapping(ds: str, raw: str) -> str:
    if ds == "collecty":
        if raw in collecty.EXCLUDED_CONFLICT:
            return "excluded (conflict)"
        return collecty.TO_V1.get(raw) or (f"outside v1: {collecty.OUTSIDE_V1[raw]}" if raw in collecty.OUTSIDE_V1
                                           else "dropped")
    if raw in nor_tmd.EXCLUDED_CONFLICT:
        return "excluded (conflict)"
    return nor_tmd.TO_V1.get(raw) or (f"outside v1: {nor_tmd.OUTSIDE_V1[raw]}" if raw in nor_tmd.OUTSIDE_V1
                                      else "dropped")


def pq3(s: pd.Series) -> str:
    return "–" if s.dropna().empty else " / ".join(f"{v:.3g}" for v in s.quantile([.05, .5, .95]))


# ---------------------------------------------------------------- per window-table dataset

def window_dataset(ds: str, name: str, still_t: float) -> dict:
    t = load(ds)
    dropped = 0
    if ds == "collecty":               # byte-identical files (tmd.collecty.duplicates; proposal in the README)
        d = collecty.duplicates()
        drop = {(str(u), str(k)) for u, k in d.loc[d["drop"], ["user", "traj"]].itertuples(index=False)}
        for k in ("minutes", "tiles", "timing", "legs", "clipping", "acc"):
            df = t[k]
            key = list(zip(df.user.astype(str), df.trip.astype(str).str.split(".").str[0]))
            t[k] = df[[x not in drop for x in key]].reset_index(drop=True)
        dropped = len(drop)
    m, tl, tim, lg, cl = t["minutes"], t["tiles"], t["timing"], t["legs"], t["clipping"]
    m["v1"] = m.cls.ne("")
    lg["hours"] = lg.dur_s / 3600
    lg["mapped"] = [mapping(ds, x) for x in lg.label]
    dev = m.groupby("user").agg(platform=("platform", "first"), model=("model", "first"))

    # 1. inventory
    trips = lg.groupby(["user", "trip"]).agg(legs=("label", "size"), labels=("label", "nunique"),
                                             start=("t_start_ms", "min"), end=("t_end_ms", "max"))
    inv = pd.DataFrame({
        "users / devices": [m.user.nunique()],
        "platform": [" / ".join(f"{k} {v}" for k, v in dev.platform.value_counts().items())],
        "device models": [dev.model.replace("", np.nan).nunique()],
        "period": [f"{pd.to_datetime(trips.start.min(), unit='ms'):%Y-%m-%d} – {pd.to_datetime(trips.end.max(), unit='ms'):%Y-%m-%d}"],
        "trips": [len(trips)],
        "multi-mode trips": [int((trips.labels > 1).sum())],
        "recorded h (legs)": [lg.hours.sum()]})
    md(inv, f"{ds}_inventory", f"{name}: inventory (check 1)", index=False)
    hpu = lg.pivot_table(index="user", columns="label", values="hours", aggfunc="sum", fill_value=0).round(2)
    hpu["total"] = hpu.sum(axis=1)
    md(hpu.sort_values("total", ascending=False), f"{ds}_hours_per_user", f"{name}: hours per raw label per user (check 1)")
    if ds == "nor_tmd":
        md(dev.model.value_counts().rename("installations").to_frame(), f"{ds}_models", f"{name}: device models")

    # 2. units and plausibility
    ok = m[m.status.eq("ok")]
    rest = ok[ok.acc_std < REST_STD]
    units = ok.groupby("platform").agg(ok_windows=("ts", "size"), median_acc=("acc_median", "median"))
    units["rest_windows"] = rest.groupby("platform").size()
    units["median_acc_at_rest"] = rest.groupby("platform").acc_median.median()
    units["fail_U3_%"] = 100 * ok.groupby("platform").acc_implausible.mean()
    units["fail_U3_low_%"] = 100 * ok.assign(lo=ok.acc_median < q.ACC_PLAUSIBLE[0]).groupby("platform").lo.mean()
    md(units, f"{ds}_units", f"{name}: |acc| level and v1 plausibility (U3) over ok minute windows (check 2)")
    sign = pd.crosstab(ok.platform, ok.dom_axis, normalize="index").mul(100).round(1)
    md(sign, f"{ds}_axis_sign", f"{name}: gravity direction (dominant axis of the window mean), % of ok windows (check 2)")
    clip = cl.groupby("user").agg(platform=("platform", "first"), model=("model", "first"),
                                  acc_norm_max=("acc_norm_max", "max"),
                                  axis_absmax=("x_absmax", "max")).sort_values("acc_norm_max")
    clip["axis_absmax"] = cl.groupby("user")[["x_absmax", "y_absmax", "z_absmax"]].max().max(axis=1)
    clip["events_at_axis_max"] = cl.groupby("user")[["x_at_max", "y_at_max", "z_at_max"]].sum().sum(axis=1)
    md(clip.describe().T.round(2), f"{ds}_clipping_summary", f"{name}: max |acc| and per-axis max per device (check 2)")
    clip.to_csv(TABLES / f"{ds}_clipping_per_device.csv")

    # 3. timing (per device: medians over trips)
    tim = tim.copy()
    tim["throttle"] = tim.rate_after5 / tim.rate_first5
    td = tim.groupby(["platform", "sensor"]).agg(
        trips=("events", "size"), rate_hz=("rate_hz", pq3), dt_median_ms=("dt_median_ms", pq3),
        dt_p5_ms=("dt_p5_ms", "median"), dt_p95_ms=("dt_p95_ms", "median"),
        repeated_per_1k=("repeated", lambda s: 1e3 * s.sum() / tim.loc[s.index, "events"].sum()),
        trips_backwards=("backwards", lambda s: int((s > 0).sum())),
        gaps_per_h=("gaps_over_0.5s", lambda s: s.sum() / (tim.loc[s.index, "span_s"].sum() / 3600)),
        regular=("regular", "median"), later_vs_first5=("throttle", "median"))
    md(td, f"{ds}_timing", f"{name}: timing per platform and sensor; rate and dt as p5 / median / p95 over trips (check 3)")
    tdev = tim[tim.sensor == "acc"].groupby("user").agg(platform=("platform", "first"), model=("model", "first"),
                                                         trips=("events", "size"), rate_hz=("rate_hz", "median"),
                                                         dt_p5=("dt_p5_ms", "median"), dt_med=("dt_median_ms", "median"),
                                                         dt_p95=("dt_p95_ms", "median"))
    tdev.to_csv(TABLES / f"{ds}_timing_per_device.csv")
    md(pd.crosstab(tdev.platform, pd.cut(tdev.rate_hz, [0, 5, 10, 25, 50, 100, 200, 1e4])),
       f"{ds}_rate_bins", f"{name}: devices by median accelerometer rate [Hz] (check 3)")

    # 4. window yield
    st = pd.crosstab(m.platform, m.status, margins=True)
    md(st, f"{ds}_minute_status", f"{name}: minute grid, status per window (check 4)")
    tl["v1"] = tl.cls.ne("")
    y = pd.DataFrame({
        "clean minute windows": m.groupby("platform").clean.sum(),
        "  with a v1 class": m[m.v1].groupby("platform").clean.sum(),
        "clean hours (tiles)": tl.groupby("platform").clean.sum() * q.WIN_S / 3600,
        "  with a v1 class (h)": tl[tl.v1].groupby("platform").clean.sum() * q.WIN_S / 3600,
        "rate < 25 Hz (h)": tl[tl.status.eq("rate_below_25hz")].groupby("platform").size() * q.WIN_S / 3600,
        "U3 fail (h)": tl.groupby("platform").acc_implausible.sum() * q.WIN_S / 3600,
        "X3 (h)": tl.groupby("platform").label_implausible.sum() * q.WIN_S / 3600}).fillna(0)
    md(y, f"{ds}_yield", f"{name}: clean windows and hours (check 4)")

    # 5. other sensors (clean v1 windows)
    c = m[m.clean & m.v1]
    md(c.groupby("platform")[["has_gyro", "has_mag", "has_press"]].mean().mul(100).round(1),
       f"{ds}_other_sensors", f"{name}: % of clean v1 minute windows with gyroscope (>= 25 Hz, no gap > 0.5 s), magnetometer, barometer (check 5)")

    # 7. distributions
    md(pd.crosstab(c.cls, c.platform, margins=True).reindex([x for x in CLASSES if x in set(c.cls)] + ["All"]),
       f"{ds}_class_platform", f"{name}: clean v1 minute windows per class and platform (check 7)")
    cu = m[m.clean].groupby(["label", "user"]).size()
    users = pd.DataFrame({"users >= 5 windows": (cu >= 5).groupby(level=0).sum(),
                          "top user share": (cu.groupby(level=0).max() / cu.groupby(level=0).sum()).round(2),
                          "clean windows": cu.groupby(level=0).sum()})
    users["mapping"] = [mapping(ds, x) for x in users.index]
    md(users, f"{ds}_users_per_label", f"{name}: users per raw label among clean minute windows (check 7)")
    if ds == "nor_tmd":
        md(pd.crosstab(c.position, c.platform, margins=True), f"{ds}_position",
           f"{name}: phone position, clean v1 minute windows (check 7)")

    # 8. labels: mapping, hours and users per raw label, segment lengths, plausibility flags
    lab = lg.groupby("label").agg(mapped=("mapped", "first"), hours=("hours", "sum"), users=("user", "nunique"),
                                  legs=("trip", "size"), leg_min=("dur_s", lambda s: pq3(s / 60)))
    lab["clean windows"] = m[m.clean].groupby("label").size()
    okl = m[m.status.eq("ok")]
    lab["X3 quiet % (walk/run/bike)"] = 100 * okl.groupby("label").label_implausible.mean()
    lab["U3 fail %"] = 100 * okl.groupby("label").acc_implausible.mean()
    lab["quiet (std < 0.03) %"] = 100 * okl.groupby("label").quiet.mean()
    md(lab.rename(columns={"leg_min": "leg minutes p5 / median / p95"}), f"{ds}_labels",
       f"{name}: raw labels: v1 mapping, hours and users (legs), segment lengths, flags (check 8)")

    # 9. duplicates: identical clean-or-ok windows in different trips
    acc = t["acc"]
    h = [hashlib.md5(np.concatenate([x, y_, z]).astype(np.float32).tobytes()).hexdigest()
         for x, y_, z in zip(acc.x.to_numpy(), acc.y.to_numpy(), acc.z.to_numpy())] if len(acc) else []
    acc = acc[["user", "trip", "ts"]].assign(h=h)
    dup = acc[acc.duplicated("h", keep=False)]
    dups = {"windows": len(acc), "identical windows": len(dup), "trips involved": dup.groupby(["user", "trip"]).ngroups,
            "across users": int((dup.groupby("h").user.nunique() > 1).sum()), "files dropped as duplicates": dropped}
    if ds == "nor_tmd":
        tr = nor_tmd.trips()
        k = tr.groupby(["user", "journey", "trip"]).agg(t_first=("t_first", "min"), t_last=("t_last", "max"),
                                                        events=("events", "sum")).reset_index()
        same = k[k.duplicated(["t_first", "t_last", "events"], keep=False)]
        dups["trips with identical span and events"] = len(same)
        dups["  across users"] = int((same.groupby(["t_first", "t_last", "events"]).user.nunique() > 1).sum())

    top = c.groupby(["cls", "user"]).size().rename("n").reset_index()
    tops = top.groupby("cls").agg(users=("user", "size"), windows=("n", "sum"), top_user=("n", "max"))
    tops["top user share"] = (tops.top_user / tops.windows).round(2)
    tops["top 3 users share"] = top.groupby("cls").n.apply(lambda s: s.nlargest(3).sum() / s.sum()).round(2)
    md(tops.reindex([k for k in CLASSES if k in tops.index]), f"{ds}_top_users",
       f"{name}: users and top-user shares per v1 class, clean v1 minute windows (open decision: per-user cap)")

    # summary for the cross-dataset table
    vc = c.groupby(["cls", "user"]).size()
    upc = (vc >= 5).groupby(level=0).sum()
    return {"dataset": name, "clean v1 windows": len(c), "clean v1 hours (tiles)": tl[tl.v1].clean.sum() * q.WIN_S / 3600,
            "users": c.user.nunique(), "iOS %": 100 * c.platform.eq("ios").mean(),
            "Android %": 100 * c.platform.eq("android").mean(), "gyroscope %": 100 * c.has_gyro.mean(),
            "GPS coverage %": "none", "users per class (>= 5)": ", ".join(f"{k} {upc[k]}" for k in CLASSES if k in upc),
            "classes outside v1": ", ".join(sorted({mapping(ds, x).split(": ")[1] for x in lg.label
                                                     if mapping(ds, x).startswith("outside")})),
            "v1 classes (clean)": sorted(set(c.cls)), "dups": dups,
            "outside": lg[lg.mapped.str.startswith("outside")].assign(mode=lambda d: d.mapped.str.split(": ").str[1])
            .groupby("mode").agg(hours=("hours", "sum"), users=("user", "nunique")),
            "excluded": lg[lg.mapped.str.startswith("excluded")].groupby("label").agg(hours=("hours", "sum"),
                                                                                     users=("user", "nunique"))}


def nor_tmd_raw_units() -> None:
    """Check 2 on the raw accelerometer of every trip (iOS has no ok windows at 5 Hz): median |acc| in sensor units
    and the dominant axis (with sign) of the trip's mean vector, per platform and sensor."""
    rows = []
    for u in nor_tmd.users():
        d = nor_tmd.stream(u, ["accelerometer", "CMAccelerometerData"])
        for (j, k, sen), e in d.groupby(["journey", "trip", "sensor"], observed=True):
            a = e[["x", "y", "z"]].to_numpy(float)
            m = a.mean(0)
            ax = int(np.abs(m).argmax())
            rows.append({"sensor": str(sen), "median_norm": float(np.median(np.linalg.norm(a, axis=1))),
                         "dom_axis": ("-" if m[ax] < 0 else "+") + "xyz"[ax]})
    r = pd.DataFrame(rows)
    out = r.groupby("sensor").agg(trips=("dom_axis", "size"), median_norm=("median_norm", "median"),
                                  norm_p5_p95=("median_norm", lambda s: f"{s.quantile(.05):.3g} / {s.quantile(.95):.3g}"))
    out = out.join(pd.crosstab(r.sensor, r.dom_axis, normalize="index").mul(100).round(1))
    md(out, "nor_tmd_raw_units", "NOR-TMD: raw accelerometer per trip (sensor units): median |acc| and the dominant "
                                 "axis of the trip mean, % of trips (check 2)")


# ---------------------------------------------------------------- MobilitApp (inventory only)

def mobilitapp_report(still_t: float) -> dict:
    f = pd.read_csv(TABLES / "mobilitapp_files.csv", parse_dates=["stamp"])
    tr = pd.read_csv(TABLES / "mobilitapp_trips.csv", parse_dates=["stamp"])
    en = pd.read_csv(TABLES / "mobilitapp_energy.csv")
    acc = f[f.sensors == "ACC-MAG-GYR"]
    timed = tr[tr.timed]
    # hours: timed trips from timestamps; untimed estimated as samples / the id's median timed rate (else overall)
    rate_id = timed.groupby("id").rate_hz.median()
    tr["rate_est"] = tr.id.map(rate_id).fillna(timed.rate_hz.median())
    tr["hours"] = np.where(tr.timed, tr.span_s / 3600, tr.samples / tr.rate_est / 3600)
    inv = pd.DataFrame({"ids (phones / installations)": [tr.id.nunique()], "ids with timed trips": [timed.id.nunique()],
                        "platform": ["android (all)"], "device models": ["not recorded"],
                        "period (file stamps)": [f"{acc.stamp.min():%Y-%m-%d} – {acc.stamp.max():%Y-%m-%d}"],
                        "trips": [len(tr)], "timed trips": [len(timed)], "multi-mode trips": [int((tr.modes_in_trip > 1).sum())],
                        "files": [len(acc)], "untimed files": [int((acc.time_kind != "epoch_ms").sum())],
                        "timed h": [timed.span_s.sum() / 3600], "untimed h (estimated)": [tr[~tr.timed].hours.sum()]})
    md(inv, "mobilitapp_inventory", "MobilitApp: inventory (check 1)", index=False)
    md(pd.crosstab(acc.stamp.dt.to_period("Q").astype(str), acc.time_kind), "mobilitapp_time_kinds",
       "MobilitApp: files per quarter and timestamp kind (check 3)")
    hpm = tr.groupby("mode").agg(cls=("cls", "first"), trips=("trip", "size"), timed_trips=("timed", "sum"),
                                 ids=("id", "nunique"), timed_ids=("id", lambda s: s[tr.loc[s.index, "timed"]].nunique()),
                                 timed_h=("hours", lambda s: s[tr.loc[s.index, "timed"]].sum()),
                                 all_h_est=("hours", "sum"),
                                 trip_min=("span_s", lambda s: pq3(s / 60)))
    hpm["cls"] = hpm.cls.fillna("").replace("", "outside v1")
    md(hpm.rename(columns={"trip_min": "timed trip minutes p5 / median / p95"}).sort_values("all_h_est", ascending=False),
       "mobilitapp_modes", "MobilitApp: modes: v1 class, trips, ids, hours, trip lengths (checks 1, 8)")
    hpu = tr.pivot_table(index="id", columns="mode", values="hours", aggfunc="sum", fill_value=0).round(2)
    hpu["total"] = hpu.sum(axis=1)
    hpu.sort_values("total", ascending=False).to_csv(TABLES / "mobilitapp_hours_per_id.csv")
    md(hpu.sort_values("total", ascending=False).head(10), "mobilitapp_hours_per_id_top10",
       "MobilitApp: hours per mode, top 10 ids (all files, untimed estimated; full table in mobilitapp_hours_per_id.csv)")
    # 3. timing (timed trips; per id medians)
    tid = timed.groupby("id").agg(trips=("trip", "size"), rate_hz=("rate_hz", "median"), dt_med=("dt_median_ms", "median"),
                                  dt_p5=("dt_p5_ms", "median"), dt_p95=("dt_p95_ms", "median"))
    t3 = pd.DataFrame({"timed trips": [len(timed)], "rate_hz p5 / median / p95 (trips)": [pq3(timed.rate_hz)],
                       "dt median ms p5 / median / p95": [pq3(timed.dt_median_ms)],
                       "ids by median rate < 25 / 25-100 / > 100 Hz": [f"{(tid.rate_hz < 25).sum()} / "
                                                                         f"{tid.rate_hz.between(25, 100).sum()} / {(tid.rate_hz > 100).sum()}"],
                       "repeated per 1k events": [1e3 * timed.repeated.sum() / timed.events.sum()],
                       "trips with backward steps": [int((timed.backwards > 0).sum())],
                       "gaps > 0.5 s per h": [timed["gaps_over_0.5s"].sum() / (timed.span_s.sum() / 3600)],
                       "regular (median)": [timed.regular.median()],
                       "rate after 5 min / first 5 min (median)": [(timed.rate_after5 / timed.rate_first5).median()]})
    md(t3, "mobilitapp_timing", "MobilitApp: accelerometer timing over timed trips (check 3)", index=False)
    tid.to_csv(TABLES / "mobilitapp_timing_per_id.csv")
    # 8. flags on 15 s tiles of timed trips (linear acc)
    en = en.merge(tr[["trip", "mode", "cls", "id"]], on="trip")
    en["still_high_energy"] = en.cls.eq("still") & (en.lin_std > still_t)
    en["quiet_moving"] = en.cls.isin(q.LABEL_QUIET_CLASSES) & (en.lin_std < q.QUIET_STD)
    fl = en.groupby("mode").agg(tiles=("trip", "size"), lin_median=("lin_median", "median"),
                                lin_std_median=("lin_std", "median"),
                                still_high_energy_pct=("still_high_energy", lambda s: 100 * s.mean()),
                                quiet_walk_run_bike_pct=("quiet_moving", lambda s: 100 * s.mean()))
    md(fl, "mobilitapp_flags", f"MobilitApp: 15 s tiles of timed trips, |linear acc| level and flags; still flagged "
                                f"if std > {still_t:.2f} m/s2 (check 8)")
    # 9. duplicates
    d1 = acc[acc.duplicated("md5", keep=False)]
    d2 = acc[acc.duplicated("acc_md5", keep=False)]
    dups = {"files": len(acc), "identical files": len(d1), "identical acc content": len(d2),
            "trips involved": d2.trip.nunique(), "across ids": int((d2.groupby("acc_md5").id.nunique() > 1).sum()),
            "content groups with different modes": int((d2.groupby("acc_md5")["mode"].nunique() > 1).sum()),
            "content groups": d2.acc_md5.nunique()}
    upc = tr.groupby(["cls", "id"]).size()
    return {"tr": tr, "dups": dups,
            "outside": tr[tr.cls.isna() | tr.cls.eq("")].groupby("mode").agg(hours=("hours", "sum"), users=("id", "nunique")),
            "summary": {"dataset": "MobilitApp (inventory only)", "clean v1 windows": "– (no windows)",
                        "clean v1 hours (tiles)": "–", "users": tr.id.nunique(), "iOS %": 0.0, "Android %": 100.0,
                        "gyroscope %": "all files", "GPS coverage %": "none",
                        "users per class (>= 5)": "ids with >= 1 trip (no windows): " + ", ".join(
                            f"{k} {upc[k].size}" for k in CLASSES if k in upc.index.get_level_values(0)),
                        "classes outside v1": ", ".join(sorted(mobilitapp.OUTSIDE_V1))}}


# ---------------------------------------------------------------- v1

def v1_rows() -> list[dict]:
    rows = []
    for src, name in (("extrasensory", "v1 ExtraSensory (official, dev + test)"), ("shl_user1", "v1 SHL User 1 (official)")):
        e = bench.examples(src)
        u = e.groupby(["cls", "user"]).size()
        upc = (u >= 5).groupby(level=0).sum()
        rows.append({"dataset": name, "clean v1 windows": len(e), "clean v1 hours (tiles)": f"{len(e) * 15 / 3600:.1f} (minute grid)",
                     "users": e.user.nunique(), "iOS %": 100 * e.platform.eq("iphone").mean(),
                     "Android %": 100 * e.platform.eq("android").mean(), "gyroscope %": "not in benchmark",
                     "GPS coverage %": f"{100 * e.has_location.mean():.1f}",
                     "users per class (>= 5)": ", ".join(f"{k} {upc[k]}" for k in CLASSES if k in upc),
                     "classes outside v1": "– (excluded by label rules)"})
    return rows


def still_threshold() -> float:
    """p95 of window std |acc| over v1's official still examples, dev users only (fixed before the candidates)."""
    s = pd.concat([bench.examples(src).query("split == 'dev' and cls == 'still'").acc_std
                   for src in ("extrasensory", "shl_user1")])
    return float(s.quantile(0.95))


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    still_t = still_threshold()
    res = {}
    for ds, name in (("nor_tmd", "NOR-TMD"), ("collecty", "Collecty")):
        if (PROCESSED / f"{ds}_windows_25hz_w15s" / "minutes.parquet").exists():
            res[ds] = window_dataset(ds, name, still_t)
    if "nor_tmd" in res:
        nor_tmd_raw_units()
    mob = mobilitapp_report(still_t)
    # cross-dataset table
    cols = ["dataset", "clean v1 windows", "clean v1 hours (tiles)", "users", "iOS %", "Android %", "gyroscope %",
            "GPS coverage %", "users per class (>= 5)", "classes outside v1"]
    cross = pd.DataFrame([{k: r[k] for k in cols} for r in res.values()] + [mob["summary"]] + v1_rows())
    md(cross, "cross_dataset", "Cross-dataset summary (clean = tmd.candidates rules; v1 = official examples)", index=False)
    # contributions per dataset
    contrib = []
    for ds, r in res.items():
        cl = r["v1 classes (clean)"]
        contrib.append({"dataset": r["dataset"], "v1 classes, clean": ", ".join(k for k in CLASSES if k in cl),
                        "only vehicle classes": "yes" if set(cl) <= VEHICLES else
                        ("vehicles + bike/run only" if not ({"still", "walk"} & set(cl)) else "no"),
                        "other modes present": r["classes outside v1"],
                        "excluded (conflict)": ", ".join(f"{k} {v.hours:.1f} h / {int(v.users)} users" for k, v in r["excluded"].iterrows())})
    tr = mob["tr"]
    contrib.append({"dataset": "MobilitApp (inventory)", "v1 classes, clean": "none (inventory only)",
                    "only vehicle classes": "–", "other modes present": ", ".join(sorted(mobilitapp.OUTSIDE_V1)),
                    "excluded (conflict)": "–"})
    md(pd.DataFrame(contrib), "contributions", "v1 classes contributed cleanly, other modes, shortcut flag", index=False)
    # modes outside v1 across datasets
    o = []
    for ds, r in res.items():         # MobilitApp: final verdict "not usable" (approved 2026-11-02), not counted
        for mode, v in r["outside"].iterrows():
            o.append({"mode": mode, "dataset": r["dataset"], "hours": v.hours, "users": v.users})
    o = pd.DataFrame(o)
    agg = o.groupby("mode").agg(datasets=("dataset", "nunique"), users=("users", "sum"), hours=("hours", "sum"),
                                per_dataset=("dataset", lambda s: "; ".join(
                                    f"{d}: {o.loc[i, 'hours']:.1f} h, {int(o.loc[i, 'users'])} users" for i, d in s.items())))
    agg["candidate class (>= 10 users, >= 2 datasets)"] = np.where((agg.users >= 10) & (agg.datasets >= 2), "yes", "no")
    md(agg, "modes_outside_v1", "Modes outside v1 across the usable datasets (NOR-TMD, Collecty; hours from legs)")
    dd = pd.DataFrame({r["dataset"]: r["dups"] for r in res.values()} | {"MobilitApp": mob["dups"]}).T
    md(dd, "duplicates", "Duplicates (check 9): identical 15 s windows (window sets) or files / acc content (MobilitApp)")
    (TABLES / "report.md").write_text(f"still-energy threshold (v1 dev still p95 of std |acc|): {still_t:.3f} m/s2\n\n"
                                      + "\n".join(MD))
    print(f"report: {len(MD)} tables -> {TABLES / 'report.md'}; still threshold {still_t:.3f}")
    print(cross[["dataset", "clean v1 windows", "users", "iOS %", "gyroscope %"]].to_string(index=False))


if __name__ == "__main__":
    main()
