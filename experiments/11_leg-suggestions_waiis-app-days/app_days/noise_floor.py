"""Accelerometer noise floor of the WAIIS app days, with the definitions of 10's diag_device.py, against ExtraSensory
Android / iPhone dev users, 4E98F91F and NOR-TMD subway; quiet windows inside or between GPS-covered stretches.
Descriptive; no models. Prints summaries only, never coordinates.

Windows as waiis_bench_v2 (`tmd.candidates.trip_windows`, one 15 s window per clock minute, 25 Hz; tiles of 15 s
back to back as a second count); quiet = std |acc| < 0.03 m/s2. Noise floor = median std |acc| over quiet windows;
quantisation step, raw std / median |acc| from the raw samples of each window (`tmd.resample.raw_window_stats`).
GPS-covered stretch: consecutive fixes <= 60 s apart; a window whose centre lies in a longer interval is between
stretches, split by the speed across that gap as in gaps.py (stationary < 0.5 m/s, moving >= 2).
ExtraSensory Android, 4E98F91F and NOR-TMD subway: 10's results/tables/diag_device_users.csv. ExtraSensory iPhone:
computed here the same way (official still windows of the dev users, up to 60 raw windows per user, quiet first).

    uv run python noise_floor.py [<data root of the WAIIS days>]
"""
from pathlib import Path

import numpy as np
import pandas as pd

from describe import all_days, gps_day, hav
from tmd.readers import waiis
from tmd import bench, candidates
from tmd.readers import extrasensory as es
from tmd.resample import G, raw_window_stats

BENCH = "waiis_bench_v2"
DIAG10 = Path(__file__).resolve().parents[2] / "10_gps-acc-baseline_waiis-bench-v2" / "results" / "tables" / \
    "diag_device_users.csv"
RAW_N = 60
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


def waiis_windows() -> pd.DataFrame:
    rows = []
    gps = pd.concat([gps_day(d) for d in all_days()], ignore_index=True)
    for d in all_days():
        a = waiis.read_imu(d)
        t, xyz = a.t_ms.to_numpy() / 1e3, a[["ax", "ay", "az"]].to_numpy(float)
        lab = np.full(len(t), "", dtype=object)
        g = gps[gps.day == d.date].reset_index(drop=True)
        for grid in ("minute", "tile"):
            w, _ = candidates.trip_windows(t, xyz, t, lab, grid=grid)
            w = w[w.status == "ok"].copy()
            w["quiet"] = w.acc_std < candidates.QUIET_STD
            raw = [raw_window_stats(t[i:j], xyz[i:j]) for i, j in
                   (np.searchsorted(t, [lo, lo + candidates.WIN_S]) for lo in w.t_start)]
            w = pd.concat([w.reset_index(drop=True), pd.DataFrame(raw)], axis=1)
            w["where"] = [gps_place(g, lo + candidates.WIN_S / 2) for lo in w.t_start]
            rows.append(w.assign(day=d.date, grid=grid))
    return pd.concat(rows, ignore_index=True)


def gps_place(g: pd.DataFrame, c: float) -> str:
    """Where a window centre c [s] lies relative to the day's fixes; inside a stretch: the 60 s state of the nearer fix."""
    t = g.t.to_numpy()
    j = np.searchsorted(t, c)
    if j == 0 or j == len(t):
        return "outside the day's fixes"
    a, b = g.iloc[j - 1], g.iloc[j]
    if b.t - a.t <= 60:
        near = a if c - a.t <= b.t - c else b
        return f"inside: {near.state}"
    v = hav(a.lat, a.lon, b.lat, b.lon) / (b.t - a.t)
    return "between: " + ("stationary gap" if v < 0.5 else "moving gap" if v >= 2 else "0.5-2 m/s gap")


def es_iphone() -> pd.DataFrame:
    lab = bench.read("extrasensory", "labels", columns=["user", "ts", "platform", "split", "cls", "official"],
                     version=BENCH)
    acc = bench.read("extrasensory", "accelerometer", columns=["user", "ts", "unit", "raw_rate_hz", "acc_median",
                                                               "acc_std", "quiet"], version=BENCH)
    e = lab.merge(acc, on=["user", "ts"], validate="1:1")
    e = e[(e.platform == "iphone") & (e.cls == "still") & (e.split != "test") & e.official]
    rng = np.random.RandomState(0)
    out = []
    for u, w in e.groupby("user"):
        q, nq = w[w.quiet], w[~w.quiet]
        s = pd.concat([q.sample(min(len(q), RAW_N), random_state=rng),
                       nq.sample(min(len(nq), RAW_N - min(len(q), RAW_N)), random_state=rng)])
        raw = []
        for r in s.itertuples():
            x = es.read_snippet(es.snippet_path(u, int(r.ts)))
            if x is None or len(x) < 10:
                continue
            x = x[~np.isnan(x[:, 1:]).any(axis=1)]
            raw.append({**raw_window_stats(x[:, 0], x[:, 1:] * (G if r.unit == "g" else 1.0)), "quiet": r.quiet})
        raw = pd.DataFrame(raw)
        out.append({"group": "ES iPhone", "user": u[:8], "windows": len(w), "quiet_share": w.quiet.mean(),
                    "rate_hz": w.raw_rate_hz.median(), "acc_median": w.acc_median.median(),
                    "std_quiet_median": q.acc_std.median() if len(q) else np.nan,
                    "quant_step": raw.quant_step.median(),
                    "raw_std_quiet_median": raw[raw.quiet].raw_std.median() if raw.quiet.any() else np.nan})
    return pd.DataFrame(out)


def main():
    w = waiis_windows()
    m = w[w.grid == "minute"]
    print("== WAIIS windows (ok status), per day and grid")
    print(w.groupby(["day", "grid"]).agg(windows=("quiet", "size"), quiet=("quiet", "sum")).to_string())
    rows = []
    for day, d in list(m.groupby("day")) + [("both", m)]:
        q = d[d.quiet]
        rows.append({"group": f"WAIIS {day}", "windows": len(d), "quiet_share": d.quiet.mean(),
                     "rate_hz": d.raw_rate_hz.median(), "acc_median": d.acc_median.median(),
                     "acc_median_quiet": q.acc_median.median(), "std_quiet_median": q.acc_std.median(),
                     "std_quiet_p10_p90": f"{q.acc_std.quantile(.1):.4f}-{q.acc_std.quantile(.9):.4f}",
                     "quant_step": d.quant_step.median(), "distinct_share": d.distinct_share.median(),
                     "raw_std_quiet_median": q.raw_std.median()})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    t10 = pd.read_csv(DIAG10)
    t10 = t10[t10.windows >= 20]
    ip = es_iphone()
    cols = ["quiet_share", "rate_hz", "acc_median", "std_quiet_median", "quant_step", "raw_std_quiet_median"]
    ref = pd.concat([t10[t10.group.isin(["4E98F91F", "ES Android, other", "NOR-TMD subway"])], ip[ip.windows >= 20]])
    s = ref.groupby("group")[cols].agg(["median", "min", "max"]).round(4)
    s["users"] = ref.groupby("group").size()
    print("\n== reference groups (users with >= 20 windows; median, min, max over users)")
    print(s.to_string())
    allq = ref.std_quiet_median.dropna()
    v = m[m.quiet].acc_std.median()
    print(f"\nWAIIS quiet std median {v:.4f}: below {(allq > v).sum()} of {len(allq)} reference users with quiet windows; "
          f"ES iPhone users below it: {(ip.std_quiet_median < v).sum()} of {ip.std_quiet_median.notna().sum()}")

    print("\n== where the quiet windows lie relative to GPS (minute and tile grids)")
    tq = w[w.quiet]
    c = pd.crosstab(tq["where"], tq.grid)
    print(c.to_string())
    print("\nall windows by place:")
    print(pd.crosstab(w["where"], w.grid, margins=True).to_string())
    print("\nquiet minute windows by day and place:")
    print(pd.crosstab(m[m.quiet]["where"], m[m.quiet].day).to_string())
    print("\nquiet std by place (minute grid):", m[m.quiet].groupby("where").acc_std.median().round(4).to_dict())


if __name__ == "__main__":
    main()
