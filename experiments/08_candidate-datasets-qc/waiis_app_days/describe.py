"""Descriptive look at two WAIIS app days (GPS + accelerometer); README.md has the results. Prints summaries
only, never coordinates.

    uv run python describe.py [<package dir>]      default: <data>/waiis/m-waiis-days-2026-10-30_31
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tmd.paths import DATA

B = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "waiis" / "m-waiis-days-2026-10-30_31"
TZ = "Europe/Madrid"
pd.set_option("display.width", 200)


def hav(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def state(v):
    return pd.cut(v, [-np.inf, 0.5, 2.5, 4.0, np.inf], right=False, labels=["still", "walk", "2.5-4", "vehicle"])


def baseline_speed(g, forward=True, base=60, max_span=180):
    """Displacement speed over the first fix >= base s ahead (or behind); NaN if that fix is > max_span s away."""
    t = g.t.to_numpy()
    out = np.full(len(g), np.nan)
    span = np.full(len(g), np.nan)
    for i in range(len(g)):
        if forward:
            j = np.searchsorted(t, t[i] + base)
            if j >= len(g):
                continue
        else:
            j = np.searchsorted(t, t[i] - base, side="right") - 1
            if j < 0:
                continue
        dt = abs(t[j] - t[i])
        if dt > max_span:
            continue
        out[i] = hav(g.lat.iat[i], g.lon.iat[i], g.lat.iat[j], g.lon.iat[j]) / dt
        span[i] = dt
    return out, span


def q(s, ps=(0.05, 0.25, 0.5, 0.75, 0.95)):
    s = pd.Series(s).dropna()
    return "n=%d " % len(s) + " ".join(f"p{int(p*100)}={s.quantile(p):.1f}" for p in ps) if len(s) else "n=0"


def gps_day(f):
    g = pd.read_parquet(f).sort_values("t_ms").reset_index(drop=True)
    g["t"] = g.t_ms / 1000
    g["dt_next"] = g.t.shift(-1) - g.t
    g["v_fwd"], _ = baseline_speed(g, True)
    g["v_bwd"], _ = baseline_speed(g, False)
    g["v60"] = g.v_fwd.fillna(g.v_bwd)            # state of a fix: 60 s baseline ahead, else behind
    g["state"] = state(g.v60).astype(object).fillna("unknown")
    d = hav(g.lat, g.lon, g.lat.shift(-1), g.lon.shift(-1))
    g["v_next"] = d / g.dt_next                    # consecutive-fix derived speed
    g["day"] = f.parent.name.split("=")[1]
    return g


def main():
    days = sorted(B.glob("user=*/date=*"))
    G = pd.concat([gps_day(d / "gps.parquet") for d in days], ignore_index=True)
    print("== GPS")
    print("columns:", list(pd.read_parquet(days[0] / "gps.parquet").columns))
    for day, g in G.groupby("day"):
        lt = pd.to_datetime(g.t_ms, unit="ms", utc=True).dt.tz_convert(TZ)
        print(f"{day}: {len(g)} fixes, local {lt.min():%H:%M}–{lt.max():%H:%M} ({lt.min():%Z}), t_ms whole seconds: "
              f"{(g.t_ms % 1000 == 0).all()}")
    print("activity x is_moving:", G.groupby(["activity", "is_moving"]).size().to_dict())
    print("null share:", G[["accuracy_m", "speed_mps", "speed_accuracy_mps", "heading_deg", "altitude_m",
                            "activity_confidence", "event"]].isna().mean().round(3).to_dict())

    iv = G.dropna(subset=["dt_next"])
    print("\n-- interval to next fix [s], overall:", q(iv.dt_next))
    bins = [0, 2, 5, 10, 15, 20, 30, 60, 300, 1800, np.inf]
    print(pd.cut(iv.dt_next, bins, right=False).value_counts(sort=False).to_string())
    print("by state of the fix (60 s baseline):")
    for s, x in iv.groupby("state"):
        print(f"  {s:8s} {q(x.dt_next)}  share<=20s={(x.dt_next <= 20).mean():.2f}")
    print("state counts:", G.state.value_counts().to_dict())

    print("\n-- gaps > 60 s")
    gaps = []
    for day, g in G.groupby("day"):
        g = g.reset_index(drop=True)
        for i in np.flatnonzero(g.dt_next > 60):
            a, b = g.iloc[i], g.iloc[i + 1]
            across = hav(a.lat, a.lon, b.lat, b.lon)
            gaps.append({"day": day, "start": pd.to_datetime(a.t_ms, unit="ms", utc=True).tz_convert(TZ).strftime("%H:%M"),
                         "len_s": int(a.dt_next), "v_before": a.v_bwd, "v_after": b.v_fwd,
                         "across_m": across, "v_across": across / a.dt_next,
                         "rep_before": a.speed_mps, "rep_after": b.speed_mps,
                         "acc_before": a.accuracy_m, "acc_after": b.accuracy_m})
    gp = pd.DataFrame(gaps)
    gp["st_before"] = state(gp.v_before).astype(object).fillna("unknown")
    gp["st_after"] = state(gp.v_after).astype(object).fillna("unknown")
    print(f"{len(gp)} gaps; length {q(gp.len_s)}")
    print(pd.cut(gp.len_s, [60, 120, 300, 900, 3600, np.inf], right=False).value_counts(sort=False).to_string())
    print("before -> after:", gp.groupby(["st_before", "st_after"]).size().to_dict())
    print(gp.round(1).to_string(index=False))

    print("\n-- speed_mps missing by state:", G.groupby("state").speed_mps.apply(lambda s: round(s.isna().mean(), 2)).to_dict())
    print("   missing by activity:", G.groupby("activity").speed_mps.apply(lambda s: round(s.isna().mean(), 2)).to_dict())
    both = G.dropna(subset=["speed_mps", "v_next"])
    both = both[both.dt_next <= 30]
    print(f"reported vs derived (consecutive fixes <= 30 s apart, n={len(both)}): "
          f"corr={both.speed_mps.corr(both.v_next):.2f}, median(rep - der)={np.median(both.speed_mps - both.v_next):.2f} m/s")
    b60 = G.dropna(subset=["speed_mps", "v60"])
    print(f"reported vs 60 s baseline (n={len(b60)}): corr={b60.speed_mps.corr(b60.v60):.2f}")
    for s, x in b60.groupby("state"):
        print(f"  {s:8s} n={len(x):3d} reported {q(x.speed_mps, (0.25, 0.5, 0.75))} | derived60 {q(x.v60, (0.25, 0.5, 0.75))}")
    print("reported speed range:", q(G.speed_mps, (0.5, 0.9, 0.99, 1.0)))

    print("\n-- accuracy_m by state")
    for s, x in G.groupby("state"):
        print(f"  {s:8s} {q(x.accuracy_m)}  >50m={(x.accuracy_m > 50).mean():.2f}")
    print(f"overall {q(G.accuracy_m)}  >50m={(G.accuracy_m > 50).mean():.3f}; ==2000 m: {(G.accuracy_m >= 2000).sum()}")
    print("top accuracy values:", G.accuracy_m.round(2).value_counts().head(5).to_dict())

    print("\n== Accelerometer")
    for d in days:
        a = pd.read_parquet(d / "imu.parquet").sort_values("t_ms").reset_index(drop=True)
        dt = a.t_ms.diff()
        burst = a.gap.cumsum() + (0 if a.gap.iat[0] else 1)
        inb = dt[~a.gap].iloc[1:]
        print(f"{d.name}: {len(a)} samples, {burst.nunique()} bursts, gap rows {a.gap.sum()}, "
              f"dup t {a.t_ms.duplicated().sum()}, sorted {pd.read_parquet(d / 'imu.parquet').t_ms.is_monotonic_increasing}")
        print("  dt in bursts [ms]:", inb.value_counts().sort_index().head(12).to_dict(), "…",
              f"p1={inb.quantile(.01):.0f} p99={inb.quantile(.99):.0f} max={inb.max():.0f} >1000: {(inb > 1000).sum()}")
        by = a.groupby(burst).agg(n=("t_ms", "size"), t0=("t_ms", "min"), t1=("t_ms", "max"))
        by["dur"] = (by.t1 - by.t0) / 1000
        by["hz"] = (by.n - 1) / by.dur
        print(f"  burst duration [s] {q(by.dur)}; real rate per burst [Hz] {q(by.hz[by.dur > 10])}")
        print(f"  gaps between bursts [s] {q(dt[a.gap] / 1000)}")
        m = np.sqrt(a.ax ** 2 + a.ay ** 2 + a.az ** 2)
        sat = (a[["ax", "ay", "az"]].abs() >= 65).any(axis=1)
        print(f"  saturated samples {sat.sum()} ({sat.mean():.4f}); null {a[['ax','ay','az']].isna().any(axis=1).sum()}")
        # still: 5 s blocks (125 samples) inside one burst with std |acc| < 0.05 m/s2
        blk = pd.DataFrame({"b": burst, "k": a.groupby(burst).cumcount() // 125, "m": m,
                            "ax": a.ax, "ay": a.ay, "az": a.az})
        s = blk.groupby(["b", "k"]).agg(n=("m", "size"), sd=("m", "std"), mm=("m", "median"),
                                        ax=("ax", "median"), ay=("ay", "median"), az=("az", "median"))
        s = s[s.n == 125]
        still = s[s.sd < 0.05]
        print(f"  5 s blocks {len(s)}, still (std|acc|<0.05) {len(still)}; median |acc| still = "
              f"{still.mm.median():.3f} m/s2 (p5 {still.mm.quantile(.05):.3f}, p95 {still.mm.quantile(.95):.3f}); all samples median {m.median():.2f}")
        flat = still[(still.az.abs() > 9.3) & (still[["ax", "ay"]].abs().max(axis=1) < 1.5)]
        print(f"  still & flat (|az|>9.3, |ax|,|ay|<1.5): {len(flat)} blocks, az median {flat.az.median() if len(flat) else np.nan:.2f}, "
              f"az>0 in {(flat.az > 0).sum()}")
        dom = still[["ax", "ay", "az"]].abs().idxmax(axis=1)
        print("  still blocks by dominant axis and sign:",
              {f"{c}{'+' if still.loc[dom == c, c].median() > 0 else '-'}": int((dom == c).sum()) for c in ["ax", "ay", "az"] if (dom == c).any()},
              "| all blocks with sd<0.3:", s[s.sd < 0.3].shape[0])


if __name__ == "__main__":
    main()
