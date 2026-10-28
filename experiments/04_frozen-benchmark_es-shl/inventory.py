"""Stop-point 1: inventory of what the benchmark will be built from (nothing is written to processed/).

Usage: uv run python inventory.py
Reads the per-minute acc audit of experiment 02 (<data>/processed/es_minutes_audit.parquet, made by
02's audit_es.py from raw_acc), the label CSVs (rules v2), the location columns of the main CSV, the absolute
location files and the SHL Hips_Location.txt files. Tables go to results/inventory/, the full log to
logs/inventory.log; stdout gets a short summary.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tmd import extrasensory as es
from tmd import shl
from tmd.paths import ES_RAW, PROCESSED

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "inventory"
FS = 25
WIN = 15 * FS                                  # 375 samples
MIN_SPAN_S = (WIN - 1) / FS                    # 14.96 s of snippet give 375 samples at 25 Hz
CLASS_ORDER = ["still", "walk", "run", "bike", "car", "bus", "train"]

log = logging.getLogger("inventory")


def say(msg: str = "") -> None:
    print(msg)
    log.info(msg)


def acc_status(a: pd.DataFrame) -> pd.Series:
    """02's snippet cleaning (tmd.resample.es_snippet_to_25hz) on raw_acc alone, plus the 15 s rule.
    First failing reason wins, in the resampler's order."""
    s = pd.Series("ok", index=a.index, dtype=object)
    s[a.acc_dur_s < MIN_SPAN_S] = "under_15s"
    s[a.acc_nan > 0] = "nan"
    s[a.acc_dt_max_ms > 500] = "hole_over_0.5s"
    s[a.acc_dur_s > 40] = "span_over_40s"
    s[a.acc_dt_nonpos > 0] = "time_not_increasing"
    s[~a.acc_file] = "no_file"
    return s


def location_table(users) -> pd.DataFrame:
    parts = []
    for u in users:
        p = ES_RAW / "per_uuid_features_labels" / f"{u}.features_labels.csv"
        p = p / p.name if p.is_dir() else p
        loc = pd.read_csv(p, usecols=lambda c: c == "timestamp" or c.startswith("location"))
        q = ES_RAW / "per_uuid_absolute_location" / f"{u}.absolute_locations.csv"
        q = q / q.name if q.is_dir() else q
        ab = pd.read_csv(q)
        if not ab.timestamp.isin(loc.timestamp).all():
            say(f"  {u}: {(~ab.timestamp.isin(loc.timestamp)).sum()} absolute-location rows without a CSV minute")
        loc = loc.merge(ab, on="timestamp", how="left", indicator="abs_row")
        parts.append(loc.assign(user=u))
    out = pd.concat(parts, ignore_index=True).rename(columns={"timestamp": "ts"})
    out["abs_row"] = out.abs_row == "both"
    return out


def share(x: pd.Series) -> str:
    return f"{100 * x.mean():.1f}%"


def es_inventory() -> pd.DataFrame:
    say("== ExtraSensory ==")
    users = es.uuids()
    plat = es.platforms()
    mins = es.minutes_table(users)
    ok = mins.reason == "ok"
    keep = np.zeros(len(mins), bool)
    keep[np.flatnonzero(ok)] = es.cap_resting_still(mins[ok], seed=0)       # as 02 (eval_tables.py)
    mins["included_v2"] = keep
    mins["capped"] = ok & ~keep
    mins["platform"] = mins.user.map(plat)

    a = pd.read_parquet(PROCESSED / "es_minutes_audit.parquet",
                        columns=["uuid", "timestamp", "acc_file", "acc_n", "acc_nan", "acc_dur_s",
                                 "acc_dt_med_ms", "acc_dt_max_ms", "acc_dt_nonpos", "acc_norm_med"])
    a = a.rename(columns={"uuid": "user", "timestamp": "ts"})
    m = mins.merge(a, on=["user", "ts"], how="left", validate="one_to_one")
    m["acc_status"] = acc_status(m)
    m["n25"] = np.floor(m.acc_dur_s * FS).fillna(-1).astype(int) + 1       # samples on the 25 Hz grid
    say(f"users {m.user.nunique()} ({m.groupby('platform').user.nunique().to_dict()}), minutes {len(m):,}, "
        f"included_v2 {m.included_v2.sum():,}, capped {m.capped.sum():,}")

    # Snippet length at 25 Hz, per platform (snippets that pass cleaning except the 15 s rule).
    clean = m[m.acc_status.isin(["ok", "under_15s"])]
    q = clean.groupby("platform").n25.describe(percentiles=[.01, .05, .1, .25, .5, .75, .9]).round(0)
    q.to_csv(OUT / "es_snippet_len25_by_platform.csv")
    say("\nSamples per clean acc snippet at 25 Hz (375 needed):\n" + q.to_string())
    bins = [0, 100, 200, 300, 350, 375, 400, 450, 500, 600, 1001]
    h = pd.crosstab(pd.cut(clean.n25, bins, right=False), clean.platform)
    h.to_csv(OUT / "es_snippet_len25_hist.csv")
    say("\n" + h.to_string())

    # Drop reasons, all minutes and included_v2 minutes, per platform.
    for name, sub in [("all minutes", m), ("included_v2", m[m.included_v2])]:
        t = pd.crosstab(sub.acc_status, sub.platform, margins=True)
        t.to_csv(OUT / f"es_acc_status_{name.replace(' ', '_')}.csv")
        say(f"\nacc status, {name}:\n" + t.to_string())
    inc = m[m.included_v2]
    t = pd.crosstab(inc.cls, inc.acc_status).reindex(CLASS_ORDER)
    t["kept_share"] = (t["ok"] / t.sum(axis=1)).round(3)
    t.to_csv(OUT / "es_acc_status_by_class.csv")
    say("\nacc status by class (included_v2):\n" + t.to_string())

    # Users losing most to the 15 s rule (Android users at ~100 Hz give 8 s snippets).
    u = m[m.acc_status.isin(["ok", "under_15s"])].groupby(["user", "platform"]).agg(
        clean=("acc_status", "size"), under_15s=("acc_status", lambda s: (s == "under_15s").sum()),
        dt_med_ms=("acc_dt_med_ms", "median"))
    u["share_under"] = (u.under_15s / u.clean).round(3)
    u = u.sort_values("under_15s", ascending=False)
    u.to_csv(OUT / "es_users_under_15s.csv")
    say("\nUsers with most snippets under 15 s:\n" + u[u.under_15s > 0].head(12).to_string())

    # The three users without gyroscope.
    nog = ["61359772", "CCAF77F0", "F50235E0"]
    g = m[m.user.str[:8].isin(nog)]
    say("\nGyro-less users, included_v2 minutes with a usable 15 s acc window, by class:\n" + pd.crosstab(
        g[g.included_v2 & (g.acc_status == "ok")].user.str[:8], g[g.included_v2 & (g.acc_status == "ok")].cls)
        .to_string())
    return m


def es_location(m: pd.DataFrame) -> None:
    say("\n== ExtraSensory location ==")
    loc = location_table(sorted(m.user.unique()))
    m = m.merge(loc, on=["user", "ts"], how="left", validate="one_to_one")
    m["has_abs"] = m.latitude.notna()
    m["has_feat"] = m["location:num_valid_updates"].fillna(0) > 0
    say(f"absolute-location rows: {m.abs_row.sum():,} of {len(m):,} minutes; lat/lon present {m.has_abs.sum():,}; "
        f"location features with num_valid_updates > 0: {m.has_feat.sum():,}")
    say(f"lat present but no location features: {(m.has_abs & ~m.has_feat).sum():,}; "
        f"features but no lat: {(~m.has_abs & m.has_feat).sum():,}")

    inc = m[m.included_v2]
    cov = inc.groupby("cls").agg(minutes=("ts", "size"), has_latlon=("has_abs", "mean"),
                                 has_loc_features=("has_feat", "mean")).reindex(CLASS_ORDER)
    for p in ["iphone", "android"]:
        cov[f"latlon_{p}"] = inc[inc.platform == p].groupby("cls").has_abs.mean()
    mb = m[m.reason == "check:MOTORBIKE"]
    cov.loc["motorbike (check)"] = [len(mb), mb.has_abs.mean(), mb.has_feat.mean(),
                                    mb[mb.platform == "iphone"].has_abs.mean(),
                                    mb[mb.platform == "android"].has_abs.mean()]
    cov.loc["all minutes"] = [len(m), m.has_abs.mean(), m.has_feat.mean(),
                              m[m.platform == "iphone"].has_abs.mean(), m[m.platform == "android"].has_abs.mean()]
    cov.round(3).to_csv(OUT / "es_location_coverage_by_class.csv")
    say("\nLocation coverage, included_v2 minutes:\n" + cov.round(3).to_string())

    # Units / invalid values of the speed and accuracy features, per platform (minutes with updates only).
    f = m[m.has_feat]
    rows = []
    for p, g in f.groupby("platform"):
        for c in ["location:min_speed", "location:max_speed", "location:best_horizontal_accuracy",
                  "location:num_valid_updates"]:
            x = g[c]
            rows.append({"platform": p, "column": c, "n": len(x), "nan": x.isna().mean(), "neg": (x < 0).mean(),
                         "eq_-1": (x == -1).mean(), "p50": x[x >= 0].median(), "p95": x[x >= 0].quantile(.95),
                         "max": x.max()})
    r = pd.DataFrame(rows).round(3)
    r.to_csv(OUT / "es_location_feature_values.csv", index=False)
    say("\nLocation feature values (minutes with updates):\n" + r.to_string(index=False))

    f = f.assign(speed_valid=f["location:max_speed"] >= 0)
    s = f[f.included_v2].groupby(["cls", "platform"]).agg(
        n=("ts", "size"), speed_valid=("speed_valid", "mean"),
        max_speed_p50=("location:max_speed", lambda x: x[x >= 0].median()),
        max_speed_p90=("location:max_speed", lambda x: x[x >= 0].quantile(.9)),
        acc_p50=("location:best_horizontal_accuracy", lambda x: x[x >= 0].median()),
        updates_p50=("location:num_valid_updates", "median")).round(2)
    s.to_csv(OUT / "es_location_speed_by_class.csv")
    say("\nSpeed / accuracy by class and platform (included_v2 minutes with updates):\n" + s.to_string())

    # Gaps between consecutive minutes with lat/lon (one fix per minute at most).
    g = m[m.has_abs].sort_values(["user", "ts"])
    d = g.groupby("user").ts.diff().dropna()
    say(f"\nGap between consecutive lat/lon minutes [s]: p50 {d.median():.0f}, p90 {d.quantile(.9):.0f}; "
        f"share > 120 s (new session): {(d > 120).mean():.3f}")


def shl_location() -> None:
    say("\n== SHL Hips_Location.txt (cols: time ms, -, -, accuracy m, lat, lon, altitude; no speed/bearing) ==")
    rows = []
    for session in ["User1", "User2_4phones", "User3_4phones"]:
        for rec in shl.recordings(session):
            p = rec.path / "Hips_Location.txt"
            if not p.exists():
                rows.append({"session": session, "rec": rec.rec, "fixes": 0})
                continue
            x = pd.read_csv(p, sep=" ", header=None)
            t = x[0].to_numpy()
            dt = np.diff(t) / 1e3
            rows.append({"session": session, "rec": rec.rec, "ncol": x.shape[1], "fixes": len(x),
                         "dur_h": (t[-1] - t[0]) / 3.6e6, "dt_p50_s": np.median(dt) if len(dt) else np.nan,
                         "dt_p99_s": np.percentile(dt, 99) if len(dt) else np.nan,
                         "gaps_over_60s": int((dt > 60).sum()), "acc_p50_m": x[3].median(),
                         "acc_p90_m": x[3].quantile(.9), "latlon_nan": int(x[[4, 5]].isna().any(axis=1).sum())})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "shl_hips_location.csv", index=False)
    agg = r.groupby("session").agg(recs=("rec", "size"), fixes=("fixes", "sum"), dur_h=("dur_h", "sum"),
                                   dt_p50_s=("dt_p50_s", "median"), gaps_over_60s=("gaps_over_60s", "sum"),
                                   acc_p50_m=("acc_p50_m", "median"), latlon_nan=("latlon_nan", "sum"),
                                   ncol=("ncol", "max")).round(2)
    say(agg.to_string())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (HERE / "logs").mkdir(exist_ok=True)
    logging.basicConfig(filename=HERE / "logs" / "inventory.log", filemode="w", level=logging.INFO,
                        format="%(message)s")
    m = es_inventory()
    es_location(m)
    shl_location()


if __name__ == "__main__":
    main()
