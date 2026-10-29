"""Numbers behind decision_register.md, from waiis_bench_v1 (read-only).

Usage: uv run python register_numbers.py
Writes results/tables/decision_details.csv: one row per (id, measure, breakdown, key) with the full numbers per
class, platform, user and split, plus each decision's code location ('code') and commit ('commit'). The IDs are the
register's. stdout: the headline number per ID (the register's Effect column).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from tmd import bench
from tmd.paths import PROJECT

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "tables" / "decision_details.csv"
C7 = ["still", "walk", "run", "bike", "car", "bus", "train"]
NO_GYRO = ["61359772", "CCAF77F0", "F50235E0"]

# id -> (file, anchor text, commit where it was decided / implemented)
WHERE = {
    "S1": ("src/tmd/extrasensory.py", "def uuids", "7f435c5"),
    "S2": ("experiments/04_frozen-benchmark_es-shl/build.py", "def es_user", "9ae5d9c"),
    "S3": ("experiments/04_frozen-benchmark_es-shl/build.py", "def shl_recordings", "a711d65"),
    "S4": ("experiments/04_frozen-benchmark_es-shl/build.py", "r.day not in", "7f435c5"),
    "U1": ("src/tmd/resample.py", "def to_ms2", "1f52628"),
    "U2": ("src/tmd/resample.py", "def decimate_100_to_25", "1f52628"),
    "U3": ("experiments/04_frozen-benchmark_es-shl/build.py", "ACC_PLAUSIBLE = ", "cf1085c"),
    "C1": ("src/tmd/resample.py", "MAX_SPAN_S = ", "1f52628"),
    "C2": ("src/tmd/resample.py", 'info["hole_trimmed"] = True', "b1e2a9c"),
    "C3": ("src/tmd/resample.py", "a = a[np.argsort(a[:, 0]", "b1e2a9c"),
    "W1": ("src/tmd/resample.py", "def centre", "a711d65"),
    "W2": ("experiments/04_frozen-benchmark_es-shl/build.py", 'r["acc_status"] = "short"', "b1e2a9c"),
    "W3": ("experiments/04_frozen-benchmark_es-shl/build.py", 'out_path("extrasensory", "snippets")', "b1e2a9c"),
    "L1": ("src/tmd/extrasensory.py", "PRIORITY = [", "cb8a024"),
    "L2": ("src/tmd/extrasensory.py", '("train", ["orig:ON_A_TRAIN"])', "7f435c5"),
    "L3": ("src/tmd/extrasensory.py", 'hit["car"] = ', "7f435c5"),
    "L4": ("src/tmd/extrasensory.py", '"label_source": clean', "7f435c5"),
    "L5": ("src/tmd/bench.py", "def unlabelled_minutes", "a711d65"),
    "X1": ("src/tmd/extrasensory.py", '"drop:SLEEPING"', "cb8a024"),
    "X2": ("src/tmd/extrasensory.py", '"drop:not_carried"', "cb8a024"),
    "X3": ("experiments/04_frozen-benchmark_es-shl/build.py", "LABEL_QUIET_CLASSES = ", "cf1085c"),
    "X4": ("src/tmd/extrasensory.py", "def cap_resting_still", "cb8a024"),
    "X5": ("src/tmd/extrasensory.py", 'reason[n_moving > 1] = "conflict"', "7f435c5"),
    "X6": ("src/tmd/extrasensory.py", 'CHECK = ["MOTORBIKE"]', "7f435c5"),
    "X7": ("src/tmd/extrasensory.py", "DROP = [", "7f435c5"),
    "X8": ("src/tmd/extrasensory.py", "EXCLUDED_USERS = ", "cf1085c"),
    "P1": ("splits/README.md", "## How the test users were picked", "1d03c9d"),   # picked; adopted in aef0a4c
    "P2": ("src/tmd/splits.py", "def balanced_folds", "1d03c9d"),
    "P3": ("src/tmd/bench.py", "def _guard", "b1e2a9c"),
    "P4": ("experiments/04_frozen-benchmark_es-shl/build.py", "def shl_recordings", "7f435c5"),
    "G1": ("src/tmd/extrasensory.py", "def read_location", "a711d65"),
    "G2": ("src/tmd/extrasensory.py", "def location_missing_reason", "a711d65"),
    "G3": ("experiments/04_frozen-benchmark_es-shl/build.py", '"speed": np.where(spd', "a711d65"),
    "G4": ("experiments/04_frozen-benchmark_es-shl/build.py", 'tr["bearing_from_fixes"] = ', "30550d9"),
    "G5": ("experiments/04_frozen-benchmark_es-shl/build.py", '"time": t.ts.astype(float)', "a711d65"),
    "G6": ("src/tmd/gps.py", "SESSION_GAP_S = ", "a711d65"),
    "H1": ("experiments/04_frozen-benchmark_es-shl/build.py", "centre_ms = minutes", "a711d65"),
    "H2": ("experiments/04_frozen-benchmark_es-shl/build.py", 'r["label_status"] = "mixed"', "a711d65"),
    "H3": ("src/tmd/gps.py", "CADENCE = ", "a711d65"),
    "H4": ("experiments/04_frozen-benchmark_es-shl/build.py", "pick_ts.append", "a711d65"),
    "H5": ("src/tmd/gps.py", "def blank_probability", "b1e2a9c"),
    "H6": ("experiments/04_frozen-benchmark_es-shl/build.py", "stream_s = pa.schema", "b1e2a9c"),
    "F1": ("experiments/04_frozen-benchmark_es-shl/build.py", "def write_manifest", "a711d65"),
    "F2": ("experiments/04_frozen-benchmark_es-shl/build.py", "def acc_schema", "a711d65"),
    "F3": ("experiments/04_frozen-benchmark_es-shl/build.py", "def official_flags", "cf1085c"),
    "F4": ("experiments/04_frozen-benchmark_es-shl/build.py", "src = pd.concat([clean.drop", "a711d65"),
}

rows: list[dict] = []
headline: dict[str, str] = {}


def add(id_, measure, breakdown, key, value):
    rows.append({"id": id_, "measure": measure, "breakdown": breakdown, "key": key, "value": value})


def minutes(id_, measure, df, head=None):
    """Count rows of df (a label frame) in total and per class, platform, split and user."""
    add(id_, measure, "total", "", len(df))
    for col, name in (("cls", "class"), ("platform", "platform"), ("split", "split"), ("user", "user")):
        if col in df:
            for k, v in df[col].replace("", "(none)").value_counts().sort_index().items():
                add(id_, measure, name, k, int(v))
    headline.setdefault(id_, head if head is not None else f"{len(df):,}")


def share(id_, measure, df, flag, by=("cls", "platform")):
    add(id_, measure, "total", "", round(float(df[flag].mean()), 4))
    for col in by:
        for k, v in df.groupby(col)[flag].mean().items():
            add(id_, measure, col, k, round(float(v), 4))


def code_line(path, anchor):
    lines = (PROJECT / path).read_text().splitlines()
    hits = [i + 1 for i, l in enumerate(lines) if anchor in l]
    assert hits, (path, anchor)
    return f"{path}:{hits[0]}"


def main():
    L = bench.read("extrasensory", "labels")
    ex = L[L.official]
    inc = L[L.included_v2]
    sn = pq.read_table(bench.path("extrasensory", "snippets"),
                       columns=["user", "ts", "n25", "unit", "raw_rate_hz", "span_s"]).to_pandas()
    acc = pq.read_table(bench.path("extrasensory", "accelerometer"),
                        columns=["user", "ts", "span_s", "start_offset_s"]).to_pandas()
    sn = sn.merge(L[["user", "ts", "platform", "cls", "split"]], on=["user", "ts"])
    acc = acc.merge(L[["user", "ts", "platform", "cls"]], on=["user", "ts"])
    man = bench.manifest()

    # Sources
    minutes("S1", "minutes", L, f"{len(L):,} min, {L.user.nunique()} users")
    minutes("S2", "official examples of the users without gyroscope", ex[ex.user.str[:8].isin(NO_GYRO)])
    shl = {s: bench.read(s, "labels") for s in ("shl_user1", "shl_user23")}
    for s, lab in shl.items():
        minutes("S3", f"{s} official examples", lab[lab.official].assign(platform="android"))
    headline["S3"] = f"{sum(int(l.official.sum()) for l in shl.values()):,} examples"
    add("S4", "duplicate days kept once", "total", "", 3)
    for d in ("220617", "260617", "270617"):
        add("S4", "duplicate days kept once", "day", d, 1)
    headline["S4"] = "−3 days"

    # Units & resampling
    minutes("U1", "snippets converted from g", sn[sn.unit == "g"])
    headline["U1"] = f"{(sn.unit == 'g').sum():,} snippets"
    minutes("U2", "15 s windows at 25 Hz", L[L.has_acc])
    for p, g in sn.groupby("platform"):
        for q in (0.01, 0.5, 0.99):
            add("U2", "raw rate [Hz]", f"platform p{int(q * 100)}", p, round(float(g.raw_rate_hz.quantile(q)), 1))
    margin = (np.floor(acc.span_s * 25).astype(int) + 1 - 375) // 2
    minutes("U2", "windows within 0.4 s of the filter edge", acc[margin < 10])
    headline["U2"] = f"{int(L.has_acc.sum()):,} windows"
    w = L[L.has_acc & L.included_v2 & ~L.user_excluded & L.acc_implausible]
    minutes("U3", "included windows outside 7-20 m/s2 (excluded user left out)", w)
    for s, lab in shl.items():
        ww = lab[lab.has_acc & lab.included & lab.acc_implausible].assign(platform="android")
        minutes("U3", f"{s} windows outside 7-20 m/s2", ww)
    headline["U3"] = f"−{len(w) + sum(int((l.has_acc & l.included & l.acc_implausible).sum()) for l in shl.values())} windows"

    # Cleaning
    minutes("C1", "minutes without a snippet (no file, span > 40 s)", L[~L.has_snippet],
            f"−{int((~L.has_snippet).sum())} min")
    for k, v in L[~L.has_snippet].acc_status.value_counts().items():
        add("C1", "minutes without a snippet", "reason", k, int(v))
    minutes("C2", "snippets trimmed to the longest clean stretch", L[L.hole_trimmed & L.has_snippet])
    minutes("C2", "official examples from trimmed snippets", ex[ex.hole_trimmed])
    headline["C2"] = f"+{int(ex.hole_trimmed.sum())} examples"
    minutes("C3", "time-repaired snippets (never official)", L[L.time_repaired & L.has_snippet],
            f"−{int((L.time_repaired & L.has_snippet).sum()):,} snippets")
    minutes("C3", "included minutes time-repaired", inc[inc.time_repaired])

    # Windowing
    minutes("W1", "official windows", L[L.has_acc])
    d = acc.assign(discarded_s=acc.span_s - 15)
    for p, g in d.groupby("platform"):
        add("W1", "snippet seconds outside the window (mean)", "platform", p, round(float(g.discarded_s.mean()), 1))
    headline["W1"] = f"{int(L.has_acc.sum()):,} windows"
    minutes("W2", "short snippets (< 15 s), all minutes", L[L.short])
    minutes("W2", "included minutes with a short snippet", inc[inc.short], f"−{int(inc.short.sum()):,} min")
    for k, v in sn[sn.n25 < 375].groupby("user").size().sort_values(ascending=False).items():
        add("W2", "short snippets per user", "user", k, int(v))
    minutes("W3", "snippets stored in full", L[L.has_snippet])

    # Labels
    minutes("L1", "minutes with a class (rules v2 'ok')", L[L.reason == "ok"])
    tr = L[L.cls == "train"]
    minutes("L2", "train minutes from original ON_A_TRAIN", tr)
    add("L2", "train minutes the cleaned CSV calls SITTING", "total", "", int((tr["label:SITTING"] == 1).sum()))
    pas = inc[(inc.cls == "bus") & (inc["label:DRIVE_-_I_M_A_PASSENGER"] == 1)]
    minutes("L3", "bus minutes with 'passenger'", pas)
    car = inc[inc.cls == "car"]
    add("L3", "car minutes via 'passenger' only", "total", "",
        int(((car["label:IN_A_CAR"] != 1) & (car["label:DRIVE_-_I_M_THE_DRIVER"] != 1)).sum()))
    for c, g in ex.groupby("cls"):
        for k, v in g.label_source.value_counts(normalize=True).items():
            add("L4", "label_source share of official examples", f"class {c}", int(k), round(float(v), 3))
    headline["L4"] = f"{(ex.label_source == 2).mean():.0%} retrospective"
    minutes("L5", "unlabelled minutes with a window", L[(L.reason == "unlabelled") & L.has_acc])

    # Exclusions & caps
    for i, r in (("X1", "drop:SLEEPING"), ("X2", "drop:not_carried"), ("X5", "conflict"), ("X6", "check:MOTORBIKE")):
        m = L[L.reason == r]
        minutes(i, f"minutes with reason {r}", m, f"{'−' if i != 'X6' else ''}{len(m):,} min")
    li = inc[inc.label_implausible & ~inc.user_excluded]
    minutes("X3", "included windows: quiet walk / run / bike", li, f"−{len(li):,} windows")
    minutes("X4", "still minutes capped", L[L.capped], f"−{int(L.capped.sum()):,} min")
    st = ex[ex.cls == "still"]
    add("X4", "resting share of official still", "total", "", round(float(st.resting.mean()), 3))
    add("X4", "no position reported, share of official still", "total", "", round(float((st.position == "").mean()), 3))
    other = L[L.reason.str.startswith("drop:") & ~L.reason.isin(["drop:SLEEPING", "drop:not_carried"])]
    minutes("X7", "minutes with another transport / exercise label", other, f"−{len(other):,} min")
    for k, v in other.reason.value_counts().items():
        add("X7", "minutes per label", "label", k.removeprefix("drop:"), int(v))
    minutes("X8", "included minutes of excluded users", inc[inc.user_excluded], f"−{int(inc.user_excluded.sum()):,} min")

    # Splits
    minutes("P1", "official test examples", ex[ex.split == "test"])
    t = ex[ex.split == "test"]
    for c in C7:
        add("P1", "test users with >= 5 examples", "class", c, int((t[t.cls == c].groupby("user").size() >= 5).sum()))
    add("P1", "location coverage", "split", "test", round(float(t.has_location.mean()), 3))
    add("P1", "location coverage", "split", "dev", round(float(ex[ex.split == "dev"].has_location.mean()), 3))
    headline["P1"] = f"{len(t):,} test examples"
    dv = ex[ex.split == "dev"]
    minutes("P2", "official dev examples", dv)
    for (f, c), v in dv.groupby(["fold", "cls"]).size().items():
        add("P2", "dev examples per fold", f"fold {f}", c, int(v))
    from tmd.splits import reserved_test
    rs = set(reserved_test()["extrasensory"].uuid)
    minutes("P3", "minutes with a window kept out of pre-training", L[L.has_acc & L.user.isin(rs)])
    headline["P3"] = f"{len(rs)} + 2 users"
    for s, lab in shl.items():
        add("P4", "official examples", "source", s, int(lab.official.sum()))
    headline["P4"] = f"{int(shl['shl_user1'].official.sum()):,} / {int(shl['shl_user23'].official.sum()):,}"

    # GPS (ExtraSensory)
    share("G1", "official examples with lat/lon", ex, "has_location")
    add("G1", "minutes with lat/lon", "total", "", int(L.has_location.sum()))
    headline["G1"] = f"{L.has_location.mean():.0%} of minutes"
    for k, v in L.location_missing_reason.replace("", "present").value_counts().items():
        add("G2", "minutes per reason", "reason", k, int(v))
    headline["G2"] = f"{int((L.location_missing_reason == 'no_updates').sum()):,} no_updates"
    g = bench.read("extrasensory", "gps_per_minute").merge(ex[["user", "ts", "cls", "platform"]], on=["user", "ts"])
    g["speed_missing"] = g.speed < 0
    g["bearing_missing"] = g.bearing < 0
    share("G3", "official fixes with speed -1", g, "speed_missing")
    headline["G3"] = f"{g.speed_missing.mean():.0%} speed −1"
    share("G4", "official fixes with bearing -1", g, "bearing_missing")
    headline["G4"] = f"{g.bearing_missing.mean():.0%} bearing −1"
    gp = bench.read("extrasensory", "gps_per_minute")
    add("G5", "interval to previous fix [s], median", "total", "", float(gp.dt_prev_s.median()))
    headline["G5"] = f"median {gp.dt_prev_s.median():.0f} s"
    ses = bench.read("extrasensory", "sessions")
    add("G6", "sessions", "total", "", len(ses))
    add("G6", "minutes per session, median", "total", "", float(ses.minutes.median()))
    headline["G6"] = f"{len(ses):,} sessions"

    # SHL
    for s, lab in shl.items():
        for k, v in lab.acc_status.value_counts().items():
            add("H1", f"{s} minutes per window status", "status", k, int(v))
        for k, v in lab.label_status.value_counts().items():
            add("H2", f"{s} minutes per label status", "status", k, int(v))
    headline["H1"] = f"{sum(int(l.has_acc.sum()) for l in shl.values()):,} windows"
    headline["H2"] = f"−{sum(int(l.label_status.isin(['mixed', 'null']).sum()) for l in shl.values()):,} min"
    for s in shl:
        full = len(bench.read(s, "gps_1hz", columns=["time"]))
        for var, i in (("gps_waiis_cadence", "H3"), ("gps_per_minute", "H4")):
            tr = bench.read(s, var).merge(shl[s][["user", "ts", "cls"]], on=["user", "ts"], how="left")
            add(i, f"{s} fixes", "total", "", len(tr))
            add(i, f"{s} 1 Hz fixes", "total", "", full)
            for c, gg in tr.groupby("cls"):
                add(i, f"{s} interval to previous fix [s], median", "class", c or "(none)",
                    float(gg.dt_prev_s.median()))
                add("H5", f"{s} {var} speed missing share", "class", c or "(none)", round(float((gg.speed < 0).mean()), 3))
            add("H5", f"{s} {var} speed missing share", "total", "", round(float((tr.speed < 0).mean()), 3))
    w1 = bench.read("shl_user1", "gps_waiis_cadence")
    headline["H3"] = f"{len(w1):,} fixes (User 1)"
    headline["H4"] = f"{len(bench.read('shl_user1', 'gps_per_minute')):,} fixes (User 1)"
    headline["H5"] = f"{(w1.speed < 0).mean():.0%} speed −1"
    add("H5", "v0 [m/s]", "total", "", man["params"]["blank_v0"])
    for k in ("shl_user1/stream.parquet", "shl_user23/stream.parquet", "shl_user1/accelerometer_other_positions.parquet"):
        add("H6", "rows", "table", k, man["files"][k]["rows"])
    headline["H6"] = f"{(man['files']['shl_user1/stream.parquet']['rows'] + man['files']['shl_user23/stream.parquet']['rows']) / 1e6:.1f} M samples"

    # Format
    add("F1", "files", "total", "", len(man["files"]))
    add("F1", "bytes", "total", "", sum(f["bytes"] for f in man["files"].values()))
    for k, f in man["files"].items():
        add("F1", "rows", "file", k, f["rows"])
        add("F1", "sha256", "file", k, f["sha256"])
    headline["F1"] = f"{len(man['files'])} files, {sum(f['bytes'] for f in man['files'].values()) / 1e9:.1f} GB"
    headline["F2"] = f"{int(L.has_acc.sum()) + sum(int(l.has_acc.sum()) for l in shl.values()):,} windows"
    minutes("F3", "official examples (ExtraSensory)", ex)
    for s, lab in shl.items():
        add("F3", "official examples", "source", s, int(lab.official.sum()))
    n_src = sum(c.startswith(("label:", "orig:")) for c in L.columns)
    add("F4", "source label columns", "total", "", n_src)
    headline["F4"] = f"{n_src} columns"

    for i, (path, anchor, commit) in WHERE.items():
        add(i, "code", "", "", code_line(path, anchor))
        add(i, "commit", "", "", commit)
    out = pd.DataFrame(rows)
    out["order"] = out.id.map({k: n for n, k in enumerate(WHERE)})
    out = out.sort_values(["order"], kind="stable").drop(columns="order")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"{len(out):,} rows -> {OUT.relative_to(HERE)}  ({bench.VERSION}, build commit {man['git_commit'][:7]})")
    for i in WHERE:
        print(f"{i}: {headline.get(i, '–')}")
    missing = set(WHERE) - set(headline)
    assert not missing - {"F2"}, missing


if __name__ == "__main__":
    main()
