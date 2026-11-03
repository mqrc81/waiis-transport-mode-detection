"""Build the frozen benchmark waiis_bench_v1 (04 README). One command rebuilds everything:

    uv run python experiments/04_frozen-benchmark_es-shl/build.py [--workers 8] [--sources ...]
        [--tables-dir DIR] [--log-dir DIR]

Moved here from experiments/04_frozen-benchmark_es-shl/build.py in experiment 09 (04's build.py is now a driver);
the code is unchanged apart from imports and the two optional output folders (defaults: 04's, as before).

Writes <data>/processed/waiis_bench_v1/<source>/<table>.parquet and manifest.json (parameters, git commit, sha256 of
every file; read them with `tmd.bench`). Count tables for the README go to results/tables/, the full log to
logs/build.log; stdout gets a short summary. Existing output of a rebuilt source is overwritten.
Every decision is listed in decision_register.md.

- Accelerometer: raw_acc only, units per snippet, 25 Hz per snippet; holes > 0.5 s -> longest clean stretch;
  non-increasing timestamps repaired but never official; one centred 15 s window per snippet; short snippets kept,
  never official. Flags: acc_implausible (window median |acc| outside 7-20 m/s2), quiet (std |acc| < 0.03 m/s2),
  label_implausible (quiet walk / run / bike), user_excluded (tmd.readers.extrasensory.EXCLUDED_USERS). `official` = the
  official example set.
- Labels: rules v2 (tmd.readers.extrasensory); splits from splits/es_users_v2.csv, splits/shl_days.csv (here) and
  splits/reserved_test_v2.json (project root).
- GPS: ExtraSensory one fix per minute; SHL at the WAIIS cadence and per minute, speed / bearing blanked with a
  probability that falls with speed (~23% overall).
- Stored, never official: full ExtraSensory snippets, the continuous SHL hips stream, SHL User 1 hand / torso / bag
  windows, the SHL 1 Hz track and the unblanked speed.
"""
import argparse
import json
import logging
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tmd import bench, gps
from tmd.readers import shl
from tmd.readers import extrasensory as es
from tmd.labels import CLASSES, CLASS_ID
from tmd.paths import PROJECT
from tmd.resample import DOWN, centre, decimate_100_to_25, es_acc_repair_to_25hz, window_stats
from tmd.splits import reserved_test

VERSION = "waiis_bench_v1"
RESERVED_V1 = 2                                              # splits/reserved_test_v2.json
ROOT = bench.root(VERSION)
HERE = PROJECT / "experiments" / "04_frozen-benchmark_es-shl"    # v1's experiment folder (splits, tables, logs)
RES = HERE / "results" / "tables"
ES_SPLIT = HERE / "splits" / "es_users_v2.csv"       # approved 2026-10-28
SHL_SPLIT = HERE / "splits" / "shl_days.csv"         # SHL day folds: 02's file, copied unchanged
WIN, FS = bench.WIN, bench.FS
SEED = 0
SPEED_HALF_S = 1.5             # speed / bearing of a kept SHL fix: from the 1 Hz track over ~3 s around it
CADENCE_HALF_S = 5.0           # cadence choice: speed over ~10 s around the candidate fix
ACC_PLAUSIBLE = (7.0, 20.0)    # m/s2, window median |acc|
LABEL_QUIET_CLASSES = ["walk", "run", "bike"]   # quiet window = phone not moving with the person
QUIET_STD = 0.03               # m/s2, std of |acc| over the window
OTHER_POSITIONS = ["Hand", "Torso", "Bag"]
OTHER_POSITION_SOURCES = ("shl_user1",)   # sources whose 4phones days get other-position windows (v2: + shl_user23)
COMPRESSION = "zstd"
CLS_NAME = {v: k for k, v in CLASS_ID.items()}
OFFICIAL_ACC = ("ok", "hole_trimmed")

log = logging.getLogger("build")


def say(msg: str = "") -> None:
    print(msg)
    log.info(msg)


def out_path(source: str, table: str) -> Path:
    p = ROOT / source / f"{table}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write(df: pd.DataFrame, source: str, table: str) -> None:
    df.to_parquet(out_path(source, table), index=False, compression=COMPRESSION)


def acc_schema(extra: list[pa.Field], fixed: bool = True) -> pa.Schema:
    arr = pa.list_(pa.float32(), WIN) if fixed else pa.list_(pa.float32())
    return pa.schema([("user", pa.string()), ("ts", pa.int64()), ("x", arr), ("y", arr), ("z", arr)] + extra)


def acc_batch(schema: pa.Schema, meta: pd.DataFrame, w) -> pa.RecordBatch:
    """meta: one row per window with the schema's non-array columns; w: (n, 375, 3) array (fixed schema) or a list
    of (n_i, 3) arrays (variable schema)."""
    cols = {}
    for f in schema:
        if f.name in ("x", "y", "z"):
            i = "xyz".index(f.name)
            if isinstance(f.type, pa.FixedSizeListType):
                flat = pa.array(np.ascontiguousarray(w[:, :, i]).ravel(), pa.float32())
                cols[f.name] = pa.FixedSizeListArray.from_arrays(flat, WIN)
            else:
                lens = np.array([len(a) for a in w], np.int32)
                off = pa.array(np.concatenate([[0], np.cumsum(lens)]).astype(np.int32))
                vals = pa.array(np.concatenate([a[:, i] for a in w]).astype(np.float32) if len(w) else
                                np.zeros(0, np.float32))
                cols[f.name] = pa.ListArray.from_arrays(off, vals)
        else:
            cols[f.name] = pa.array(meta[f.name].to_numpy(), f.type)
    return pa.RecordBatch.from_pydict(cols, schema=schema)


def stats_columns(w: np.ndarray) -> dict:
    med, std = window_stats(w) if len(w) else (np.zeros(0), np.zeros(0))
    return {"acc_median": med.astype(np.float32), "acc_std": std.astype(np.float32),
            "acc_implausible": (med < ACC_PLAUSIBLE[0]) | (med > ACC_PLAUSIBLE[1]), "quiet": std < QUIET_STD}


STAT_FIELDS = [pa.field("acc_median", pa.float32()), pa.field("acc_std", pa.float32()),
               pa.field("acc_implausible", pa.bool_()), pa.field("quiet", pa.bool_())]


def official_flags(lab: pd.DataFrame, included: str) -> None:
    """label_implausible (quiet window in walk / run / bike: the label describes the person, not the phone, as for
    not_carried) and `official` = the official example set, in place."""
    lab["label_implausible"] = lab.has_acc & lab.quiet & lab.cls.isin(LABEL_QUIET_CLASSES)
    lab["official"] = (lab[included] & lab.has_acc & ~lab.acc_implausible & ~lab.label_implausible
                       & ~lab.user_excluded)


# ---------------------------------------------------------------- ExtraSensory

def es_minutes() -> pd.DataFrame:
    """Rules v2 for every minute of all 60 users, flags, split and fold (es_users_v2.csv; test = reserved)."""
    m = es.minutes_table(es.uuids())
    ok = m.reason == "ok"
    keep = np.zeros(len(m), bool)
    keep[np.flatnonzero(ok)] = es.cap_resting_still(m[ok], seed=SEED)    # as 02 (eval_tables.py)
    m["included_v2"] = keep
    m["capped"] = ok & ~keep
    m["not_carried"] = m.reason == "drop:not_carried"
    m["motorbike"] = m.reason == "check:MOTORBIKE"
    m["y"] = np.where(keep, m.y, 0)
    users = pd.read_csv(ES_SPLIT)
    test = set(reserved_test(RESERVED_V1)["extrasensory"].uuid)
    assert set(users.uuid[users.partition == "test"]) == test, "es_users_v2.csv test rows differ from reserved_test()"
    assert set(users.uuid) == set(m.user)
    u = users.set_index("uuid")
    m["platform"] = m.user.map(u.platform)
    m["split"] = m.user.map(u.partition)
    m["fold"] = m.user.map(u.fold).astype(np.int8)
    m = m.sort_values(["user", "ts"]).reset_index(drop=True)
    m["session"] = m.groupby("user").ts.transform(lambda t: gps.session_ids(t.to_numpy())).astype(np.int32)
    return m


def es_user(uuid: str):
    """One user: per-minute snippet status, the official windows, every resampled snippet, source label columns,
    location columns."""
    clean = es.read_cleaned_labels(uuid)
    orig = es.read_original_labels(uuid).reindex(clean.index)
    src = pd.concat([clean.drop(columns="label_source").add_prefix("label:"),
                     orig.drop(columns="label_source", errors="ignore").add_prefix("orig:")], axis=1)
    rows, wins, snips = [], [], []
    for ts in clean.index.astype(int):
        a = es.read_snippet(es.snippet_path(uuid, ts, "acc"))
        r = {"ts": ts, "acc_status": "no_file", "unit": "", "raw_n": 0, "raw_rate_hz": np.nan, "span_s": np.nan,
             "n25": 0, "time_repaired": False, "hole_trimmed": False, "start_offset_s": np.nan, "window_start": -1}
        if a is not None:
            x, unit, info = es_acc_repair_to_25hz(a)
            r.update(unit=unit, acc_status=info["status"], time_repaired=info["time_repaired"],
                     hole_trimmed=info["hole_trimmed"])
            if x is not None:
                r.update(raw_n=info["n_raw"], raw_rate_hz=info["raw_rate_hz"], span_s=info["span_s"], n25=len(x))
                snips.append(x)
                w, s = centre(x, WIN)
                if r["time_repaired"]:
                    r["acc_status"] = "time_repaired"
                elif w is None:
                    r["acc_status"] = "short"
                else:
                    r["acc_status"] = "hole_trimmed" if r["hole_trimmed"] else "ok"
                if w is not None:
                    r.update(window_start=s, start_offset_s=s / FS)
                    if r["acc_status"] in OFFICIAL_ACC:
                        wins.append(w)
        rows.append(r)
    st = pd.DataFrame(rows)
    st["user"] = uuid
    st["has_snippet"] = st.n25 > 0
    st["short"] = st.has_snippet & (st.n25 < WIN)
    w = np.stack(wins) if wins else np.zeros((0, WIN, 3), np.float32)
    loc = es.read_location(uuid)
    loc["location_missing_reason"] = es.location_missing_reason(loc)
    return (st, w, snips, src.rename_axis("ts").reset_index().assign(user=uuid),
            loc.rename_axis("ts").reset_index().assign(user=uuid))


def build_es(workers: int) -> None:
    say("== ExtraSensory ==")
    m = es_minutes()
    acc_s = acc_schema([pa.field("unit", pa.string()), pa.field("raw_n", pa.int32()),
                        pa.field("raw_rate_hz", pa.float32()), pa.field("span_s", pa.float32()),
                        pa.field("start_offset_s", pa.float32()), pa.field("hole_trimmed", pa.bool_())]
                       + STAT_FIELDS)
    snip_s = acc_schema([pa.field("n25", pa.int32()), pa.field("unit", pa.string()), pa.field("raw_n", pa.int32()),
                         pa.field("raw_rate_hz", pa.float32()), pa.field("span_s", pa.float32()),
                         pa.field("acc_status", pa.string()), pa.field("time_repaired", pa.bool_()),
                         pa.field("hole_trimmed", pa.bool_()), pa.field("short", pa.bool_()),
                         pa.field("window_start", pa.int32())], fixed=False)
    users = sorted(m.user.unique())
    status, srcs, locs = [], [], []
    with pq.ParquetWriter(out_path("extrasensory", "accelerometer"), acc_s, compression=COMPRESSION) as wa, \
            pq.ParquetWriter(out_path("extrasensory", "snippets"), snip_s, compression=COMPRESSION) as wsn, \
            ProcessPoolExecutor(workers) as ex:
        for uuid, (st, w, snips, src, loc) in zip(users, ex.map(es_user, users)):
            ok = st[st.acc_status.isin(OFFICIAL_ACC)].copy()
            assert len(ok) == len(w)
            for k, v in stats_columns(w).items():
                ok[k] = v
            st = st.merge(ok[["ts", "acc_median", "acc_std", "acc_implausible", "quiet"]], on="ts", how="left")
            if len(w):
                wa.write_batch(acc_batch(acc_s, ok, w))
            sn = st[st.has_snippet]
            assert len(sn) == len(snips)
            if len(snips):
                wsn.write_batch(acc_batch(snip_s, sn, snips))
            status.append(st)
            srcs.append(src)
            locs.append(loc)
            log.info(f"{uuid}: {len(w)} windows, {len(snips)} snippets of {len(st)} minutes")
    st = pd.concat(status, ignore_index=True)
    loc = pd.concat(locs, ignore_index=True)
    cols = ["user", "ts", "acc_status", "has_snippet", "n25", "short", "time_repaired", "hole_trimmed",
            "acc_median", "acc_std", "acc_implausible", "quiet"]
    m = m.merge(st[cols], on=["user", "ts"], how="left", validate="one_to_one")
    m["has_acc"] = m.acc_status.isin(OFFICIAL_ACC)
    m["acc_implausible"] = m.acc_implausible.fillna(False).astype(bool)
    m["quiet"] = m.quiet.fillna(False).astype(bool)
    m["snippet_s"] = (m.n25 / FS).astype(np.float32)
    m = m.drop(columns="n25")
    lm = loc[["user", "ts", "latitude", "location_missing_reason"]]
    m = m.merge(lm, on=["user", "ts"], how="left", validate="one_to_one")
    m["has_location"] = m.latitude.notna()
    m = m.drop(columns="latitude")
    src = pd.concat(srcs, ignore_index=True)
    labels = m.merge(src, on=["user", "ts"], how="left", validate="one_to_one")
    labels["user_excluded"] = labels.user.isin(es.EXCLUDED_USERS)
    labels["user_exclusion_reason"] = labels.user.map(es.EXCLUDED_USERS).fillna("")
    official_flags(labels, "included_v2")
    lead = ["user", "ts", "platform", "split", "fold", "session", "cls", "y", "official", "included_v2", "has_acc",
            "acc_implausible", "label_implausible", "user_excluded", "user_exclusion_reason", "acc_status", "has_snippet", "snippet_s", "short", "time_repaired", "hole_trimmed",
            "quiet", "acc_median", "acc_std", "has_location", "location_missing_reason", "not_carried", "capped",
            "motorbike", "reason", "label_source", "position", "resting", "moving"]
    labels = labels[lead + [c for c in labels.columns if c not in lead]]
    write(labels, "extrasensory", "labels")
    write(sessions_table(labels, ["platform", "split", "fold"]), "extrasensory", "sessions")
    write(loc.drop(columns="location_missing_reason")[["user", "ts"] + [c for c in loc.columns if c not in
                                                                       ("user", "ts", "location_missing_reason")]],
          "extrasensory", "location_features")
    track = es_track(loc, labels)
    write(track, "extrasensory", "gps_per_minute")
    es_report(labels, track)


def es_track(loc: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """One fix per minute with coordinates. Time = minute timestamp (the fix time within the ~20 s recording is not
    published). Speed = location:max_speed, accuracy = best_horizontal_accuracy; -1 when missing. bearing = the
    bearing from the previous fix (bearing_from_fixes), -1 where speed is -1. Unchanged from v1."""
    t = loc[loc.latitude.notna() & loc.longitude.notna()].merge(labels[["user", "ts", "session"]], on=["user", "ts"])
    t = t.sort_values(["user", "ts"]).reset_index(drop=True)
    spd = t["location:max_speed"]
    acc = t["location:best_horizontal_accuracy"]
    tr = pd.DataFrame({"user": t.user, "ts": t.ts, "session": t.session, "time": t.ts.astype(float),
                       "lat": t.latitude, "lon": t.longitude,
                       "speed": np.where(spd.notna() & (spd >= 0), spd, -1.0),
                       "accuracy": np.where(acc.notna() & (acc >= 0), acc, -1.0)})
    tr = gps.add_consecutive(tr, ["user", "session"])
    # Production rule: a fix without speed has no bearing. The bearing from consecutive fixes is
    # kept, never blanked, next to the other between-fix columns.
    tr["bearing_from_fixes"] = gps.bearing_from_previous(tr, ["user", "session"])
    tr.insert(tr.columns.get_loc("accuracy") + 1, "bearing",
              np.where(tr.speed < 0, -1.0, tr.bearing_from_fixes))
    return tr


def sessions_table(labels: pd.DataFrame, keep: list[str]) -> pd.DataFrame:
    return labels.groupby(["user", "session"]).agg(
        **{k: (k, "first") for k in keep}, start_ts=("ts", "min"), end_ts=("ts", "max"), minutes=("ts", "size"),
        minutes_acc=("has_acc", "sum"), minutes_location=("has_location", "sum")).reset_index()


# ---------------------------------------------------------------- SHL

def shl_recordings(source: str) -> list[tuple[shl.Recording, str, int]]:
    """(recording, split, fold). User 1: release/User1 without the 3 days duplicated in User1_4phones, plus
    User1_4phones (02 audit), day folds from splits/shl_days.csv. Users 2-3: rail check, evaluation only."""
    days = pd.read_csv(SHL_SPLIT, dtype={"day": str})
    if source == "shl_user1":
        fold = days[(days.user == "User1") & (days.partition == "dev")].set_index("day").fold.to_dict()
        four = shl.recordings("User1_4phones")
        recs = [r for r in shl.recordings("User1") if r.day not in {f.day for f in four}] + four
        return [(r, "dev", fold[r.day]) for r in recs]
    test = reserved_test(RESERVED_V1)["shl"]
    recs = shl.recordings("User2_4phones") + shl.recordings("User3_4phones")
    assert {(r.user, r.day) for r in recs} == set(zip(test.user, test.day))
    return [(r, "rail_check", -1) for r in recs]


def shl_rec(job):
    """One recording and position: per clock minute one 15 s window centred in the minute (inside a NaN-free
    stretch, resampled 100 -> 25 Hz over the stretch as in 02), its labels; for the hips also the continuous 25 Hz
    stream (every stretch long enough for a window) and the 1 Hz location track."""
    rec, split, fold, position = job
    t, x = shl.read_motion(rec, position, ("acc",))
    _, y = shl.read_labels(rec)
    good = ~np.isnan(x).any(1)
    d = np.diff(np.concatenate([[0], good.astype(np.int8), [0]]))
    stretches = list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))
    minutes = np.arange(t[0] // 60_000, t[-1] // 60_000 + 1)
    centre_ms = minutes * 60_000 + 30_000
    rows, wins = [], []
    dec = [(lo, hi, decimate_100_to_25(x[lo:hi]), t[lo:hi:DOWN]) for lo, hi in stretches if hi - lo >= WIN * DOWN]
    for mi, c in zip(minutes, centre_ms):
        r = {"ts": int(mi * 60), "acc_status": "edge", "label_status": "edge", "cls": "", "majority_cls": "",
             "majority_share": np.nan, "t_start_ms": -1}
        j = np.searchsorted(t, c - 7_500)
        if j + WIN * DOWN <= len(t) and t[j] - (c - 7_500) < 10:
            lab = y[j:j + WIN * DOWN]
            vals, cnt = np.unique(lab, return_counts=True)
            k = cnt.argmax()
            r.update(majority_cls=CLS_NAME.get(int(vals[k]), "null"), majority_share=cnt[k] / len(lab))
            if len(vals) > 1:
                r["label_status"] = "mixed"
            elif vals[0] == 0:
                r["label_status"] = "null"
            else:
                r.update(label_status="ok", cls=CLS_NAME[int(vals[0])])
            r["acc_status"] = "nan_gap"
            for lo, hi, x25, t25 in dec:
                if t[lo] <= c - 7_500 and c + 7_500 <= t[hi - 1]:
                    s = np.searchsorted(t25, c - 7_500)
                    if s + WIN <= len(x25):
                        r.update(acc_status="ok", t_start_ms=int(t25[s]))
                        wins.append(x25[s:s + WIN])
                    break
        rows.append(r)
    st = pd.DataFrame(rows).assign(user=rec.user, rec=rec.rec, day=rec.day, split=split, fold=fold,
                                   position=position.lower())
    w = np.stack(wins) if wins else np.zeros((0, WIN, 3), np.float32)
    stream = loc = None
    if position == "Hips":
        stream = pd.DataFrame({
            "t_ms": np.concatenate([t25 for _, _, _, t25 in dec]) if dec else np.zeros(0, np.int64),
            "stretch": np.concatenate([np.full(len(t25), i, np.int32) for i, (_, _, _, t25) in enumerate(dec)])
            if dec else np.zeros(0, np.int32),
            "label": np.concatenate([y[lo:hi:DOWN] for lo, hi, _, _ in dec]).astype(np.int8) if dec else
            np.zeros(0, np.int8)})
        xs = np.concatenate([x25 for _, _, x25, _ in dec]) if dec else np.zeros((0, 3), np.float32)
        stream["x"], stream["y"], stream["z"] = xs[:, 0], xs[:, 1], xs[:, 2]
        stream = stream.assign(user=rec.user, rec=rec.rec)[["user", "rec", "stretch", "t_ms", "x", "y", "z", "label"]]
        loc = shl.read_location(rec, "Hips").assign(user=rec.user, rec=rec.rec)
    return st, w, stream, loc


def shl_full_track(loc: pd.DataFrame) -> pd.DataFrame:
    """The 1 Hz track of every recording with speed / bearing over ~3 s and speed over ~10 s."""
    parts = []
    for (user, rec), g in loc.groupby(["user", "rec"], sort=False):
        tt = g.time_ms.to_numpy() / 1e3
        lat, lon = g.lat.to_numpy(), g.lon.to_numpy()
        v3, b3 = gps.local_speed_bearing(tt, lat, lon, SPEED_HALF_S)
        v10, _ = gps.local_speed_bearing(tt, lat, lon, CADENCE_HALF_S)
        parts.append(pd.DataFrame({"user": user, "rec": rec, "time": tt, "lat": lat, "lon": lon,
                                   "accuracy": g.accuracy.to_numpy(), "altitude": g.altitude.to_numpy(),
                                   "speed": v3, "bearing": b3, "speed_10s": v10}))
    full = pd.concat(parts, ignore_index=True)
    full["ts"] = (full.time // 60 * 60).astype(np.int64)
    return full


def waiis_fixes(full: pd.DataFrame) -> np.ndarray:
    """Row indices of `full` kept at the WAIIS cadence (thinned per recording)."""
    idx = []
    for _, g in full.groupby(["user", "rec"], sort=False):
        k = gps.thin(g.time.to_numpy(), gps.cadence_interval(g.speed_10s.to_numpy()))
        idx.append(g.index.to_numpy()[k])
    return np.concatenate(idx)


def calibrate_blanking(workers: int) -> float:
    """v0 of the blanking curve, calibrated on SHL User 1's WAIIS-cadence fixes (computed the same way whichever
    sources are built, so every table uses the same curve)."""
    recs = shl_recordings("shl_user1")
    loc = pd.concat([shl.read_location(r, "Hips").assign(user=r.user, rec=r.rec) for r, _, _ in recs],
                    ignore_index=True)
    full = shl_full_track(loc)
    v = full.speed.to_numpy()[waiis_fixes(full)]
    v0 = gps.calibrate_v0(v)
    say(f"speed blanking: p = {gps.BLANK_P0} * exp(-v / {v0:.3f} m/s), calibrated on {len(v):,} SHL User 1 "
        f"WAIIS-cadence fixes (target {gps.SPEED_MISSING_SHARE:.0%} missing)")
    return v0


def shl_tracks(full: pd.DataFrame, labels: pd.DataFrame, v0: float):
    """Two GPS variants from the 1 Hz track; blanking driven by the true speed at the fix (never the label).
    Returns (gps_waiis_cadence, gps_per_minute, gps_unblanked)."""
    waiis = full.loc[waiis_fixes(full)]
    lw = labels[labels.acc_status != "edge"]
    pick_ts, pick_i = [], []
    for (user, rec), g in full.groupby(["user", "rec"], sort=False):
        tt = g.time.to_numpy()
        acc = g.accuracy.to_numpy()
        mins = lw[(lw.user == user) & (lw.rec == rec)].ts.to_numpy()
        lo = np.searchsorted(tt, mins + 30.0 - 7.5, "left")
        hi = np.searchsorted(tt, mins + 30.0 + 7.5, "right")
        for ts, a, b in zip(mins, lo, hi):
            if b > a:
                pick_ts.append(ts)
                pick_i.append(g.index[a + np.flatnonzero(acc[a:b] == acc[a:b].min())[-1]])
    minute = full.loc[pick_i].assign(ts=np.array(pick_ts, np.int64))
    out, unblanked = [], []
    for name, tr in (("gps_waiis_cadence", waiis), ("gps_per_minute", minute)):
        tr = tr.merge(labels[["user", "ts", "session"]], on=["user", "ts"], how="left", validate="many_to_one")
        out_of_table = tr.session.isna()
        if out_of_table.any():
            say(f"  {name}: {out_of_table.sum()} fixes outside the recording's minutes dropped")
        tr = tr[~out_of_table].astype({"session": np.int32})
        tr = tr.sort_values(["user", "time"]).reset_index(drop=True)
        unblanked.append(tr[["user", "ts", "time", "speed", "bearing"]].assign(variant=name))
        rng = np.random.default_rng(SEED)
        p = gps.blank_probability(tr.speed.to_numpy(), v0)
        blank = (rng.random(len(tr)) < p) & (tr.speed.to_numpy() >= 0)
        tr["speed_blanked"] = blank
        tr.loc[blank, ["speed", "bearing"]] = -1.0      # a fix without speed has no bearing
        tr = gps.add_consecutive(tr, ["user", "session"])
        lead = ["user", "ts", "session", "rec", "time", "lat", "lon", "speed", "accuracy", "bearing"]
        out.append(tr[lead + [c for c in tr.columns if c not in lead + ["altitude"]]])
    ub = pd.concat(unblanked, ignore_index=True)[["user", "ts", "time", "variant", "speed", "bearing"]]
    return out[0], out[1], ub


def build_shl(source: str, workers: int, v0: float) -> None:
    say(f"== {source} ==")
    recs = shl_recordings(source)
    jobs = [(r, s, f, "Hips") for r, s, f in recs]
    if source in OTHER_POSITION_SOURCES:
        jobs += [(r, s, f, p) for r, s, f in recs if r.session.endswith("_4phones") for p in OTHER_POSITIONS
                 if p in r.positions]
    hips_s = acc_schema([pa.field("t_start_ms", pa.int64())] + STAT_FIELDS)
    other_s = acc_schema([pa.field("position", pa.string()), pa.field("t_start_ms", pa.int64()),
                          pa.field("label_status", pa.string()), pa.field("cls", pa.string()),
                          pa.field("majority_cls", pa.string()), pa.field("majority_share", pa.float64())]
                         + STAT_FIELDS)
    stream_s = pa.schema([("user", pa.string()), ("rec", pa.string()), ("stretch", pa.int32()), ("t_ms", pa.int64()),
                          ("x", pa.float32()), ("y", pa.float32()), ("z", pa.float32()), ("label", pa.int8())])
    status, locs, n_other = [], [], 0
    wo = (pq.ParquetWriter(out_path(source, "accelerometer_other_positions"), other_s, compression=COMPRESSION)
          if source in OTHER_POSITION_SOURCES else None)
    with pq.ParquetWriter(out_path(source, "accelerometer"), hips_s, compression=COMPRESSION) as wh, \
            pq.ParquetWriter(out_path(source, "stream"), stream_s, compression=COMPRESSION) as wst, \
            ProcessPoolExecutor(workers) as ex:
        for job, (st, w, stream, loc) in zip(jobs, ex.map(shl_rec, jobs)):
            ok = st[st.acc_status == "ok"].copy()
            assert len(ok) == len(w)
            for k, v in stats_columns(w).items():
                ok[k] = v
            if job[3] == "Hips":
                st = st.merge(ok[["ts", "acc_median", "acc_std", "acc_implausible", "quiet"]], on="ts", how="left")
                if len(w):
                    wh.write_batch(acc_batch(hips_s, ok, w))
                wst.write_table(pa.Table.from_pandas(stream, schema=stream_s, preserve_index=False))
                status.append(st)
                locs.append(loc)
            elif len(w):
                wo.write_batch(acc_batch(other_s, ok, w))
                n_other += len(w)
            log.info(f"{job[0].session}/{job[0].rec}/{job[3]}: {len(w)} windows of {len(st)} minutes")
    if wo is not None:
        wo.close()
        say(f"other positions (hand / torso / bag, 4phones days): {n_other:,} windows")
    lab = pd.concat(status, ignore_index=True)
    assert not lab.duplicated(["user", "ts"]).any()
    lab = lab.sort_values(["user", "ts"]).reset_index(drop=True)
    lab["session"] = lab.groupby("user").ts.transform(lambda t: gps.session_ids(t.to_numpy())).astype(np.int32)
    lab["platform"] = "android"
    lab["included"] = lab.label_status == "ok"
    lab["has_acc"] = lab.acc_status == "ok"
    lab["acc_implausible"] = lab.acc_implausible.fillna(False).astype(bool)
    lab["quiet"] = lab.quiet.fillna(False).astype(bool)
    lab["y"] = np.where(lab.included, lab.cls.map(CLASS_ID).fillna(0), 0).astype(np.int8)
    full = shl_full_track(pd.concat(locs, ignore_index=True))
    waiis, minute, unblanked = shl_tracks(full, lab, v0)
    lab["has_location"] = lab.set_index(["user", "ts"]).index.isin(minute.set_index(["user", "ts"]).index)
    lab["user_excluded"], lab["user_exclusion_reason"] = False, ""
    official_flags(lab, "included")
    lead = ["user", "ts", "platform", "position", "split", "fold", "session", "rec", "day", "cls", "y", "official",
            "included", "has_acc", "acc_implausible", "label_implausible", "user_excluded", "acc_status", "quiet", "acc_median", "acc_std", "has_location",
            "label_status", "majority_cls", "majority_share", "t_start_ms"]
    lab = lab[lead]
    write(lab, source, "labels")
    write(sessions_table(lab, ["platform", "split", "fold", "rec"]), source, "sessions")
    write(waiis, source, "gps_waiis_cadence")
    write(minute, source, "gps_per_minute")
    write(full[["user", "rec", "ts", "time", "lat", "lon", "accuracy", "altitude", "speed", "bearing", "speed_10s"]],
          source, "gps_1hz")
    write(unblanked, source, "gps_unblanked")
    shl_report(source, lab, full, waiis, minute)


# ---------------------------------------------------------------- reports (results/tables/*.csv)

def table(df: pd.DataFrame, name: str, title: str) -> None:
    df.to_csv(RES / f"{name}.csv")
    log.info(f"\n{title} ({name}.csv):\n{df.to_string()}")


def es_report(lab: pd.DataFrame, track: pd.DataFrame) -> None:
    ex = lab[lab.official]
    inc = lab[lab.included_v2]
    say(f"minutes {len(lab):,}; included_v2 {len(inc):,}; windows {lab.has_acc.sum():,}; snippets "
        f"{lab.has_snippet.sum():,}; examples (official) {len(ex):,}")
    cls_order = [c for c in CLASSES if c in set(lab.cls)]
    t = pd.DataFrame({
        "included_v2": inc.groupby("cls").size(),
        "window": inc[inc.has_acc].groupby("cls").size(),
        "of_which_hole_trimmed": inc[inc.acc_status == "hole_trimmed"].groupby("cls").size(),
        "acc_implausible": inc[inc.has_acc & inc.acc_implausible].groupby("cls").size(),
        "label_implausible": inc[inc.has_acc & inc.label_implausible].groupby("cls").size(),
        "user_excluded": inc[inc.has_acc & inc.user_excluded].groupby("cls").size(),
        "examples": ex.groupby("cls").size(),
        "examples_iphone": ex[ex.platform == "iphone"].groupby("cls").size(),
        "examples_android": ex[ex.platform == "android"].groupby("cls").size(),
        "examples_dev": ex[ex.split == "dev"].groupby("cls").size(),
        "examples_test": ex[ex.split == "test"].groupby("cls").size(),
        "users_with_examples": ex.groupby("cls").user.nunique(),
        "quiet_share": ex.groupby("cls").quiet.mean().round(3)}).reindex(cls_order).fillna(0)
    table(t, "es_class_counts", "ExtraSensory counts per class")
    say(t.to_string())
    table(pd.crosstab(lab.acc_status, lab.platform, margins=True), "es_acc_status_all", "acc status, all minutes")
    table(pd.crosstab(inc.cls, inc.acc_status, margins=True), "es_acc_status_by_class", "acc status by class")
    table(pd.crosstab([lab.acc_status], [lab.short, lab.time_repaired], margins=True), "es_snippet_flags",
          "snippet status x short x time_repaired, all minutes")
    w = inc[inc.has_acc]
    imp = pd.crosstab(w[w.acc_implausible].user, w[w.acc_implausible].cls, margins=True)
    imp["windows"] = imp.index.map(w.groupby("user").size()).fillna(len(w))
    imp["platform"] = imp.index.map(lab.groupby("user").platform.first())
    imp["split"] = imp.index.map(lab.groupby("user").split.first())
    table(imp.sort_values("All", ascending=False), "es_implausible_by_user",
          "acc_implausible windows (included_v2) per user x class")
    allw = lab[lab.has_acc]
    table(pd.crosstab(allw[allw.acc_implausible].user, allw[allw.acc_implausible].platform, margins=True)
          .sort_values("All", ascending=False), "es_implausible_by_user_all_minutes",
          "acc_implausible windows, all minutes")
    ht = inc[inc.acc_status == "hole_trimmed"]
    table(pd.crosstab([ht.user, ht.platform, ht.split], ht.cls, margins=True), "es_hole_trimmed_by_user",
          "hole-trimmed official windows (included_v2) per user x class")
    wi = inc[inc.has_acc & ~inc.user_excluded]
    direction = np.where(wi.acc_median < ACC_PLAUSIBLE[0], "below", np.where(wi.acc_median > ACC_PLAUSIBLE[1],
                                                                               "above", "plausible"))
    table(pd.crosstab([wi.cls], [direction, wi.platform]), "es_acc_bounds_by_class",
          f"included windows (excluded users left out) below {ACC_PLAUSIBLE[0]} / above {ACC_PLAUSIBLE[1]} m/s2")
    li = inc[inc.has_acc & inc.cls.isin(LABEL_QUIET_CLASSES)]
    table(pd.crosstab(li.cls, [li.platform, li.label_implausible], margins=True), "es_label_implausible",
          "walk / run / bike windows (included_v2): label_implausible per class x platform")
    ue = inc[inc.user_excluded]
    table(pd.crosstab(ue.user, ue.cls, margins=True), "es_user_excluded", "included_v2 minutes of excluded users")
    q = ex.assign(position=ex.position.replace("", "(none)"))
    table(q.pivot_table(index="position", columns=["platform"], values="quiet", aggfunc="mean").round(3),
          "es_quiet_by_position", "quiet share of examples per position x platform")
    table(ex.pivot_table(index="cls", columns="platform", values="quiet", aggfunc="mean").reindex(cls_order).round(3),
          "es_quiet_by_class", "quiet share of examples per class x platform")
    f = ex.groupby(["split", "fold", "cls"]).size().unstack(fill_value=0)[cls_order]
    f.insert(0, "users", ex.groupby(["split", "fold"]).user.nunique())
    table(f, "es_fold_class_counts", "examples per fold x class")
    # test reporting: per platform, with / without location
    table(pd.crosstab([ex.split, ex.platform, ex.has_location], ex.cls, margins=True), "es_split_platform_location",
          "examples per split x platform x has_location")
    g = track.merge(ex[["user", "ts", "cls", "platform"]], on=["user", "ts"])
    table(g.pivot_table(index="cls", columns="platform", values="speed", aggfunc=lambda s: (s < 0).mean())
          .reindex(cls_order).round(3), "es_speed_missing_by_class", "ES gps_per_minute: speed -1 share on examples")


def shl_report(source: str, lab: pd.DataFrame, full: pd.DataFrame, waiis: pd.DataFrame, minute: pd.DataFrame) -> None:
    ex = lab[lab.official]
    w = lab[lab.included & lab.has_acc]
    table(pd.DataFrame({"windows": w.groupby("cls").size(),
                        "acc_implausible": w[w.acc_implausible].groupby("cls").size(),
                        "label_implausible": w[w.label_implausible].groupby("cls").size(),
                        "official": ex.groupby("cls").size()}).fillna(0).astype(int), f"{source}_official_by_class",
          "windows, flags and official examples per class")
    say(f"minutes {len(lab):,}; windows {lab.has_acc.sum():,}; acc_implausible {lab.acc_implausible.sum():,}; "
        f"examples (official) {len(ex):,}; 1 Hz fixes {len(full):,}; gps_waiis_cadence {len(waiis):,}; "
        f"gps_per_minute {len(minute):,}")
    cls_order = [c for c in CLASSES if c in set(ex.cls)]
    t = pd.crosstab(ex.user, ex.cls, margins=True)[cls_order + ["All"]]
    table(t, f"{source}_user_class_examples", "examples per user x class")
    say(t.to_string())
    w = lab[lab.included & lab.has_acc]
    table(pd.crosstab(w.user, [w.cls, w.acc_implausible]), f"{source}_implausible", "acc_implausible x class")
    table(pd.crosstab(w.cls, np.where(w.acc_median < ACC_PLAUSIBLE[0], "below", np.where(
        w.acc_median > ACC_PLAUSIBLE[1], "above", "plausible"))), f"{source}_acc_bounds_by_class",
        f"windows below {ACC_PLAUSIBLE[0]} / above {ACC_PLAUSIBLE[1]} m/s2")
    table(pd.crosstab(lab.label_status, lab.acc_status, margins=True), f"{source}_status",
          "label status x acc status, all minutes")
    table(ex.groupby("cls").quiet.mean().reindex(cls_order).round(3).to_frame(), f"{source}_quiet_by_class",
          "quiet share per class")
    rows = []
    cls_of = lab.set_index(["user", "ts"]).cls
    for name, tr in (("gps_waiis_cadence", waiis), ("gps_per_minute", minute)):
        c = tr.join(cls_of, on=["user", "ts"]).cls.replace("", "(no label)")
        for k, g in tr.groupby(c):
            rows.append({"variant": name, "cls": k, "fixes": len(g), "dt_prev_p50_s": g.dt_prev_s.median(),
                         "speed_missing": (g.speed < 0).mean(), "speed_blanked": g.speed_blanked.mean(),
                         "speed_p50": g.speed[g.speed >= 0].median(), "accuracy_p50": g.accuracy.median(),
                         "bearing_missing": (g.bearing < 0).mean()})
        rows.append({"variant": name, "cls": "ALL", "fixes": len(tr), "speed_missing": (tr.speed < 0).mean(),
                     "speed_blanked": tr.speed_blanked.mean(), "bearing_missing": (tr.bearing < 0).mean()})
    table(pd.DataFrame(rows).set_index(["variant", "cls"]).round(3), f"{source}_gps_by_class", "GPS by class")
    cov = ex.groupby("cls").has_location.mean().reindex(cls_order).round(3)
    table(cov.to_frame(), f"{source}_location_by_class", "examples with a per-minute fix")


# ---------------------------------------------------------------- manifest

def git() -> tuple[str, bool]:
    c = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                                cwd=PROJECT).stdout.strip())
    return c, dirty


def write_manifest(commit: str, dirty: bool, v0: float | None) -> None:
    files = {}
    for p in sorted(ROOT.rglob("*.parquet")):
        rel = str(p.relative_to(ROOT))
        files[rel] = {"sha256": bench.sha256(p), "bytes": p.stat().st_size, "rows": pq.ParquetFile(p).metadata.num_rows}
    man = {"version": VERSION, "built": date.today().isoformat(), "git_commit": commit, "git_dirty": dirty,
           "builder": "experiments/04_frozen-benchmark_es-shl/build.py",
           "params": {"rate_hz": FS, "window_samples": WIN, "window_s": WIN / FS, "seed": SEED,
                      "label_rules": "v2", "still_cap": es.RESTING_STILL_MAX_SHARE,
                      "acc_plausible_median_ms2": ACC_PLAUSIBLE, "quiet_std_ms2": QUIET_STD,
                      "label_quiet_classes": LABEL_QUIET_CLASSES, "excluded_users": es.EXCLUDED_USERS,
                      "hole_max_s": 0.5, "hole_policy": "longest clean stretch",
                      "session_gap_s": gps.SESSION_GAP_S, "gps_cadence": gps.CADENCE,
                      "speed_missing_share": gps.SPEED_MISSING_SHARE, "blank_p0": gps.BLANK_P0, "blank_v0": v0,
                      "speed_half_window_s": SPEED_HALF_S, "cadence_half_window_s": CADENCE_HALF_S,
                      "es_splits": "experiments/04_frozen-benchmark_es-shl/splits/es_users_v2.csv",
                      "shl_splits": "experiments/04_frozen-benchmark_es-shl/splits/shl_days.csv",
                      "reserved_test": "splits/reserved_test_v2.json"},
           "versions": {"numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pa.__version__},
           "files": files}
    (ROOT / "manifest.json").write_text(json.dumps(man, indent=1))
    pd.DataFrame(files).T.to_csv(RES / "manifest_files.csv")
    say(f"manifest: {len(files)} files, {sum(f['bytes'] for f in files.values()) / 1e9:.2f} GB, commit {commit[:7]}"
        f"{' (dirty)' if dirty else ''}")


def main() -> None:
    global RES
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sources", nargs="+", default=list(bench.SOURCES), choices=bench.SOURCES)
    ap.add_argument("--tables-dir", type=Path, default=RES, help="count tables (default: 04's results/tables)")
    ap.add_argument("--log-dir", type=Path, default=HERE / "logs", help="build.log (default: 04's logs)")
    a = ap.parse_args()
    RES = a.tables_dir
    commit, dirty = git()          # before the build rewrites its own results/ and logs/
    RES.mkdir(parents=True, exist_ok=True)
    a.log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=a.log_dir / "build.log", filemode="w", level=logging.INFO,
                        format="%(message)s")
    v0 = calibrate_blanking(a.workers) if any(s != "extrasensory" for s in a.sources) else None
    for s in a.sources:
        build_es(a.workers) if s == "extrasensory" else build_shl(s, a.workers, v0)
    write_manifest(commit, dirty, v0)


if __name__ == "__main__":
    main()


# ================================================================ waiis_bench_v2 (experiment 09)
# v2 runs v1's code above unchanged for ExtraSensory and SHL (into the v2 root), then adds v2's columns and rules
# (`extend_v1_sources_v2`); NOR-TMD and Collecty enter through `tmd.candidates` (the one window definition) and the
# same writers, flags and session tables (`build_new`). Rules and numbers: experiment 09's README and
# decision_register.md.

import pyarrow.compute as pc  # noqa: E402  (v2 only)

V2 = "waiis_bench_v2"
E09 = PROJECT / "experiments" / "09_frozen-benchmark_es-shl-nor-collecty"
NEW_FOLDS = E09 / "splits" / "folds_nor_tmd_collecty.csv"     # approved 2026-11-02
RESERVED_V3 = 3                                              # splits/reserved_test_v3.json
NEW_SOURCES = ("nor_tmd", "collecty")
IOS_TO_ANDROID = -1.0          # iOS accelerometers to the Android axis convention, after g -> m/s2 (08's reader rule)
CAP_TOP_SHARE, CAP_FLOOR = 0.25, 60   # rule R2 (approved): per (dataset, class), dev users only
NOR_ACC = ("accelerometer", "CMAccelerometerData")           # Android TYPE_ACCELEROMETER (m/s2), iOS (g)
NOR_FINE = {"bicycle": "bike", "bus": "bus", "car": "car", "metro": "subway", "train": "train", "tram": "tram"}
NOR_EXCLUDED = {"inside", "outside"}                         # conflict with v1's classes: unlabelled usable minutes
COL_FINE = {"Run": "run", "Bike": "bike", "Car": "car", "Bus": "bus", "Train": "train", "Tramway": "tram"}
COL_EXCLUDED = {"Walk"}                                      # includes standing


def configure_v2(tables_dir: Path) -> None:
    """Point the shared writers at the v2 root and v2's tables; SHL Users 2-3 get other-position windows too."""
    global ROOT, RES, OTHER_POSITION_SOURCES
    ROOT, RES = bench.root(V2), tables_dir
    OTHER_POSITION_SOURCES = ("shl_user1", "shl_user23")
    ROOT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)


def _negate_lists(tb: pa.Table, rows: np.ndarray) -> pa.Table:
    """x, y, z list columns negated on the rows where `rows` is true (the iOS -> Android axis convention)."""
    cols = {}
    for c in ("x", "y", "z"):
        col = tb.column(c).combine_chunks()
        flat = col.flatten().to_numpy(zero_copy_only=False).astype(np.float32)
        lens = np.asarray(pc.list_value_length(col).to_numpy(zero_copy_only=False), np.int64)
        sign = np.repeat(np.where(rows, IOS_TO_ANDROID, 1.0).astype(np.float32), lens)
        vals = pa.array(flat * sign, pa.float32())
        cols[c] = (pa.FixedSizeListArray.from_arrays(vals, col.type.list_size)
                   if isinstance(col.type, pa.FixedSizeListType)
                   else pa.ListArray.from_arrays(pa.array(np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)),
                                                 vals))
    for c, arr in cols.items():
        tb = tb.set_column(tb.schema.get_field_index(c), c, arr)
    return tb


def _rewrite(source: str, table: str, fn) -> None:
    """Read a v2 table written by v1's code, apply fn(pa.Table) -> pa.Table batch by batch, write it back."""
    p = out_path(source, table)
    tmp = p.with_suffix(".tmp.parquet")
    pf = pq.ParquetFile(p)
    writer = None
    for b in pf.iter_batches(1 << 15):
        t = fn(pa.Table.from_batches([b]))
        if writer is None:
            writer = pq.ParquetWriter(tmp, t.schema, compression=COMPRESSION)
        writer.write_table(t)
    writer.close()
    tmp.replace(p)


def _add(t: pa.Table, **cols) -> pa.Table:
    for k, v in cols.items():
        t = t.append_column(k, pa.array(v))
    return t


def _v2_label_defaults(lab: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """The columns v2 adds to every labels table, with the defaults for v1's sources (README, schema diff)."""
    lab["dataset"] = dataset
    lab["source_platform"] = np.where(lab.platform.eq("iphone"), "ios", lab.platform)
    lab["sign_converted"] = lab.source_platform.eq("ios")
    lab["cls_fine"] = lab.cls
    lab["y_fine"] = lab.y.astype(np.int8)
    lab["label_raw"] = lab.cls if "majority_cls" not in lab else np.where(lab.cls.ne(""), lab.cls, lab.majority_cls)
    for c, v in (("label_excluded", False), ("user_capped", False), ("duplicate_dropped", False), ("dup_group", -1),
                 ("clock_piece", 0), ("time_overlap", False), ("clock_wraps", 0), ("position_default", True)):
        lab[c] = v
    if "reason" not in lab:
        lab["reason"] = ""
    return lab


def _position_draw(lab: pd.DataFrame, other: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """One random position per minute among the positions with a window (hips + other positions), seed SEED:
    (draw for the labels rows, draw for the other-position rows)."""
    hips = lab.loc[lab.has_acc, ["user", "ts"]].assign(position="hips")
    cand = pd.concat([hips, other[["user", "ts", "position"]].assign(position=other.position.str.lower())])
    cand = cand.sort_values(["user", "ts", "position"]).reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    pick = cand.groupby(["user", "ts"]).position.transform(lambda s: s.iloc[rng.integers(len(s))])
    chosen = cand[cand.position == pick].set_index(["user", "ts"]).position
    key = pd.MultiIndex.from_frame(lab[["user", "ts"]])
    d_lab = chosen.reindex(key).eq("hips").to_numpy() & lab.has_acc.to_numpy()
    ko = pd.MultiIndex.from_frame(other[["user", "ts"]])
    d_oth = chosen.reindex(ko).to_numpy() == other.position.str.lower().to_numpy()
    return d_lab, d_oth


def extend_v1_sources_v2() -> None:
    """ExtraSensory and SHL in v2: v1's tables (just written by v1's code into the v2 root) plus v2's columns,
    ExtraSensory iPhone windows and snippets in the Android axis convention, the >= 24 Hz rate rule, and SHL's
    one-position-per-minute draw."""
    from tmd import candidates as q
    say("== v2 columns for ExtraSensory and SHL ==")
    # ExtraSensory
    lab = pd.read_parquet(out_path("extrasensory", "labels"))
    acc = pd.read_parquet(out_path("extrasensory", "accelerometer"), columns=["user", "ts", "raw_rate_hz"])
    lab = lab.merge(acc, on=["user", "ts"], how="left", validate="one_to_one")
    lab = _v2_label_defaults(lab, "extrasensory")
    lab["rate_below_24hz"] = lab.has_acc & (lab.raw_rate_hz < q.MIN_RATE_HZ)
    lab["position_draw"] = lab.has_acc
    was = lab.official.copy()
    lab["official"] = lab.official & ~lab.rate_below_24hz
    say(f"extrasensory: {int((was & ~lab.official).sum())} official windows below {q.MIN_RATE_HZ:g} Hz; "
        f"{int(lab.sign_converted.sum())} iPhone minutes converted to the Android axis convention")
    write(lab.drop(columns="raw_rate_hz"), "extrasensory", "labels")
    plat = lab.set_index(["user", "ts"]).source_platform
    pos = lab.set_index(["user", "ts"]).position
    for table in ("accelerometer", "snippets"):
        def fn(t, table=table):
            key = pd.MultiIndex.from_arrays([t.column("user").to_pandas(), t.column("ts").to_pandas()])
            ios = plat.reindex(key).eq("ios").to_numpy()
            t = _negate_lists(t, ios)
            extra = dict(dataset=["extrasensory"] * len(t), source_platform=np.where(ios, "ios", "android"),
                         sign_converted=ios)
            if table == "accelerometer":
                extra["position"] = pos.reindex(key).fillna("").to_numpy()
            return _add(t, **extra)
        _rewrite("extrasensory", table, fn)
    # SHL: hips labels / windows, other positions, draw
    for src in ("shl_user1", "shl_user23"):
        lab = _v2_label_defaults(pd.read_parquet(out_path(src, "labels")), "shl")
        lab["rate_below_24hz"] = False                       # 100 Hz, NaN-free stretches
        other = pd.read_parquet(out_path(src, "accelerometer_other_positions"),
                                columns=["user", "ts", "position"])
        d_lab, d_oth = _position_draw(lab, other)
        lab["position_draw"] = d_lab
        write(lab, src, "labels")
        draw = lab.set_index(["user", "ts"]).position_draw
        _rewrite(src, "accelerometer", lambda t: _add(
            t, dataset=["shl"] * len(t), source_platform=["android"] * len(t), sign_converted=[False] * len(t),
            position=["hips"] * len(t), position_draw=draw.reindex(pd.MultiIndex.from_arrays(
                [t.column("user").to_pandas(), t.column("ts").to_pandas()])).fillna(False).to_numpy()))
        state = {"i": 0}

        def fo(t, d=d_oth, state=state):
            n = len(t)
            sel = d[state["i"]:state["i"] + n]
            state["i"] += n
            return _add(t, dataset=["shl"] * n, source_platform=["android"] * n, sign_converted=[False] * n,
                        position_default=[False] * n, position_draw=sel)
        _rewrite(src, "accelerometer_other_positions", fo)
        say(f"{src}: other-position windows {len(other):,}; position draw keeps hips on "
            f"{int(d_lab.sum()):,} of {int(lab.has_acc.sum()):,} hips minutes")


# ---------------------------------------------------------------- NOR-TMD and Collecty (v2)

def _trip(user, trip, t_ms, xyz, raw_label, meta, stream_ok=True):
    """One trip (or clock piece): tmd.candidates' minute windows and the 25 Hz stream of its gap-free stretches with
    a real rate >= the minimum. raw_label per accelerometer event."""
    from tmd import candidates as q
    t_ms = np.asarray(t_ms, np.int64)
    o = np.argsort(t_ms, kind="stable")
    t, xyz, lab = t_ms[o] / 1e3, np.asarray(xyz, float)[o], np.asarray(raw_label).astype(str)[o]
    rows, w = q.trip_windows(t, xyz, t, lab, grid="minute")
    if not len(rows):
        return None, w, None
    rows = rows.drop(columns=[c for c in rows.columns if c.startswith(("gyro_", "has_"))]).assign(
        user=user, trip=trip, **meta)
    st = None
    if stream_ok:
        tt, xx, _, _ = q.distinct(t, xyz)
        parts = []
        for k, (t25, x25) in enumerate(q.stretches25(tt, xx)):
            i, j = np.searchsorted(tt, [t25[0], t25[-1] + 1e-9])
            if (j - i) / max(t25[-1] - t25[0], 1e-9) < q.MIN_RATE_HZ:
                continue                                    # e.g. NOR-TMD iOS at 5 Hz: no upsampled stream
            idx = np.clip(np.searchsorted(t, t25, side="right") - 1, 0, len(lab) - 1)
            parts.append(pd.DataFrame({"t_ms": np.round(t25 * 1e3).astype(np.int64), "stretch": np.int32(k),
                                       "x": x25[:, 0], "y": x25[:, 1], "z": x25[:, 2], "raw_label": lab[idx]}))
        if parts:
            st = pd.concat(parts, ignore_index=True).assign(user=user, trip=trip)
    return rows, w, st


def nor_tmd_user(user: str):
    """All trips of one NOR-TMD installation. Android in m/s2 as recorded; iOS from g to m/s2 and to the Android
    axis convention (x -9.80665)."""
    from tmd.readers import nor_tmd
    from tmd.resample import G
    d = nor_tmd.stream(user, list(NOR_ACC))
    d["sensor"], d["label"], d["position"] = d.sensor.astype(str), d.label.astype(str).str.lower(), \
        d.position.astype(str).str.lower()
    tr = nor_tmd.trips()
    tr = tr[tr.user == user].groupby(["journey", "trip"]).agg(os=("os", "first"), manufacturer=("manufacturer", "first"),
                                                              model=("model", "first"))
    out = []
    for (j, k), e in d.groupby(["journey", "trip"], observed=True):
        info = tr.loc[(j, k)] if (j, k) in tr.index else None
        ios = info is not None and str(info.os).lower() == "ios"
        xyz = e[["x", "y", "z"]].to_numpy(float) * (IOS_TO_ANDROID * G if ios else 1.0)
        meta = {"platform": "ios" if ios else "android", "model": "" if info is None else f"{info.manufacturer} {info.model}",
                "position": e.position.mode().iloc[0] if len(e) else "", "clock_piece": 0, "clock_wraps": 0,
                "sign_converted": ios, "duplicate_dropped": False, "dup_group": -1}
        out.append(_trip(user, f"{j}_{k}", e.t_ms, xyz, e.label, meta))
    return out


def collecty_user(user: int):
    """All trajectories of one Collecty user: total acceleration = acc + grav (stored by the converter), 12 h clock
    wraps undone and pieces at other backward steps (tmd.readers.collecty.repair_time); windows overlapping another piece's
    span get status time_overlap; files dropped as duplicates are flagged, not windowed into the stream."""
    from tmd.readers import collecty
    f = collecty.files()
    dup = collecty.duplicates()
    drop = dup[dup["drop"]].set_index(["user", "traj"]).group.to_dict()
    groups = dup.set_index(["user", "traj"]).group.to_dict()
    out = []
    for traj in sorted(f.traj[f.user == user]):
        a = collecty.stream(user, traj)
        ta, piece, wraps, back = collecty.repair_time(a.t_ms.to_numpy())
        xyz, lab = a[["ax", "ay", "az"]].to_numpy(), a.label.to_numpy()
        dropped = (user, traj) in drop
        spans = [(ta[piece == p].min(), ta[piece == p].max()) for p in np.unique(piece)]
        for p, (lo, hi) in enumerate(spans):
            k = piece == p
            if k.sum() < 2:
                continue
            meta = {"platform": "android", "model": "", "position": "", "clock_piece": p, "clock_wraps": wraps,
                    "sign_converted": False, "duplicate_dropped": dropped,
                    "dup_group": int(groups.get((user, traj), -1))}
            rows, w, st = _trip(str(user), str(traj), ta[k], xyz[k], lab[k], meta, stream_ok=not dropped)
            if rows is not None:
                others = [sp for i, sp in enumerate(spans) if i != p]
                s0 = rows.t_start.to_numpy() * 1e3
                ov = np.zeros(len(rows), bool)
                for o0, o1 in others:
                    ov |= (s0 < o1) & (s0 + 15_000 > o0)
                ok_before = rows.status.eq("ok").to_numpy()
                if (ov & ok_before).any():
                    w = w[~ov[ok_before]]
                rows.loc[ov, "status"] = "time_overlap"
                if st is not None and others:
                    tms = st.t_ms.to_numpy()
                    keep = np.ones(len(st), bool)
                    for o0, o1 in others:
                        keep &= ~((tms >= o0) & (tms <= o1))
                    st = st[keep]
            out.append((rows, w, st))
    return out


def _cap_keep(df: pd.DataFrame, cap: float, rng) -> np.ndarray:
    """Indices of df (one user, one class) to keep under the cap: a round-robin over the user's trips in a seeded
    trip order, windows in time order within a trip."""
    if len(df) <= cap:
        return df.index.to_numpy()
    trips = list(df.trip.unique())
    rng.shuffle(trips)
    queues = [list(df[df.trip == t].sort_values("ts").index) for t in trips]
    keep = []
    while len(keep) < cap:
        for qu in queues:
            if qu and len(keep) < cap:
                keep.append(qu.pop(0))
    return np.array(keep)


def cap_r2(counts: pd.Series) -> float:
    """Rule R2: the largest per-user cap with top-user share <= CAP_TOP_SHARE, floored at CAP_FLOOR; inf if the top
    user already holds <= CAP_TOP_SHARE."""
    if counts.max() / counts.sum() <= CAP_TOP_SHARE:
        return np.inf
    for c in range(int(counts.max()), CAP_FLOOR - 1, -1):
        x = np.minimum(counts, c)
        if x.max() / x.sum() <= CAP_TOP_SHARE:
            return float(c)
    return float(CAP_FLOOR)


def build_new(source: str, workers: int) -> None:
    """NOR-TMD or Collecty: labels (one row per user-minute), sessions, accelerometer, stream."""
    from tmd import candidates as q
    from tmd.readers import collecty
    from tmd.readers import nor_tmd
    from tmd.labels import FINE_ID
    say(f"== {source} ==")
    users = nor_tmd.users() if source == "nor_tmd" else sorted(collecty.files().user.unique().tolist())
    work = nor_tmd_user if source == "nor_tmd" else collecty_user
    fine, excluded = (NOR_FINE, NOR_EXCLUDED) if source == "nor_tmd" else (COL_FINE, COL_EXCLUDED)
    stream_s = pa.schema([("user", pa.string()), ("trip", pa.string()), ("stretch", pa.int32()), ("t_ms", pa.int64()),
                          ("x", pa.float32()), ("y", pa.float32()), ("z", pa.float32()), ("label", pa.int8())])
    rows = []
    with pq.ParquetWriter(out_path(source, "stream"), stream_s, compression=COMPRESSION) as wst, \
            ProcessPoolExecutor(workers) as ex:
        for user, res in zip(users, ex.map(work, users)):
            n_ok = 0
            for r, w, st in res:
                if r is None:
                    continue
                assert r.status.eq("ok").sum() == len(w)
                rows.append((r, w))
                n_ok += len(w)
                if st is not None and len(st):
                    lab = st.raw_label.map(fine).map(FINE_ID).fillna(0).astype(np.int8)
                    st = st.assign(user=str(user), label=lab)[["user", "trip", "stretch", "t_ms", "x", "y", "z",
                                                               "label"]]
                    wst.write_table(pa.Table.from_pandas(st, schema=stream_s, preserve_index=False))
            log.info(f"{source} {user}: {n_ok} windows")
    # one table of rows, windows aligned to the ok rows
    parts, warr = [], []
    for r, w in rows:
        ok = r.status.eq("ok").to_numpy()
        r = r.copy()
        r["win"] = -1
        r.loc[ok, "win"] = np.arange(len(w)) + sum(len(x) for x in warr)
        warr.append(w)
        parts.append(r)
    m = pd.concat(parts, ignore_index=True)
    W = np.concatenate(warr) if warr else np.zeros((0, WIN, 3), np.float32)
    m["user"] = m.user.astype(str)
    m["ts"] = m.ts.astype(np.int64)
    # one row per user-minute: an ok window wins over the other statuses, then the earlier trip
    m["_ok"] = m.status.eq("ok")
    m = m.sort_values(["user", "ts", "_ok", "t_start"], ascending=[True, True, False, True])
    n_dup = int(m.duplicated(["user", "ts"]).sum())
    m = m.drop_duplicates(["user", "ts"]).drop(columns="_ok").sort_values(["user", "ts"]).reset_index(drop=True)
    say(f"{source}: {len(m):,} user-minutes ({n_dup} duplicate minutes across trips / pieces dropped)")
    lab = pd.DataFrame({"user": m.user, "ts": m.ts, "dataset": source, "platform": m.platform,
                        "source_platform": m.platform, "position": m.position, "trip": m.trip,
                        "label_raw": m.label.astype(str), "acc_status": m.status, "t_start_ms":
                            np.round(m.get("t_start_25hz", m.t_start).fillna(m.t_start) * 1e3).astype(np.int64),
                        "raw_rate_hz": m.rate_hz.astype(np.float32), "max_gap_s": m.max_gap_s.astype(np.float32),
                        "acc_median": m.acc_median.astype(np.float32), "acc_std": m.acc_std.astype(np.float32),
                        "model": m.model, "sign_converted": m.sign_converted.astype(bool),
                        "duplicate_dropped": m.duplicate_dropped.astype(bool), "dup_group": m.dup_group.astype(np.int32),
                        "clock_piece": m.clock_piece.astype(np.int32), "clock_wraps": m.clock_wraps.astype(np.int32)})
    lab["cls_fine"] = lab.label_raw.map(fine).fillna("")
    lab["cls"] = lab.cls_fine.where(lab.cls_fine.isin(CLASSES), "")
    lab["label_excluded"] = lab.label_raw.isin(excluded)
    lab["reason"] = np.select([lab.cls_fine.ne(""), lab.label_excluded, lab.label_raw.str.contains("|", regex=False)],
                              ["", "excluded:" + lab.label_raw, "mixed_label"], "no_class:" + lab.label_raw)
    lab["time_overlap"] = lab.acc_status.eq("time_overlap")
    lab["has_acc"] = lab.acc_status.eq("ok")
    lab["rate_below_24hz"] = lab.raw_rate_hz < q.MIN_RATE_HZ
    lab["acc_implausible"] = lab.has_acc & ~lab.acc_median.between(*ACC_PLAUSIBLE)
    lab["quiet"] = lab.has_acc & (lab.acc_std < QUIET_STD)
    lab["user_excluded"], lab["user_exclusion_reason"] = False, ""
    lab["has_location"], lab["location_missing_reason"] = False, "no_gps_in_dataset"
    lab["position_default"], lab["position_draw"] = True, lab.has_acc
    # splits: reserved NOR-TMD installations are test; dev folds from the approved file (others -1, never scored)
    test = set(reserved_test(RESERVED_V3)["nor_tmd"].user) if source == "nor_tmd" else set()
    folds = pd.read_csv(NEW_FOLDS, dtype={"user": str})
    folds = folds[folds.dataset == source].set_index("user").fold
    lab["split"] = np.where(lab.user.isin(test), "test", "dev")
    lab["fold"] = np.where(lab.split.eq("test"), -1, lab.user.map(folds).fillna(-1)).astype(np.int8)
    lab["session"] = lab.groupby("user").trip.transform(lambda s: pd.factorize(s)[0]).astype(np.int32)
    # caps (R2) on dev users, over the windows that would otherwise be official
    lab["included_pre"] = lab.cls_fine.ne("") & ~lab.label_excluded & ~lab.duplicate_dropped
    lab["label_implausible"] = lab.has_acc & lab.quiet & lab.cls.isin(LABEL_QUIET_CLASSES)
    pre = (lab.included_pre & lab.has_acc & ~lab.acc_implausible & ~lab.label_implausible & lab.split.eq("dev"))
    lab["user_capped"] = False
    rng = np.random.default_rng(SEED)
    caps = []
    for c, g in lab[pre].groupby("cls_fine"):
        per = g.groupby("user").size()
        cap = cap_r2(per)
        kept = 0
        for u, gu in g.groupby("user"):
            keep = _cap_keep(gu, cap, rng) if np.isfinite(cap) else gu.index.to_numpy()
            lab.loc[gu.index.difference(keep), "user_capped"] = True
            kept += len(keep)
        caps.append({"dataset": source, "class": c, "dev_users": len(per), "windows": int(per.sum()),
                     "cap": "" if np.isinf(cap) else int(cap), "kept": kept,
                     "top_share_before": round(per.max() / per.sum(), 3),
                     "top_share_after": round(np.minimum(per, cap).max() / np.minimum(per, cap).sum(), 3),
                     "users_capped": int((per > cap).sum())})
    table(pd.DataFrame(caps).set_index(["dataset", "class"]), f"{source}_caps", "R2 caps")
    lab["included"] = lab.included_pre & ~lab.user_capped
    lab["y"] = np.where(lab.included, lab.cls.map(CLASS_ID).fillna(0), 0).astype(np.int8)
    lab["y_fine"] = np.where(lab.included, lab.cls_fine.map(FINE_ID).fillna(0), 0).astype(np.int8)
    official_flags(lab, "included")
    lead = ["user", "ts", "dataset", "platform", "source_platform", "position", "split", "fold", "session", "trip",
            "cls", "y", "cls_fine", "y_fine", "official", "included", "has_acc", "acc_implausible",
            "label_implausible", "user_excluded", "user_exclusion_reason", "acc_status", "quiet", "acc_median",
            "acc_std", "raw_rate_hz", "max_gap_s", "rate_below_24hz", "label_raw", "label_excluded", "user_capped",
            "duplicate_dropped", "dup_group", "clock_piece", "time_overlap", "clock_wraps", "sign_converted",
            "position_default", "position_draw", "has_location", "location_missing_reason", "reason", "model",
            "t_start_ms"]
    lab = lab[lead]
    write(lab, source, "labels")
    write(sessions_table(lab, ["platform", "split", "fold", "trip"]), source, "sessions")
    # windows of the has_acc rows, in label order
    idx = m.win.to_numpy()
    sel = lab.has_acc.to_numpy()
    w = W[idx[sel]] if sel.any() else np.zeros((0, WIN, 3), np.float32)
    meta = lab.loc[sel, ["user", "ts", "dataset", "source_platform", "sign_converted", "position", "raw_rate_hz",
                         "max_gap_s", "t_start_ms", "acc_median", "acc_std", "acc_implausible", "quiet"]]
    sch = acc_schema([pa.field("dataset", pa.string()), pa.field("source_platform", pa.string()),
                      pa.field("sign_converted", pa.bool_()), pa.field("position", pa.string()),
                      pa.field("raw_rate_hz", pa.float32()), pa.field("max_gap_s", pa.float32()),
                      pa.field("t_start_ms", pa.int64())] + STAT_FIELDS)
    with pq.ParquetWriter(out_path(source, "accelerometer"), sch, compression=COMPRESSION) as wa:
        for s0 in range(0, len(meta), 8192):
            wa.write_batch(acc_batch(sch, meta.iloc[s0:s0 + 8192], w[s0:s0 + 8192]))
    new_report(source, lab)


def new_report(source: str, lab: pd.DataFrame) -> None:
    """Count tables for a new source (results/tables of 09)."""
    from tmd.labels import FINE_CLASSES
    ex = lab[lab.official]
    order = [c for c in FINE_CLASSES if c in set(ex.cls_fine)]
    table(pd.crosstab([ex.split, ex.source_platform], ex.cls_fine, margins=True).reindex(columns=order + ["All"]),
          f"{source}_class_counts", "official examples")
    table(pd.crosstab(lab.acc_status, lab.source_platform, margins=True), f"{source}_acc_status", "acc status")
    ok = lab[lab.has_acc & lab.cls_fine.ne("")]
    fl = ok.groupby("cls_fine")[["acc_implausible", "quiet", "label_implausible"]].mean().mul(100).round(2)
    table(fl.reindex([c for c in FINE_CLASSES if c in fl.index]), f"{source}_flag_rates", "flag rates (% of windows)")
    u = ex.groupby(["cls_fine", "user"]).size()
    table(pd.DataFrame({"users": u.groupby(level=0).size(), "users_ge5": (u >= 5).groupby(level=0).sum(),
                        "top_user_share": (u.groupby(level=0).max() / u.groupby(level=0).sum()).round(3)}),
          f"{source}_users_per_class", "users per class (official)")


def write_manifest_v2(commit: str, dirty: bool, v0: float | None) -> None:
    from tmd import candidates as q
    files = {}
    for p in sorted(ROOT.rglob("*.parquet")):
        rel = str(p.relative_to(ROOT))
        files[rel] = {"sha256": bench.sha256(p), "bytes": p.stat().st_size, "rows": pq.ParquetFile(p).metadata.num_rows}
    v1 = bench.reference_manifest("waiis_bench_v1")["files"]
    same_as_v1 = sorted(f for f in files if f in v1 and files[f]["sha256"] == v1[f]["sha256"])
    man = {"version": V2, "built": date.today().isoformat(), "git_commit": commit, "git_dirty": dirty,
           "builder": "experiments/09_frozen-benchmark_es-shl-nor-collecty/build.py (code: tmd.bench.build)",
           "params": {"rate_hz": FS, "window_samples": WIN, "window_s": WIN / FS, "seed": SEED, "label_rules": "v2",
                      "fine_classes": "still walk run bike car bus train subway tram",
                      "score6": "still foot(walk+run) bike car bus rail(train+subway+tram)",
                      "still_cap": es.RESTING_STILL_MAX_SHARE, "acc_plausible_median_ms2": ACC_PLAUSIBLE,
                      "quiet_std_ms2": QUIET_STD, "label_quiet_classes": LABEL_QUIET_CLASSES,
                      "min_rate_hz": q.MIN_RATE_HZ, "max_gap_s": q.MAX_GAP_S, "excluded_users": es.EXCLUDED_USERS,
                      "ios_to_android": "x (-1) after g -> m/s2", "cap_rule": {"top_share": CAP_TOP_SHARE,
                                                                               "floor": CAP_FLOOR, "on": "dev users"},
                      "excluded_labels": {"nor_tmd": sorted(NOR_EXCLUDED), "collecty": sorted(COL_EXCLUDED)},
                      "run_user_flag": {"rule": "user median window std |acc| over labelled run windows < reference",
                                        "reference_ms2": RUN_FLAG_REF[0],
                                        "reference": "ExtraSensory dev official walk windows, median acc_std"},
                      "session_gap_s": gps.SESSION_GAP_S, "gps_cadence": gps.CADENCE, "blank_v0": v0,
                      "es_splits": "experiments/04_frozen-benchmark_es-shl/splits/es_users_v2.csv",
                      "shl_splits": "experiments/04_frozen-benchmark_es-shl/splits/shl_days.csv",
                      "new_splits": "experiments/09_frozen-benchmark_es-shl-nor-collecty/splits/folds_nor_tmd_collecty.csv",
                      "reserved_test": "splits/reserved_test_v3.json"},
           "versions": {"numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pa.__version__},
           "files_identical_to_v1": same_as_v1, "files": files}
    (ROOT / "manifest.json").write_text(json.dumps(man, indent=1))
    pd.DataFrame(files).T.to_csv(RES / "manifest_files.csv")
    say(f"manifest: {len(files)} files, {sum(f['bytes'] for f in files.values()) / 1e9:.2f} GB, commit {commit[:7]}"
        f"{' (dirty)' if dirty else ''}; identical to v1: {len(same_as_v1)}")


def run_user_flag() -> None:
    """Run labels whose user moves like walking (approved 2026-11-02): a user whose median window std |acc| over
    their labelled run windows is below ExtraSensory dev walk's median (official windows) gets every run minute
    flagged `run_user_flagged`. Flagged minutes are never labels (out of `included` / `official`, y = 0) and stay
    as unlabelled usable minutes, like the other conflicting labels. Every source and split, test included."""
    say("== run_user_flagged ==")
    es_lab = pd.read_parquet(out_path("extrasensory", "labels"), columns=["split", "cls_fine", "official", "acc_std"])
    ref = float(es_lab[es_lab.official & (es_lab.split == "dev") & (es_lab.cls_fine == "walk")].acc_std.median())
    rows = []
    for src in V2_SOURCES_ALL:
        lab = pd.read_parquet(out_path(src, "labels"))
        labelled = lab.included_v2 if "included_v2" in lab else (lab.included | lab.user_capped)
        run = lab.cls_fine.eq("run") & lab.has_acc & labelled & ~lab.label_excluded & ~lab.duplicate_dropped
        med = lab[run].groupby("user").acc_std.median()
        flagged = set(med.index[med < ref])
        f = lab.cls_fine.eq("run") & lab.user.isin(flagged)
        before = lab[lab.official & lab.cls_fine.eq("run")].groupby("user").size()
        lab["run_user_flagged"] = f
        for c in ("included", "included_v2"):
            if c in lab:
                lab[c] &= ~f
        lab["official"] &= ~f
        lab["y"] = np.where(f, 0, lab.y).astype(np.int8)
        lab["y_fine"] = np.where(f, 0, lab.y_fine).astype(np.int8)
        write(lab, src, "labels")
        for u in sorted(set(med.index)):
            g = lab[(lab.user == u) & lab.cls_fine.eq("run")]
            rows.append({"source": src, "user": u, "split": g.split.iloc[0], "platform": g.source_platform.iloc[0],
                         "run_windows": int(run[lab.user == u].sum()), "median_std_acc": round(float(med[u]), 3),
                         "flagged": u in flagged, "official_run_before": int(before.get(u, 0)),
                         "official_run_after": int((lab.official & lab.cls_fine.eq("run") & (lab.user == u)).sum())})
    t = pd.DataFrame(rows).set_index(["source", "user"])
    table(t, "run_user_flag", f"run users vs ExtraSensory dev walk median std |acc| = {ref:.3f} m/s2")
    say(f"run_user_flagged: reference {ref:.3f} m/s2; {int(t.flagged.sum())} of {len(t)} run users flagged; "
        f"official run {int(t.official_run_before.sum())} -> {int(t.official_run_after.sum())}")
    RUN_FLAG_REF[0] = ref


V2_SOURCES_ALL = ("extrasensory", "shl_user1", "shl_user23", "nor_tmd", "collecty")
RUN_FLAG_REF = [np.nan]
