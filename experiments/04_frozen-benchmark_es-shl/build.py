"""Build the frozen benchmark es_bench_v1 (README, step 2). One command rebuilds everything:

    uv run python build.py [--workers 8] [--sources extrasensory shl_user1 shl_user23]

Writes <data>/processed/es_bench_v1/<source>/<table>.parquet and manifest.json (parameters, git commit, sha256 of
every file; read them with `tmd.bench`). Count tables for the README go to results/tables/, the full log to
logs/build.log; stdout gets a short summary. Existing output of a rebuilt source is overwritten.
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

from tmd import bench, gps, shl
from tmd import extrasensory as es
from tmd.labels import CLASSES, CLASS_ID
from tmd.paths import PROJECT
from tmd.resample import DOWN, centre, decimate_100_to_25, es_acc_to_25hz
from tmd.splits import reserved_test

HERE = Path(__file__).resolve().parent
RES = HERE / "results" / "tables"
SPLITS_02 = PROJECT / "experiments" / "02_family-baseline_shl-es" / "splits"
WIN, FS = bench.WIN, bench.FS
SEED = 0
SPEED_HALF_S = 1.5             # speed / bearing of a kept SHL fix: from the 1 Hz track over ~3 s around it
CADENCE_HALF_S = 5.0           # cadence choice: speed over ~10 s around the candidate fix
COMPRESSION = "zstd"
CLS_NAME = {v: k for k, v in CLASS_ID.items()}

log = logging.getLogger("build")


def say(msg: str = "") -> None:
    print(msg)
    log.info(msg)


def acc_schema(extra: list[pa.Field]) -> pa.Schema:
    win = pa.list_(pa.float32(), WIN)
    return pa.schema([("user", pa.string()), ("ts", pa.int64()), ("x", win), ("y", win), ("z", win)] + extra)


def acc_batch(schema: pa.Schema, meta: pd.DataFrame, w: np.ndarray) -> pa.RecordBatch:
    """meta: one row per window with the schema's non-array columns; w: (n, 375, 3) float32."""
    cols = {}
    for f in schema:
        if f.name in ("x", "y", "z"):
            flat = pa.array(np.ascontiguousarray(w[:, :, "xyz".index(f.name)]).ravel(), pa.float32())
            cols[f.name] = pa.FixedSizeListArray.from_arrays(flat, WIN)
        else:
            cols[f.name] = pa.array(meta[f.name].to_numpy(), f.type)
    return pa.RecordBatch.from_pydict(cols, schema=schema)


def write(df: pd.DataFrame, source: str, table: str) -> None:
    p = bench.path(source, table)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False, compression=COMPRESSION)


# ---------------------------------------------------------------- ExtraSensory

def es_minutes() -> pd.DataFrame:
    """Rules v2 for every minute of all 60 users, flags, split and fold (02's es_users.csv; test = reserved)."""
    m = es.minutes_table(es.uuids())
    ok = m.reason == "ok"
    keep = np.zeros(len(m), bool)
    keep[np.flatnonzero(ok)] = es.cap_resting_still(m[ok], seed=SEED)    # as 02 (eval_tables.py)
    m["included_v2"] = keep
    m["capped"] = ok & ~keep
    m["not_carried"] = m.reason == "drop:not_carried"
    m["motorbike"] = m.reason == "check:MOTORBIKE"
    m["y"] = np.where(keep, m.y, 0)
    users = pd.read_csv(SPLITS_02 / "es_users.csv")
    test = set(reserved_test()["extrasensory"].uuid)
    assert set(users.uuid[users.partition == "test"]) == test, "02's test rows differ from reserved_test.json"
    assert set(users.uuid) == set(m.user)
    u = users.set_index("uuid")
    m["platform"] = m.user.map(u.platform)
    m["split"] = m.user.map(u.partition)
    m["fold"] = m.user.map(u.fold).astype(np.int8)
    m = m.sort_values(["user", "ts"]).reset_index(drop=True)
    m["session"] = m.groupby("user").ts.transform(lambda t: gps.session_ids(t.to_numpy())).astype(np.int32)
    return m


def es_user(uuid: str) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """One user: per-minute acc status, windows of the 'ok' minutes, source label columns, location columns."""
    clean = es.read_cleaned_labels(uuid)
    orig = es.read_original_labels(uuid).reindex(clean.index)
    src = pd.concat([clean.drop(columns="label_source").add_prefix("label:"),
                     orig.drop(columns="label_source", errors="ignore").add_prefix("orig:")], axis=1)
    rows, wins = [], []
    for ts in clean.index.astype(int):
        a = es.read_snippet(es.snippet_path(uuid, ts, "acc"))
        r = {"ts": ts, "acc_status": "no_file", "raw_n": 0, "span_s": np.nan, "raw_rate_hz": np.nan, "unit": ""}
        if a is not None:
            span = float(a[-1, 0] - a[0, 0])
            r.update(raw_n=len(a), span_s=span, raw_rate_hz=(len(a) - 1) / span if span > 0 else np.nan)
            x, unit, status = es_acc_to_25hz(a)
            r.update(unit=unit, acc_status=status)
            if x is not None:
                w, s = centre(x, WIN)
                if w is None:
                    r["acc_status"] = "under_15s"
                else:
                    r["start_offset_s"] = s / FS
                    wins.append(w)
        rows.append(r)
    st = pd.DataFrame(rows)
    st["user"] = uuid
    w = np.stack(wins) if wins else np.zeros((0, WIN, 3), np.float32)
    loc = es.read_location(uuid)
    loc["location_missing_reason"] = es.location_missing_reason(loc)
    return st, w, src.rename_axis("ts").reset_index().assign(user=uuid), loc.rename_axis("ts").reset_index().assign(
        user=uuid)


def build_es(workers: int) -> None:
    say("== ExtraSensory ==")
    m = es_minutes()
    schema = acc_schema([pa.field("unit", pa.string()), pa.field("raw_n", pa.int32()),
                         pa.field("raw_rate_hz", pa.float32()), pa.field("span_s", pa.float32()),
                         pa.field("start_offset_s", pa.float32())])
    p = bench.path("extrasensory", "accelerometer")
    p.parent.mkdir(parents=True, exist_ok=True)
    users = sorted(m.user.unique())
    status, srcs, locs = [], [], []
    with pq.ParquetWriter(p, schema, compression=COMPRESSION) as wr, ProcessPoolExecutor(workers) as ex:
        for uuid, (st, w, src, loc) in zip(users, ex.map(es_user, users)):
            ok = st[st.acc_status == "ok"]
            assert len(ok) == len(w)
            if len(w):
                wr.write_batch(acc_batch(schema, ok, w))
            status.append(st)
            srcs.append(src)
            locs.append(loc)
            log.info(f"{uuid}: {len(w)} windows of {len(st)} minutes")
    st = pd.concat(status, ignore_index=True)
    loc = pd.concat(locs, ignore_index=True)
    m = m.merge(st[["user", "ts", "acc_status"]], on=["user", "ts"], how="left", validate="one_to_one")
    m["has_acc"] = m.acc_status == "ok"
    lm = loc[["user", "ts", "latitude", "location_missing_reason"]]
    m = m.merge(lm, on=["user", "ts"], how="left", validate="one_to_one")
    m["has_location"] = m.latitude.notna()
    m = m.drop(columns="latitude")
    src = pd.concat(srcs, ignore_index=True)
    labels = m.merge(src, on=["user", "ts"], how="left", validate="one_to_one")
    lead = ["user", "ts", "platform", "split", "fold", "session", "cls", "y", "included_v2", "has_acc",
            "acc_status", "has_location", "location_missing_reason", "not_carried", "capped", "motorbike", "reason",
            "label_source", "position", "resting", "moving"]
    labels = labels[lead + [c for c in labels.columns if c not in lead]]
    write(labels, "extrasensory", "labels")
    write(sessions_table(labels, ["platform", "split", "fold"]), "extrasensory", "sessions")
    write(loc.drop(columns="location_missing_reason")[["user", "ts"] + [c for c in loc.columns if c not in
                                                                       ("user", "ts", "location_missing_reason")]],
          "extrasensory", "location_features")
    write(es_track(loc, labels), "extrasensory", "gps_per_minute")
    es_report(labels, st, loc)


def es_track(loc: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """One fix per minute with coordinates. Time = minute timestamp (the fix time within the ~20 s recording is not
    published). Speed = location:max_speed, accuracy = best_horizontal_accuracy; -1 when missing."""
    t = loc[loc.latitude.notna() & loc.longitude.notna()].merge(labels[["user", "ts", "session"]], on=["user", "ts"])
    t = t.sort_values(["user", "ts"]).reset_index(drop=True)
    spd = t["location:max_speed"]
    acc = t["location:best_horizontal_accuracy"]
    tr = pd.DataFrame({"user": t.user, "ts": t.ts, "session": t.session, "time": t.ts.astype(float),
                       "lat": t.latitude, "lon": t.longitude,
                       "speed": np.where(spd.notna() & (spd >= 0), spd, -1.0),
                       "accuracy": np.where(acc.notna() & (acc >= 0), acc, -1.0)})
    tr["bearing"] = gps.bearing_from_previous(tr, ["user", "session"])
    return gps.add_consecutive(tr, ["user", "session"])


def sessions_table(labels: pd.DataFrame, keep: list[str]) -> pd.DataFrame:
    return labels.groupby(["user", "session"]).agg(
        **{k: (k, "first") for k in keep}, start_ts=("ts", "min"), end_ts=("ts", "max"), minutes=("ts", "size"),
        minutes_acc=("has_acc", "sum"), minutes_location=("has_location", "sum")).reset_index()


# ---------------------------------------------------------------- SHL

def shl_recordings(source: str) -> list[tuple[shl.Recording, str, int]]:
    """(recording, split, fold). User 1: release/User1 without the 3 days duplicated in User1_4phones, plus
    User1_4phones (02 audit), folds from 02's shl_days.csv. Users 2-3: rail check, evaluation only."""
    days = pd.read_csv(SPLITS_02 / "shl_days.csv", dtype={"day": str})
    if source == "shl_user1":
        fold = days[(days.user == "User1") & (days.partition == "dev")].set_index("day").fold.to_dict()
        four = shl.recordings("User1_4phones")
        recs = [r for r in shl.recordings("User1") if r.day not in {f.day for f in four}] + four
        return [(r, "dev", fold[r.day]) for r in recs]
    test = reserved_test()["shl"]
    recs = shl.recordings("User2_4phones") + shl.recordings("User3_4phones")
    assert {(r.user, r.day) for r in recs} == set(zip(test.user, test.day))
    return [(r, "rail_check", -1) for r in recs]


def shl_rec(job) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """One recording, hips: per clock minute one 15 s window centred in the minute (inside a NaN-free stretch,
    resampled 100 -> 25 Hz over the stretch as in 02), its labels, and the 1 Hz location track."""
    rec, split, fold = job
    t, x = shl.read_motion(rec, "Hips", ("acc",))
    _, y = shl.read_labels(rec)
    good = ~np.isnan(x).any(1)
    d = np.diff(np.concatenate([[0], good.astype(np.int8), [0]]))
    stretches = list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))
    minutes = np.arange(t[0] // 60_000, t[-1] // 60_000 + 1)
    centre_ms = minutes * 60_000 + 30_000
    rows, wins = [], []
    # 25 Hz version of every stretch long enough for a window
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
    st = pd.DataFrame(rows).assign(user=rec.user, rec=rec.rec, day=rec.day, split=split, fold=fold)
    w = np.stack(wins) if wins else np.zeros((0, WIN, 3), np.float32)
    loc = shl.read_location(rec, "Hips").assign(user=rec.user, rec=rec.rec)
    return st, w, loc


def shl_tracks(loc: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two GPS variants from the 1 Hz track of each recording.
    gps_waiis_cadence: thinned by the speed over ~10 s at the candidate fix (CADENCE: >= 3 m/s every 10 s,
      >= 0.5 m/s every 20 s, else 180 s); natural gaps stay gaps.
    gps_per_minute: one fix per window minute, as ExtraSensory: among the fixes inside the 15 s window, the best
      accuracy, latest on ties.
    In both, speed and bearing come from the 1 Hz track over ~3 s around the fix; speed and bearing are then set to
    -1 on 23% of the kept fixes (seed 0), as phones record a fix without speed."""
    waiis, minute = [], []
    for (user, rec), g in loc.groupby(["user", "rec"], sort=False):
        tt = g.time_ms.to_numpy() / 1e3
        lat, lon = g.lat.to_numpy(), g.lon.to_numpy()
        v3, b3 = gps.local_speed_bearing(tt, lat, lon, SPEED_HALF_S)
        v10, _ = gps.local_speed_bearing(tt, lat, lon, CADENCE_HALF_S)
        full = pd.DataFrame({"user": user, "rec": rec, "time": tt, "lat": lat, "lon": lon, "speed": v3,
                             "accuracy": g.accuracy.to_numpy(), "bearing": b3, "speed_10s": v10})
        k = gps.thin(tt, gps.cadence_interval(v10))
        waiis.append(full.iloc[k])
        lw = labels[(labels.user == user) & (labels.rec == rec) & (labels.acc_status != "edge")]
        c = lw.ts.to_numpy() + 30.0
        lo = np.searchsorted(tt, c - 7.5, "left")
        hi = np.searchsorted(tt, c + 7.5, "right")
        pick = []
        for ts, a, b in zip(lw.ts.to_numpy(), lo, hi):
            if b > a:
                acc = full.accuracy.to_numpy()[a:b]
                i = a + np.flatnonzero(acc == acc.min())[-1]
                pick.append((ts, i))
        if pick:
            ts_, i_ = map(np.array, zip(*pick))
            minute.append(full.iloc[i_].assign(ts=ts_))
    out = []
    for name, parts in (("waiis", waiis), ("minute", minute)):
        tr = pd.concat(parts, ignore_index=True)
        if name == "waiis":
            tr["ts"] = (tr.time // 60 * 60).astype(np.int64)
        tr = tr.merge(labels[["user", "ts", "session"]], on=["user", "ts"], how="left", validate="many_to_one")
        out_of_table = tr.session.isna()
        if out_of_table.any():
            say(f"  {name}: {out_of_table.sum()} fixes outside the recording's minutes dropped")
        tr = tr[~out_of_table].astype({"session": np.int32})
        tr = tr.sort_values(["user", "time"]).reset_index(drop=True)
        rng = np.random.default_rng(SEED)
        blank = rng.random(len(tr)) < gps.SPEED_MISSING_SHARE
        tr["speed_blanked"] = blank
        tr.loc[blank, ["speed", "bearing"]] = -1.0      # a fix without speed has no bearing (Marc, 2026-10-28)
        tr = gps.add_consecutive(tr, ["user", "session"])
        lead = ["user", "ts", "session", "rec", "time", "lat", "lon", "speed", "accuracy", "bearing"]
        out.append(tr[lead + [c for c in tr.columns if c not in lead]])
    return out[0], out[1]


def build_shl(source: str, workers: int) -> None:
    say(f"== {source} ==")
    jobs = shl_recordings(source)
    schema = acc_schema([pa.field("t_start_ms", pa.int64())])
    p = bench.path(source, "accelerometer")
    p.parent.mkdir(parents=True, exist_ok=True)
    status, locs = [], []
    with pq.ParquetWriter(p, schema, compression=COMPRESSION) as wr, ProcessPoolExecutor(workers) as ex:
        for job, (st, w, loc) in zip(jobs, ex.map(shl_rec, jobs)):
            ok = st[st.acc_status == "ok"]
            assert len(ok) == len(w)
            if len(w):
                wr.write_batch(acc_batch(schema, ok, w))
            status.append(st)
            locs.append(loc)
            log.info(f"{job[0].session}/{job[0].rec}: {len(w)} windows of {len(st)} minutes, {len(loc)} fixes")
    lab = pd.concat(status, ignore_index=True)
    assert not lab.duplicated(["user", "ts"]).any()
    lab = lab.sort_values(["user", "ts"]).reset_index(drop=True)
    lab["session"] = lab.groupby("user").ts.transform(lambda t: gps.session_ids(t.to_numpy())).astype(np.int32)
    lab["platform"], lab["position"] = "android", "hips"
    lab["included"] = lab.label_status == "ok"
    lab["has_acc"] = lab.acc_status == "ok"
    lab["y"] = np.where(lab.included, lab.cls.map(CLASS_ID).fillna(0), 0).astype(np.int8)
    loc = pd.concat(locs, ignore_index=True)
    waiis, minute = shl_tracks(loc, lab)
    lab["has_location"] = lab.set_index(["user", "ts"]).index.isin(minute.set_index(["user", "ts"]).index)
    lead = ["user", "ts", "platform", "position", "split", "fold", "session", "rec", "day", "cls", "y", "included",
            "has_acc", "acc_status", "has_location", "label_status", "majority_cls", "majority_share", "t_start_ms"]
    lab = lab[lead]
    write(lab, source, "labels")
    write(sessions_table(lab, ["platform", "split", "fold", "rec"]), source, "sessions")
    write(waiis, source, "gps_waiis_cadence")
    write(minute, source, "gps_per_minute")
    shl_report(source, lab, loc, waiis, minute)


# ---------------------------------------------------------------- reports (results/tables/*.csv)

ORDER = CLASSES + ["", "null"]


def table(df: pd.DataFrame, name: str, title: str) -> None:
    df.to_csv(RES / f"{name}.csv")
    log.info(f"\n{title} ({name}.csv):\n{df.to_string()}")


def es_report(lab: pd.DataFrame, st: pd.DataFrame, loc: pd.DataFrame) -> None:
    ex = lab[lab.included_v2 & lab.has_acc]
    say(f"minutes {len(lab):,}; included_v2 {lab.included_v2.sum():,}; examples (included_v2 + window) {len(ex):,}; "
        f"windows {lab.has_acc.sum():,}")
    moving = ex.cls != "still"
    cls_order = [c for c in CLASSES if c in set(lab.cls)]
    t = pd.DataFrame({
        "minutes_labelled": lab[lab.cls != ""].groupby("cls").size(),
        "included_v2": lab[lab.included_v2].groupby("cls").size(),
        "examples": ex.groupby("cls").size(),
        "examples_iphone": ex[ex.platform == "iphone"].groupby("cls").size(),
        "examples_android": ex[ex.platform == "android"].groupby("cls").size(),
        "examples_dev": ex[ex.split == "dev"].groupby("cls").size(),
        "examples_test": ex[ex.split == "test"].groupby("cls").size(),
        "users_with_examples": ex.groupby("cls").user.nunique()}).reindex(cls_order).fillna(0).astype(int)
    table(t, "es_class_counts", "ExtraSensory counts per class")
    say(t.to_string())
    table(pd.crosstab(lab.user, lab.cls.replace("", "(unlabelled)"), margins=True), "es_user_class_all_minutes",
          "all minutes per user x class")
    table(pd.crosstab(ex.user, ex.cls, margins=True).assign(platform=lambda d: d.index.map(
        lab.groupby("user").platform.first()), split=lambda d: d.index.map(lab.groupby("user").split.first())),
        "es_user_class_examples", "examples per user x class")
    table(pd.crosstab(ex[moving].user, ex[moving].cls, margins=True), "es_user_class_examples_moving",
          "examples per user x moving class")
    table(pd.crosstab([ex.platform, ex.position.replace("", "(none)")], ex.cls, margins=True),
          "es_platform_position_class", "examples per platform x position x class")
    # drops
    table(pd.crosstab(lab.acc_status, lab.platform, margins=True), "es_acc_status_all", "acc status, all minutes")
    inc = lab[lab.included_v2]
    table(pd.crosstab(inc.acc_status, inc.platform, margins=True), "es_acc_status_included",
          "acc status, included_v2")
    table(pd.crosstab(inc.cls, inc.acc_status, margins=True), "es_acc_status_by_class", "acc status by class")
    table(pd.crosstab(lab.reason, lab.platform, margins=True), "es_label_reason", "label rule outcome per minute")
    u = inc.groupby(["user", "platform", "split"]).agg(
        included=("ts", "size"), under_15s=("acc_status", lambda s: (s == "under_15s").sum()),
        other_drop=("acc_status", lambda s: (~s.isin(["ok", "under_15s"])).sum()))
    ut = pd.crosstab(inc[inc.acc_status == "under_15s"].user, inc[inc.acc_status == "under_15s"].cls)
    u = u.join(ut).fillna(0)
    u = u[u.under_15s > 0].sort_values("under_15s", ascending=False)
    rate = st.groupby("user").raw_rate_hz.median().round(1).rename("raw_rate_p50_hz")
    u = u.join(rate, on="user")
    table(u, "es_15s_loss_by_user", "15 s rule losses per user (included_v2)")
    # folds
    f = ex.groupby(["split", "fold", "cls"]).size().unstack(fill_value=0)[cls_order]
    share = (f / f.sum()).round(3)
    f.insert(0, "users", ex.groupby(["split", "fold"]).user.nunique())
    table(f, "es_fold_class_counts", "examples per fold x class")
    table(share, "es_fold_class_share", "share of each class's examples per fold")
    # location
    cov = ex.groupby("cls").agg(examples=("ts", "size"), has_location=("has_location", "mean")).reindex(cls_order)
    for p in ("iphone", "android"):
        cov[f"has_location_{p}"] = ex[ex.platform == p].groupby("cls").has_location.mean()
    mr = pd.crosstab(ex.cls, ex.location_missing_reason.replace("", "present"), normalize="index").reindex(
        cls_order)
    table(cov.join(mr).round(3), "es_location_by_class", "location coverage and missing reason by class")
    w = lab[lab.location_missing_reason == "withheld"]
    say(f"location_missing_reason over all minutes: "
        f"{lab.location_missing_reason.replace('', 'present').value_counts().to_dict()}; "
        f"withheld by class {w.cls.replace('', '(unlabelled)').value_counts().to_dict()}")
    say(f"examples with a 15 s window lost per platform: "
        f"{inc[inc.acc_status == 'under_15s'].platform.value_counts().to_dict()}")


def shl_report(source: str, lab: pd.DataFrame, loc: pd.DataFrame, waiis: pd.DataFrame, minute: pd.DataFrame) -> None:
    ex = lab[lab.included & lab.has_acc]
    say(f"minutes {len(lab):,}; windows {lab.has_acc.sum():,}; examples (one label + window) {len(ex):,}; "
        f"1 Hz fixes {len(loc):,}; gps_waiis_cadence {len(waiis):,}; gps_per_minute {len(minute):,}")
    cls_order = [c for c in CLASSES if c in set(ex.cls)]
    t = pd.crosstab(ex.user, ex.cls, margins=True)[cls_order + ["All"]]
    table(t, f"{source}_user_class_examples", "examples per user x class")
    say(t.to_string())
    table(pd.crosstab(lab.label_status, lab.acc_status, margins=True), f"{source}_status",
          "label status x acc status, all minutes")
    # GPS: cadence per class (interval to previous kept fix inside a session) and speed / accuracy
    rows = []
    cls_of = lab.set_index(["user", "ts"]).cls
    for name, tr in (("gps_waiis_cadence", waiis), ("gps_per_minute", minute)):
        c = tr.join(cls_of, on=["user", "ts"]).cls.replace("", "(no label)")
        for k, g in tr.groupby(c):
            rows.append({"variant": name, "cls": k, "fixes": len(g), "dt_prev_p50_s": g.dt_prev_s.median(),
                         "dt_prev_p90_s": g.dt_prev_s.quantile(.9), "speed_missing": (g.speed < 0).mean(),
                         "speed_p50": g.speed[g.speed >= 0].median(), "accuracy_p50": g.accuracy.median(),
                         "bearing_missing": (g.bearing < 0).mean()})
    table(pd.DataFrame(rows).set_index(["variant", "cls"]).round(3), f"{source}_gps_by_class", "GPS by class")
    cov = ex.groupby("cls").has_location.mean().reindex(cls_order).round(3)
    table(cov.to_frame(), f"{source}_location_by_class", "examples with a per-minute fix")


# ---------------------------------------------------------------- manifest

def git() -> tuple[str, bool]:
    c = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                                cwd=PROJECT).stdout.strip())
    return c, dirty


def write_manifest(commit: str, dirty: bool) -> None:
    files = {}
    for p in sorted(bench.ROOT.rglob("*.parquet")):
        rel = str(p.relative_to(bench.ROOT))
        files[rel] = {"sha256": bench.sha256(p), "bytes": p.stat().st_size, "rows": pq.ParquetFile(p).metadata.num_rows}
    man = {"version": bench.VERSION, "built": date.today().isoformat(), "git_commit": commit, "git_dirty": dirty,
           "builder": "experiments/04_frozen-benchmark_es-shl/build.py",
           "params": {"rate_hz": FS, "window_samples": WIN, "window_s": WIN / FS, "seed": SEED,
                      "label_rules": "v2", "still_cap": es.RESTING_STILL_MAX_SHARE,
                      "session_gap_s": gps.SESSION_GAP_S, "gps_cadence": gps.CADENCE,
                      "speed_missing_share": gps.SPEED_MISSING_SHARE, "speed_half_window_s": SPEED_HALF_S,
                      "cadence_half_window_s": CADENCE_HALF_S,
                      "es_splits": "experiments/02_family-baseline_shl-es/splits/es_users.csv",
                      "shl_splits": "experiments/02_family-baseline_shl-es/splits/shl_days.csv",
                      "reserved_test": "splits/reserved_test.json"},
           "versions": {"numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pa.__version__},
           "files": files}
    (bench.ROOT / "manifest.json").write_text(json.dumps(man, indent=1))
    pd.DataFrame(files).T.to_csv(RES / "manifest_files.csv")
    say(f"manifest: {len(files)} files, {sum(f['bytes'] for f in files.values()) / 1e9:.2f} GB, commit {commit[:7]}"
        f"{' (dirty)' if dirty else ''}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sources", nargs="+", default=list(bench.SOURCES), choices=bench.SOURCES)
    a = ap.parse_args()
    commit, dirty = git()          # before the build rewrites its own results/ and logs/
    RES.mkdir(parents=True, exist_ok=True)
    (HERE / "logs").mkdir(exist_ok=True)
    logging.basicConfig(filename=HERE / "logs" / "build.log", filemode="w", level=logging.INFO, format="%(message)s")
    for s in a.sources:
        build_es(a.workers) if s == "extrasensory" else build_shl(s, a.workers)
    write_manifest(commit, dirty)


if __name__ == "__main__":
    main()
