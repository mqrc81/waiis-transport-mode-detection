"""NOR-TMD reader (Skretting, Gronli, Mukkamala, J. Intell. Transp. Syst. 2025; experiment 08 README, stop-point 1).

Raw: <data>/raw/nor_tmd/nor-tmd.zip (Kaggle scholarone/nor-tmd v1), kept zipped: one long CSV `nor_tmd.csv`, one row
per sensor event (installationId, journeyNumber, tripNumber, manufacturer, model, timestamp, transportType,
typeInteger, typeString, deviceLocation, value_0..5, activity_recognition, OS). Rows are not sorted by time.
Timestamps have no time zone; they are used as given.

Converted (`convert`, one streaming pass over the zip):
    <data>/processed/nor_tmd_events_acc-grav-gyro_mag-press-times/
        events/user=<installationId>/part-<batch>.parquet
            journey, trip, sensor, t_ms, x, y, z, label, position
            x, y, z = value_0..2 for every kept sensor except magnetometer / pressure (time only, NaN values)
        trips.parquet     one row per (user, journey, trip): manufacturer, model, os, labels, positions, events,
                          first / last t_ms
        counts.parquet    events per (os, sensor), kept or dropped
Dropped sensors (counted only): rotation vectors, quaternion / attitude, activity recognition.
"""
from __future__ import annotations

import io
import re
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from tmd.paths import PROCESSED, RAW

ZIP = RAW / "nor_tmd" / "nor-tmd.zip"
MEMBER = "nor_tmd.csv"
ZIP_MEMBER_BYTES = 73_611_929_573
CHUNK_BYTES = 256 << 20
OUT = PROCESSED / "nor_tmd_events_acc-grav-gyro_mag-press-times"
DROP = re.compile(r"rotation_vector|quaternion|attitude|activity", re.I)
TIME_ONLY = re.compile(r"magnet|pressure|baro", re.I)

# v1 class per NOR-TMD mode (approved 2026-11-02): inside / outside conflict with v1 (still and walk mixed) ->
# excluded; '' = no v1 class. Keys are lower-cased transportType.
TO_V1 = {"bicycle": "bike", "bus": "bus", "car": "car", "metro": "subway", "train": "train",
         "tram": "", "boat": "", "e-scooter": "", "escooter": "", "e_scooter": "", "inside": "", "outside": ""}
EXCLUDED_CONFLICT = {"inside", "outside"}
OUTSIDE_V1 = {"tram": "tram", "boat": "ferry/boat", "e-scooter": "e-scooter", "escooter": "e-scooter",
              "e_scooter": "e-scooter"}

COLUMNS = ["installationId", "journeyNumber", "tripNumber", "manufacturer", "model", "timestamp", "transportType",
           "typeInteger", "typeString", "deviceLocation", "value_0", "value_1", "value_2", "value_3", "value_4",
           "value_5", "activity_recognition", "OS"]


def _t_ms(col: pa.ChunkedArray | pa.Array) -> tuple[np.ndarray, int, list[str]]:
    """Timestamp strings -> int64 ms; (values, n unparsed, up to 5 strings Arrow could not parse). Arrow's ISO 8601
    parser first; if it fails on the batch, pandas (format='ISO8601', unit fixed to ms: pandas keeps the input's
    resolution, so a plain astype('int64') is not ns)."""
    try:
        return pc.cast(pc.cast(col, pa.timestamp("us")), pa.int64()).to_numpy(zero_copy_only=False) // 1000, 0, []
    except pa.ArrowInvalid:
        s = pd.Series(col.to_pylist())
        t = pd.to_datetime(s, format="ISO8601", errors="coerce")
        ms = t.dt.as_unit("ms").astype("int64").to_numpy()
        ok = ~t.isna().to_numpy()
        bad = [v for v in pd.unique(s[~s.str.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(\.\d{1,6})?", na=False)])][:5]
        return np.where(ok, ms, np.iinfo(np.int64).min), int((~ok).sum()), bad


def convert(progress=None, max_batches: int | None = None) -> str:
    """One pass: events per user, trips and counts tables. Returns a one-line summary. max_batches: test runs only."""
    opts = pacsv.ConvertOptions(column_types={c: pa.string() for c in COLUMNS if not c.startswith("value_")}
                                | {f"value_{i}": pa.float64() for i in range(6)}
                                | {"journeyNumber": pa.int32(), "tripNumber": pa.int32()})
    trips, counts, bad_t, n_rows, fallback = [], [], 0, 0, []
    with zipfile.ZipFile(ZIP) as z, z.open(MEMBER) as f:
        rd = pacsv.open_csv(io.BufferedReader(f, 8 << 20), read_options=pacsv.ReadOptions(block_size=CHUNK_BYTES),
                            convert_options=opts)
        for b, batch in enumerate(rd):
            if max_batches is not None and b >= max_batches:
                break
            df = batch.to_pandas()
            n_rows += len(df)
            counts.append(df.groupby(["OS", "typeString"], dropna=False).size().rename("events").reset_index())
            df["t_ms"], nb, ex = _t_ms(batch.column("timestamp"))
            bad_t += nb
            if ex:
                fallback.append((b, ex))
            key = ["installationId", "journeyNumber", "tripNumber"]
            trips.append(df.groupby(key + ["manufacturer", "model", "OS", "transportType", "deviceLocation"],
                                    dropna=False).agg(events=("t_ms", "size"), t_first=("t_ms", "min"),
                                                      t_last=("t_ms", "max")).reset_index())
            keep = df[~df.typeString.fillna("").str.contains(DROP)]
            vals = ~keep.typeString.fillna("").str.contains(TIME_ONLY)
            ev = pd.DataFrame({"journey": keep.journeyNumber, "trip": keep.tripNumber,
                               "sensor": keep.typeString.str.replace("android.sensor.", "", regex=False),
                               "t_ms": keep.t_ms, "label": keep.transportType, "position": keep.deviceLocation})
            for i, c in enumerate("xyz"):
                ev[c] = keep[f"value_{i}"].where(vals).astype(np.float32)
            for user, g in ev.groupby(keep.installationId, sort=False):
                p = OUT / "events" / f"user={user}" / f"part-{b:05d}.parquet"
                p.parent.mkdir(parents=True, exist_ok=True)
                tb = pa.Table.from_pandas(g.sort_values(["journey", "trip", "sensor", "t_ms"]), preserve_index=False)
                pq.write_table(tb, p, compression="zstd")
            if progress:
                progress(f"batch {b}: {n_rows:,} rows")
    tr = pd.concat(trips).groupby(["installationId", "journeyNumber", "tripNumber", "manufacturer", "model", "OS",
                                   "transportType", "deviceLocation"], dropna=False).agg(
        events=("events", "sum"), t_first=("t_first", "min"), t_last=("t_last", "max")).reset_index()
    tr.rename(columns={"installationId": "user", "journeyNumber": "journey", "tripNumber": "trip", "OS": "os",
                       "transportType": "label", "deviceLocation": "position"}).to_parquet(OUT / "trips.parquet",
                                                                                           index=False)
    ct = pd.concat(counts).groupby(["OS", "typeString"], dropna=False).events.sum().reset_index()
    ct["kept"] = np.where(ct.typeString.fillna("").str.contains(DROP), "dropped",
                          np.where(ct.typeString.fillna("").str.contains(TIME_ONLY), "time only", "values"))
    ct.to_parquet(OUT / "counts.parquet", index=False)
    pd.DataFrame(fallback, columns=["batch", "examples"]).astype(str).to_parquet(OUT / "timestamp_fallback.parquet")
    return (f"{n_rows:,} rows, {tr.installationId.nunique()} users, {bad_t} unparsed timestamps, "
            f"{len(fallback)} batches parsed by the fallback")


def trips() -> pd.DataFrame:
    """One row per (user, journey, trip, manufacturer, model, os, label, position) combination seen."""
    return pd.read_parquet(OUT / "trips.parquet")


def counts() -> pd.DataFrame:
    return pd.read_parquet(OUT / "counts.parquet")


def users() -> list[str]:
    return sorted(p.name.split("=", 1)[1] for p in (OUT / "events").iterdir())


def stream(user: str, sensors: list[str] | None = None) -> pd.DataFrame:
    """Raw-stream accessor: the kept events of one installation (optionally only some sensors), sorted by
    (journey, trip, sensor, t_ms). Strings come back as categoricals (the largest installation has ~30 M rows)."""
    tb = pq.read_table(OUT / "events" / f"user={user}", filters=[("sensor", "in", sensors)] if sensors else None,
                       read_dictionary=["sensor", "label", "position"])
    d = tb.to_pandas()
    return d.sort_values(["journey", "trip", "sensor", "t_ms"], kind="stable").reset_index(drop=True)
