"""Collecty reader (Erdelic et al., Data in Brief 50, 2023; experiment 08 README, stop-point 1).

Raw: <data>/raw/collecty/collecty.zip (Kaggle merdelic/dataset-for-multimodal-transport-analytics v4), kept zipped.
`Collecty data/archive/<user>_<trajectory>.txt`: a 10 ms timer copies the latest value of each sensor, and
`lChangeTime<sensor>` is that sensor's last new event. So rows are held values; the real samples are the distinct
change times (approved 2026-11-02). acc* is linear acceleration, grav* gravity, both m/s2: total = acc + grav.
Timestamps have no time zone; they are used as given.

Converted (`convert`, one pass over the zip):
    <data>/processed/collecty_events_acc-grav-gyro_mag-times/
        acc/user=<u>/<traj>.parquet    traj, t_ms, ax, ay, az (total = acc + grav), label: one row per distinct
                                       lChangeTimeacc (first row of each)
        gyro/user=<u>/<traj>.parquet   traj, t_ms, x, y, z: one row per distinct lChangeTimegyro
        mag/user=<u>/<traj>.parquet    traj, t_ms: distinct lChangeTimemagn
        files.parquet                  one row per raw file: rows, ticks, held rows, labels, time problems
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from tmd.paths import PROCESSED, RAW

ZIP = RAW / "collecty" / "collecty.zip"
OUT = PROCESSED / "collecty_events_acc-grav-gyro_mag-times"
NAME = re.compile(r"^Collecty data/archive/(\d+)_(\d+)\.txt$")
TIME_COLS = ["sampleTime", "lChangeTimeacc", "lChangeTimegyro", "lChangeTimegrav", "lChangeTimemagn"]
PLATFORM = "android"

# v1 class per Collecty label (approved 2026-11-02): Walk includes standing -> excluded; '' = no v1 class.
TO_V1 = {"Run": "run", "Bike": "bike", "Car": "car", "Bus": "bus", "Train": "train",
         "Walk": "", "Tramway": "", "E scooter": "", "Unknown": ""}
EXCLUDED_CONFLICT = {"Walk"}         # conflicts with v1 (still inside walk)
OUTSIDE_V1 = {"Tramway": "tram", "E scooter": "e-scooter"}


def members() -> list[tuple[str, int, int]]:
    """(zip member, user, trajectory) for every trajectory file."""
    with zipfile.ZipFile(ZIP) as z:
        return [(n, int(m[1]), int(m[2])) for n in z.namelist() if (m := NAME.match(n))]


def _read(member: str) -> pa.Table:
    with zipfile.ZipFile(ZIP) as z, z.open(member) as f:
        types = {c: pa.timestamp("us") for c in TIME_COLS}
        return pacsv.read_csv(io.BufferedReader(f, 1 << 20),
                              convert_options=pacsv.ConvertOptions(column_types=types))   # ISO 8601, fraction optional


def _ms(col) -> np.ndarray:
    return pc.cast(col, pa.int64()).to_numpy(zero_copy_only=False) // 1000


def convert_one(job: tuple[str, int, int]) -> dict:
    member, user, traj = job
    tb = _read(member)
    st, ta, tg, tm = (_ms(tb[c]) for c in ("sampleTime", "lChangeTimeacc", "lChangeTimegyro", "lChangeTimemagn"))
    lab = tb["label"].to_numpy(zero_copy_only=False).astype(str)
    num = {c: tb[c].to_numpy(zero_copy_only=False).astype(np.float32) for c in
           ["accX", "accY", "accZ", "gravX", "gravY", "gravZ", "gyroX", "gyroY", "gyroZ"]}
    first = lambda t: np.concatenate([[True], t[1:] != t[:-1]])     # first row of each run of a held value
    ka, kg, km = first(ta), first(tg), first(tm)
    acc = pd.DataFrame({"traj": np.int16(traj), "t_ms": ta[ka],
                        **{f"a{x}": num[f"acc{x.upper()}"][ka] + num[f"grav{x.upper()}"][ka] for x in "xyz"},
                        "label": lab[ka]})
    gyro = pd.DataFrame({"traj": np.int16(traj), "t_ms": tg[kg], **{x: num[f"gyro{x.upper()}"][kg] for x in "xyz"}})
    mag = pd.DataFrame({"traj": np.int16(traj), "t_ms": np.unique(tm[km])})
    for kind, df in (("acc", acc), ("gyro", gyro), ("mag", mag)):
        p = OUT / kind / f"user={user}" / f"{traj}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), p, compression="zstd")
    dst = np.diff(st)
    legs = (lab[1:] != lab[:-1]).sum() + 1 if len(lab) else 0
    return {"user": user, "traj": traj, "rows": len(st), "t_first_ms": int(st[0]), "t_last_ms": int(st[-1]),
            "tick_median_ms": float(np.median(dst)) if len(dst) else np.nan, "tick_gaps_over_0.5s": int((dst > 500).sum()),
            "ticks_backwards": int((dst < 0).sum()), "acc_events": int(ka.sum()), "gyro_events": int(kg.sum()),
            "mag_events": int(km.sum()), "acc_change_backwards": int((np.diff(ta[ka]) < 0).sum()),
            "acc_change_repeated": int(len(ta[ka]) - len(np.unique(ta[ka]))), "legs": int(legs),
            "labels": "|".join(pd.unique(lab)), "null_values": int(sum(np.isnan(v).sum() for v in num.values()))}


def convert(workers: int = 6, progress=None) -> pd.DataFrame:
    """One pass over the zip; returns and writes the files table."""
    from concurrent.futures import ProcessPoolExecutor
    jobs = members()
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(convert_one, jobs):
            rows.append(r)
            if progress:
                progress(r)
    files = pd.DataFrame(rows).sort_values(["user", "traj"])
    files.to_parquet(OUT / "files.parquet", index=False)
    return files


def files() -> pd.DataFrame:
    return pd.read_parquet(OUT / "files.parquet")


def stream(user: int, traj: int, kind: str = "acc") -> pd.DataFrame:
    """Raw-stream accessor: one converted table (acc: total acceleration m/s2 and label per event)."""
    return pd.read_parquet(OUT / kind / f"user={user}" / f"{traj}.parquet")


WRAP_MS = 12 * 3_600_000      # 12-hour clock without AM / PM: 12:59:59.998 -> 01:00:00.009 is a -12 h step
WRAP_TOL_MS = 5_000


def repair_time(t_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Event times in file order -> (times with 12 h wraps undone, piece id per event, n wraps, n other backward
    steps). A step of -12 h (+- 5 s) adds 12 h to everything after it; any other backward step starts a new piece
    (the clock is not trusted across it; experiment 08 README, Results)."""
    t = np.asarray(t_ms, np.int64).copy()
    d = np.diff(t)
    wrap = np.abs(d + WRAP_MS) < WRAP_TOL_MS
    if wrap.any():
        t[1:] += np.cumsum(wrap) * WRAP_MS
        d = np.diff(t)
    back = d < 0
    piece = np.concatenate([[0], np.cumsum(back)]).astype(np.int32)
    return t, piece, int(wrap.sum()), int(back.sum())


def duplicates() -> pd.DataFrame:
    """Byte-identical trajectory files (zip CRC-32 and size): user, traj, group, drop. Proposal (experiment 08
    README, Results): within a user the lower trajectory number is kept; a file shared by several users is dropped
    in every copy (the owner is unknown)."""
    with zipfile.ZipFile(ZIP) as z:
        rows = [(int(m[1]), int(m[2]), i.CRC, i.file_size) for i in z.infolist() if (m := NAME.match(i.filename))]
    f = pd.DataFrame(rows, columns=["user", "traj", "crc", "size"])
    d = f[f.duplicated(["crc", "size"], keep=False)].sort_values(["crc", "user", "traj"]).copy()
    d["group"] = d.groupby(["crc", "size"]).ngroup()
    multi_user = d.groupby("group").user.transform("nunique") > 1
    d["drop"] = multi_user | d.duplicated("group", keep="first")
    return d[["user", "traj", "group", "drop"]].reset_index(drop=True)
