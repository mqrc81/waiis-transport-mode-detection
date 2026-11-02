"""MobilitApp Sensor Dataset reader (Aguilar Igartua et al., doi:10.34810/data1948; experiment 08 README).

Raw: <data>/raw/mobilitapp/unzipped/MobilitApp-SensorDataset-OK/*.csv, file name
`<id16>_[<NNNN>_]<mode>_<sensors>_<YYYYMMDDhhmmss>_<seg6>.csv`; columns Timestamp, ACC{X,Y,Z} (Android linear
acceleration, m/s2, gravity removed), MAG{X,Y,Z} (uT), GYR{X,Y,Z} (rad/s). The id is a phone / installation (it
persists across trips), a trip is (id, stamp), each trip is one mode, segments are back to back.
Timestamps (README, stop-point 1): epoch ms only in 2022-11 - 2023-06; afterwards a fill value (99999999999) or one
second-resolution value per file. **Inventory only** (approved 2026-11-02): no plausibility check, no window tables;
untimed files are counted, never clean.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from tmd.paths import RAW

DIR = RAW / "mobilitapp" / "unzipped" / "MobilitApp-SensorDataset-OK"
NAME = re.compile(r"^(.{16})_(?:(\d{4})_)?(.+?)_([A-Z]+(?:-[A-Z]+)*)_(\d{14})_(\d{6})\.csv$")

# File-name mode -> canonical mode (English, Catalan and Spanish spellings).
MODE = {"Stationary": "still", "Estacionario": "still", "Estacionari": "still", "Walk": "walk", "Caminar": "walk",
        "Run": "run", "Bici": "bike", "Bicycle": "bike", "Car": "car", "Cotxe": "car", "Bus": "bus",
        "Train": "train", "Tren": "train", "Metro": "subway", "Tram": "tram", "Moto": "motorbike",
        "e-Scooter": "e-scooter", "E-Scooter": "e-scooter", "e_Scooter": "e-scooter", "e-Bike": "e-bike",
        "e_Bicycle": "e-bike", "e-Bicicleta": "e-bike", "e-Bicing": "e-bike"}
V1 = {"still", "walk", "run", "bike", "car", "bus", "train", "subway"}
OUTSIDE_V1 = {"tram", "motorbike", "e-scooter", "e-bike"}
PLATFORM = "android"


def _kind(v: str) -> str:
    if re.fullmatch(r"\d{13}", v):
        return "epoch_ms"
    if v == "99999999999":
        return "fill"
    if re.fullmatch(r"\d{14}", v):
        return "yyyymmddhhmmss"
    if re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", v):
        return "datetime_s"
    return "other"


def files() -> pd.DataFrame:
    """One row per file: id, counter, mode (raw and canonical), sensors, stamp (local time), seg, bytes,
    time_kind (from the first row)."""
    rows = []
    for p in sorted(DIR.glob("*.csv")):
        m = NAME.match(p.name)
        with open(p) as f:
            f.readline()
            first = f.readline().split(",")[0].strip('"')
        rows.append({"name": p.name, "id": m[1], "counter": m[2], "mode_raw": m[3], "sensors": m[4],
                     "stamp": pd.Timestamp(m[5]), "seg": int(m[6]), "bytes": p.stat().st_size,
                     "time_kind": _kind(first)})
    df = pd.DataFrame(rows)
    df["mode"] = df.mode_raw.map(MODE).fillna("?" + df.mode_raw)
    df["cls"] = df["mode"].where(df["mode"].isin(V1), "")
    df["trip"] = df.id + "_" + df.stamp.dt.strftime("%Y%m%d%H%M%S")
    return df


def read(name: str) -> pd.DataFrame:
    """One file; Timestamp as int64 ms where the file has epoch ms (else NaN); sensor columns numeric (bad cells
    -> NaN); rows without any accelerometer value dropped."""
    d = pd.read_csv(DIR / name, dtype=str)
    t = d.pop("Timestamp")
    d = d.apply(pd.to_numeric, errors="coerce")
    d.insert(0, "t_ms", pd.to_numeric(t.where(t.str.fullmatch(r"\d{13}")), errors="coerce"))
    return d.dropna(subset=["ACCX", "ACCY", "ACCZ"]).reset_index(drop=True)


def trip_stream(f: pd.DataFrame, trip: str) -> pd.DataFrame:
    """Raw-stream accessor: the ACC-MAG-GYR files of one trip, in segment order, with the segment number."""
    g = f[(f.trip == trip) & (f.sensors == "ACC-MAG-GYR")].sort_values("seg")
    return pd.concat([read(n).assign(seg=s) for n, s in zip(g.name, g.seg)], ignore_index=True)
