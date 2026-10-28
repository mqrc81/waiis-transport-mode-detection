"""Reader for the frozen benchmark es_bench_v1 (built by experiments/04_frozen-benchmark_es-shl/build.py).

    <data>/processed/es_bench_v1/manifest.json                 version, parameters, sha256 of every file
    <data>/processed/es_bench_v1/<source>/<table>.parquet       source in SOURCES, table in TABLES[source]

Tables join on (user, ts): ts is the minute timestamp [unix s]. `labels` has one row per minute (every minute is
kept, with flags), `accelerometer` one 15 s window (375 x 3, m/s2, 25 Hz) per minute that has one.
The official example set is `examples(source)`: included minutes that have a window.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from tmd.paths import PROCESSED

VERSION = "es_bench_v1"
ROOT = PROCESSED / VERSION
FS, WIN = 25, 375
SOURCES = ("extrasensory", "shl_user1", "shl_user23")
TABLES = {
    "extrasensory": ("labels", "sessions", "accelerometer", "gps_per_minute", "location_features"),
    "shl_user1": ("labels", "sessions", "accelerometer", "gps_waiis_cadence", "gps_per_minute"),
    "shl_user23": ("labels", "sessions", "accelerometer", "gps_waiis_cadence", "gps_per_minute"),
}
INCLUDED = {"extrasensory": "included_v2", "shl_user1": "included", "shl_user23": "included"}
KEY = ["user", "ts"]


def path(source: str, table: str):
    assert table in TABLES[source], (source, table)
    return ROOT / source / f"{table}.parquet"


def read(source: str, table: str, columns: list[str] | None = None, filters=None) -> pd.DataFrame:
    """One table as a DataFrame; `filters` as in pyarrow (e.g. [("split", "==", "dev")])."""
    return pd.read_parquet(path(source, table), columns=columns, filters=filters)


def windows(acc: pd.DataFrame) -> np.ndarray:
    """(n, 375, 3) float32 from the x, y, z list columns of an `accelerometer` frame."""
    return np.stack([np.stack(acc[c].to_numpy()) for c in ("x", "y", "z")], axis=-1).astype(np.float32)


def examples(source: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Label rows of the official example set: included minutes with an accelerometer window."""
    lab = read(source, "labels", columns=None if columns is None else list(dict.fromkeys(
        KEY + columns + [INCLUDED[source], "has_acc"])))
    return lab[lab[INCLUDED[source]] & lab.has_acc].reset_index(drop=True)


def manifest() -> dict:
    return json.loads((ROOT / "manifest.json").read_text())


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify() -> list[str]:
    """Files whose sha256 differs from the manifest (empty list = intact)."""
    return [f for f, m in manifest()["files"].items() if sha256(ROOT / f) != m["sha256"]]
