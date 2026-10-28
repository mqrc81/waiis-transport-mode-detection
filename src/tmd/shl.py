"""SHL readers. Layout and columns follow doc/doc_dataset.pdf (sections 2-3), checked 2026-10-24.

    release/User1/<rec>/Hips_Motion.txt, Label.txt       User 1, hips only; rec 'mDDMMYY' = a second
                                                          (morning) recording of day DDMMYY
    release/User{1,2,3}_4phones/<rec>/<Pos>_Motion.txt    Pos in Hand, Hips, Torso, Bag

Motion.txt is already on a regular 100 Hz grid, space-separated, one row per Label.txt row.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tmd.paths import SHL_RAW

FS = 100                     # Hz, SHL resampled grid (doc 3.2)
POSITIONS = ["Hand", "Hips", "Torso", "Bag"]

# 0-based columns of <Pos>_Motion.txt (doc 3.2: 1 time ms, 2-4 acc m/s2, 5-7 gyro rad/s, 8-10 mag uT,
# 11-14 orientation wxyz, 15-17 gravity, 18-20 linear acc, 21 pressure hPa, 22 altitude, 23 temperature).
COLS = {"t": [0], "acc": [1, 2, 3], "gyro": [4, 5, 6], "mag": [7, 8, 9], "pressure": [20]}
# Label.txt (doc 3.11): 1 time ms, 2 coarse label (0 Null, 1 Still ... 8 Subway), 3 fine, ...
LABEL_COLS = {"t": 0, "coarse": 1}


@dataclass(frozen=True)
class Recording:
    user: str        # 'User1', 'User2', 'User3'
    session: str     # folder: 'User1', 'User1_4phones', ...
    rec: str         # folder name, e.g. '010317' or 'm010317'
    path: Path

    @property
    def day(self) -> str:
        """Calendar day DDMMYY; 'm010317' and '010317' are the same day."""
        return self.rec.lstrip("m")

    @property
    def positions(self) -> list[str]:
        return [p for p in POSITIONS if (self.path / f"{p}_Motion.txt").exists()]


def recordings(session: str) -> list[Recording]:
    """All recording folders of one session folder, sorted by name."""
    user = session.split("_")[0]
    return [Recording(user, session, p.name, p) for p in sorted((SHL_RAW / session).iterdir())
            if p.is_dir() and (p / "Label.txt").exists()]


def info(rec: Recording) -> dict:
    """00inf.txt as a dict (sessionid, timemsmin, timemsmax, recdate, reclength, recid); {} if absent
    (User1 180417 and 200317 ship without it)."""
    out = {}
    p = rec.path / "00inf.txt"
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        k, _, v = line.partition(":")
        out[k.strip()] = v.strip()
    return out


def read_motion(rec: Recording, position: str, sensors=("acc", "gyro")) -> tuple[np.ndarray, np.ndarray]:
    """Time [ms] (int64) and the requested sensor columns (float32, NaN where missing)."""
    cols = COLS["t"] + [c for s in sensors for c in COLS[s]]
    df = pd.read_csv(rec.path / f"{position}_Motion.txt", sep=" ", header=None, usecols=cols,
                     dtype=np.float64, engine="c")
    a = df[cols].to_numpy()
    return a[:, 0].astype(np.int64), a[:, 1:].astype(np.float32)


def read_location(rec: Recording, position: str) -> pd.DataFrame:
    """<Pos>_Location.txt (doc 3.10): time_ms, accuracy [m, 68% radius], lat, lon, altitude, sorted by time with
    duplicate times dropped. There is no speed or bearing column."""
    df = pd.read_csv(rec.path / f"{position}_Location.txt", sep=" ", header=None, usecols=[0, 3, 4, 5, 6],
                     names=["time_ms", "_a", "_b", "accuracy", "lat", "lon", "altitude"])
    df["time_ms"] = df.time_ms.astype(np.int64)
    return df.sort_values("time_ms").drop_duplicates("time_ms").reset_index(drop=True)


def read_labels(rec: Recording) -> tuple[np.ndarray, np.ndarray]:
    """Time [ms] and coarse label (int8, 0 = Null) per 100 Hz row."""
    df = pd.read_csv(rec.path / "Label.txt", sep=" ", header=None, usecols=[0, 1], dtype=np.int64)
    return df[0].to_numpy(), df[1].to_numpy().astype(np.int8)
