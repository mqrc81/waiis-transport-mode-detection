"""GPS feature tables (tmd.gps_features, gps-v1 / gps-v2) for every usable minute of 05's feature tables (same rows, same order).

    uv run python make_gps.py                       # dev: extrasensory, shl_user1
    uv run python make_gps.py --part test           # stop-point 3 only: extrasensory test users, shl_user23
    uv run python make_gps.py --version gps-v2      # 06 stop-point 2 follow-up
Writes <data>/processed/06_gps-acc-baseline_waiis-bench/<source>_<part>_<version>_w2min.parquet and prints coverage.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tmd import bench
from tmd import gps_features as gf
from tmd.paths import PROCESSED

HERE = Path(__file__).resolve().parent
OUT = PROCESSED / HERE.name
P05 = PROCESSED / "05_acc-baseline_waiis-bench"
KEYS = ["user", "ts", "session", "official", "y", "platform", "fold", "quiet"]


def gps_path(source: str, part: str, version: str = gf.FEATURES_VERSION) -> Path:
    return OUT / f"{source}_{part}_{version}_w2min.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["dev", "test"], default="dev")
    ap.add_argument("--version", choices=list(gf.VERSIONS), default=gf.FEATURES_VERSION)
    ap.add_argument("--out", help="write here instead of the cache (e.g. to compare a regenerated table)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    srcs = ["extrasensory", "shl_user1"] if a.part == "dev" else ["extrasensory", "shl_user23"]
    for s in srcs:
        t0 = time.time()
        keys = pd.read_parquet(P05 / f"{s}_{a.part}_25hz_w15s_acc-v1_lp-none.parquet", columns=KEYS)
        gps = bench.read(s, "gps_per_minute")
        keep = bench.reserved_users(s)
        gps = gps[~gps.user.isin(keep)] if a.part == "dev" else gps[gps.user.isin(keep)]
        f = gf.features(keys, gps, a.version)
        df = pd.concat([keys, pd.DataFrame(f, columns=gf.names(a.version))], axis=1)
        dest = gps_path(s, a.part, a.version) if not a.out else Path(a.out) / gps_path(s, a.part, a.version).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dest, index=False)
        cov = gf.covered(f)
        off = keys.official.to_numpy()
        print(f"{s} {a.part}: {len(df)} rows, covered {cov.mean():.3f} (official {cov[off].mean():.3f}), "
              f"{time.time() - t0:.0f}s -> {dest}")


if __name__ == "__main__":
    main()
