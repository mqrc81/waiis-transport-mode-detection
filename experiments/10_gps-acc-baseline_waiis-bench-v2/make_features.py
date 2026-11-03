"""Feature tables on waiis_bench_v2: 05's acc features (B: 580, no low-pass) and 06's gps-v2 features, for every
usable minute (has_acc, not acc_implausible, not user_excluded; no label enters).

dev: the dev users of every source (fold >= 0); test (stop-point 3 only): the reserved users. SHL other positions
(hand, torso, bag) as separate tables (`<source>_other`). GPS only for ExtraSensory and SHL (the sources with GPS).

    uv run python make_features.py                    # dev: all sources + shl_user1 other positions
    uv run python make_features.py --part test        # stop-point 3 only
Writes <data>/processed/10_gps-acc-baseline_waiis-bench-v2/<source>_<part>_25hz_w15s_acc-v1-B.parquet and
<source>_<part>_gps-v2_w2min.parquet; log in logs/make_features_<part>.log.
"""
from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from tmd import bench
from tmd.features import acc as af
from tmd.features import gps as gf
from tmd.paths import PROCESSED

BENCH = "waiis_bench_v2"
HERE = Path(__file__).resolve().parent
OUT = PROCESSED / HERE.name
GPS_VERSION = "gps-v2"
KEYS = ["user", "ts", "session", "official", "y", "platform", "fold", "quiet", "day"]   # y = fine id (tram = 9)
SOURCES = {"dev": ["extrasensory", "shl_user1", "nor_tmd", "collecty"], "test": ["extrasensory", "shl_user23", "nor_tmd"]}
GPS_SOURCES = ("extrasensory", "shl_user1", "shl_user23")
log = logging.getLogger()


def acc_path(source: str, part: str) -> Path:
    return OUT / f"{source}_{part}_25hz_w15s_acc-v1-B.parquet"


def gps_path(source: str, part: str) -> Path:
    return OUT / f"{source}_{part}_{GPS_VERSION}_w2min.parquet"


def usable(source: str, part: str) -> pd.DataFrame:
    lab = bench.read(source, "labels", version=BENCH)
    u = lab[lab.has_acc & ~lab.acc_implausible & ~lab.user_excluded]
    reserved = bench.reserved_users(source, version=BENCH)
    u = u[~u.user.isin(reserved) & (u.fold >= 0)] if part == "dev" else u[u.user.isin(reserved)]
    u = u.assign(y=u.y_fine.astype(int), day=u.day.astype(str) if "day" in u else "")
    return u[KEYS].reset_index(drop=True)


def other_positions(source: str, part: str) -> tuple[pd.DataFrame, np.ndarray]:
    """SHL hand / torso / bag windows (plausible ones), with the hips minute's keys and the `position_draw` flag."""
    o = bench.shl_other_positions(include_reserved=part == "test", version=BENCH, source=source)
    o = o[~o.acc_implausible]
    lab = bench.read(source, "labels", version=BENCH)
    lab = lab.assign(y=lab.y_fine.astype(int), day=lab.day.astype(str))
    keys = o[["user", "ts", "position", "position_draw"]].merge(lab[KEYS], on=["user", "ts"], how="left",
                                                                 validate="m:1")
    keys["position"] = keys.position.str.lower()
    return keys.reset_index(drop=True), bench.windows(o)


def _feat(w):
    return af.features(w, True, False)


def featurize(w: np.ndarray, workers: int) -> np.ndarray:
    with ProcessPoolExecutor(workers) as ex:
        return np.concatenate(list(ex.map(_feat, [w[i:i + 4000] for i in range(0, len(w), 4000)])))


def write_gps(name: str, gps_source: str, keys: pd.DataFrame, part: str) -> None:
    gps = bench.read(gps_source, "gps_per_minute", version=BENCH)
    reserved = bench.reserved_users(gps_source, version=BENCH)
    gps = gps[~gps.user.isin(reserved)] if part == "dev" else gps[gps.user.isin(reserved)]
    f = gf.features(keys, gps, GPS_VERSION)
    df = pd.concat([keys[["user", "ts"] + (["position"] if "position" in keys else [])],
                    pd.DataFrame(f, columns=gf.names(GPS_VERSION))], axis=1)
    df.to_parquet(gps_path(name, part), index=False)
    cov = gf.covered(f)
    msg = f"{name} {part} GPS: {len(df)} rows, covered {cov.mean():.3f} -> {gps_path(name, part).name}"
    log.info(msg)
    print(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["dev", "test"], default="dev")
    ap.add_argument("--sources", nargs="+")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    (HERE / "logs").mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=HERE / "logs" / f"make_features_{a.part}.log", filemode="a", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    names = af.names(True, False)
    srcs = a.sources or SOURCES[a.part] + [s + "_other" for s in SOURCES[a.part] if s.startswith("shl")]
    for name in srcs:
        t0 = time.time()
        src = name.removesuffix("_other")
        if name.endswith("_other"):
            keys, w = other_positions(src, a.part)
        else:
            keys, w = bench.load_windows(src, usable(src, a.part), version=BENCH)
        f = featurize(w, a.workers)
        df = pd.concat([keys, pd.DataFrame(f, columns=names)], axis=1)
        df.to_parquet(acc_path(name, a.part), index=False)
        msg = (f"{name} {a.part}: {len(df)} rows ({int(df.official.fillna(False).sum())} official), {len(names)} "
               f"features, {np.isnan(f).mean():.2%} NaN, {time.time() - t0:.0f}s -> {acc_path(name, a.part).name}")
        log.info(msg)
        print(msg, flush=True)
        if src in GPS_SOURCES:
            write_gps(name, src, keys, a.part)


if __name__ == "__main__":
    main()
