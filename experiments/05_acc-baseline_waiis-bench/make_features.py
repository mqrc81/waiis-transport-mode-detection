"""Feature tables (tmd.acc_features, all blocks) for every usable minute, with and without the 10 Hz low-pass.

usable = has_acc and not acc_implausible and not user_excluded (no label enters). Development part only: the reserved
test users are dropped (tmd.bench.drop_reserved); `--part test` (stop-point 3 only) keeps only them.

    uv run python make_features.py                 # extrasensory + shl_user1, dev, lp none and 10 Hz
Writes <data>/processed/05_acc-baseline_waiis-bench/<source>_<part>_25hz_w15s_acc-v1_lp-<none|10hz>.parquet
"""
from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from tmd import acc_features as af
from tmd import bench
from tmd.paths import PROCESSED

HERE = Path(__file__).resolve().parent
EXP = HERE.name
OUT = PROCESSED / EXP
KEYS = ["user", "ts", "session", "official", "y", "platform", "fold", "quiet"]
log = logging.getLogger()


def table_path(source: str, part: str, lp: float | None) -> Path:
    return OUT / f"{source}_{part}_25hz_w15s_{af.FEATURES_VERSION}_lp-{'none' if lp is None else f'{lp:g}hz'}.parquet"


def usable(source: str, part: str) -> pd.DataFrame:
    lab = bench.read(source, "labels")
    u = lab[lab.has_acc & ~lab.acc_implausible & ~lab.user_excluded]
    reserved = bench.reserved_users(source)
    u = u[~u.user.isin(reserved)] if part == "dev" else u[u.user.isin(reserved)]
    return u[KEYS].reset_index(drop=True)


def load_windows(source: str, keep: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    want = set(zip(keep.user, keep.ts))
    keys, arrs = [], []
    for b in pq.ParquetFile(bench.path(source, "accelerometer")).iter_batches(1 << 14, columns=["user", "ts", "x", "y", "z"]):
        u, t = b.column("user").to_numpy(zero_copy_only=False), b.column("ts").to_numpy()
        m = np.fromiter(((a, c) in want for a, c in zip(u, t)), bool, len(t))
        if not m.any():
            continue
        xyz = [b.column(c).values.to_numpy().reshape(-1, bench.WIN)[m] for c in ("x", "y", "z")]
        arrs.append(np.stack(xyz, -1))
        keys.append(pd.DataFrame({"user": u[m], "ts": t[m]}))
    k = pd.concat(keys, ignore_index=True)
    return k.merge(keep, on=["user", "ts"], how="left", validate="1:1"), np.concatenate(arrs)


def _feat(args):
    w, lp = args
    return af.features(w, lowpass_hz=lp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["extrasensory", "shl_user1"])
    ap.add_argument("--part", choices=["dev", "test"], default="dev")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    (HERE / "logs").mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=HERE / "logs" / f"make_features_{a.part}.log", filemode="a", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    names = af.names()
    for src in a.sources:
        t0 = time.time()
        keys, w = load_windows(src, usable(src, a.part))
        log.info("%s %s: %d usable windows loaded in %.0fs", src, a.part, len(w), time.time() - t0)
        for lp in (None, 10.0):
            t1 = time.time()
            chunks = [(w[i:i + 4000], lp) for i in range(0, len(w), 4000)]
            with ProcessPoolExecutor(a.workers) as ex:
                f = np.concatenate(list(ex.map(_feat, chunks)))
            df = pd.concat([keys, pd.DataFrame(f, columns=names)], axis=1)
            p = table_path(src, a.part, lp)
            df.to_parquet(p, index=False)
            msg = (f"{src} {a.part} lp={lp}: {len(df)} rows ({int(df.official.sum())} official), {len(names)} "
                   f"features, {np.isnan(f).mean():.2%} NaN, {time.time() - t1:.0f}s -> {p.name}")
            log.info(msg)
            print(msg)


if __name__ == "__main__":
    main()
