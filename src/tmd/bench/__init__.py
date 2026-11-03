"""Reader for the frozen benchmarks: waiis_bench_v2, the default (experiments/09_frozen-benchmark_es-shl-nor-collecty/build.py),
and waiis_bench_v1 (experiments/04_frozen-benchmark_es-shl/build.py), read with version="waiis_bench_v1".

Package layout (experiment 09): this module is the reader API, unchanged; `tmd.bench.build` holds the build code.
The per-dataset readers are in `tmd.readers`, the window rules in `tmd.candidates`.

    <data>/processed/<version>/manifest.json                 parameters, git commit, sha256 of every file
    <data>/processed/<version>/<source>/<table>.parquet       source in sources(version)

Tables join on (user, ts): ts is the minute timestamp [unix s]. `labels` has one row per minute (every minute is
kept, with flags), `accelerometer` one 15 s window (375 x 3, m/s2, 25 Hz) per minute that has one.

Accessors:
- `examples(source)`: **the official example set** (the `official` column). It includes the reserved test users,
  marked by `split`, because evaluation needs them. Excluded users (tmd.readers.extrasensory.EXCLUDED_USERS) are never in it.
- `pretraining_minutes`, `unlabelled_minutes` and the extras (`snippets`, `shl_stream`, `shl_other_positions`,
  `shl_gps_1hz`, `shl_gps_unblanked`): **exclude the reserved test users by default** (v2: `tmd.splits.reserved_test(3)`,
  12 ExtraSensory users, SHL Users 2-3, 14 NOR-TMD installations; v1: `reserved_test(2)`; include_reserved=True only for evaluation) and always the excluded users.
- `read(source, table)`: a raw table, no filtering.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from tmd.paths import PROCESSED

VERSIONS = ("waiis_bench_v1", "waiis_bench_v2")
VERSION = "waiis_bench_v2"         # default since 2026-11-03; waiis_bench_v1 stays readable with version=
ROOT = PROCESSED / VERSION
FS, WIN = 25, 375
SOURCES = ("extrasensory", "shl_user1", "shl_user23")                  # v1's sources
SOURCES_BY_VERSION = {"waiis_bench_v1": SOURCES,
                      "waiis_bench_v2": SOURCES + ("nor_tmd", "collecty")}  # experiment 09
RESERVED_BY_VERSION = {"waiis_bench_v1": 2, "waiis_bench_v2": 3}          # tmd.splits.reserved_test(version)
MANIFEST_DIR = {"waiis_bench_v1": "experiments/04_frozen-benchmark_es-shl/results/manifests",
                "waiis_bench_v2": "experiments/09_frozen-benchmark_es-shl-nor-collecty/results/manifests"}
KEY = ["user", "ts"]


def sources(version: str = VERSION) -> tuple[str, ...]:
    """The sources of a benchmark version."""
    return SOURCES_BY_VERSION[version]


def root(version: str = VERSION):
    assert version in VERSIONS, version
    return PROCESSED / version


def path(source: str, table: str, version: str = VERSION):
    assert source in sources(version), source
    p = root(version) / source / f"{table}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{version} has no table {source}/{table}")
    return p


def read(source: str, table: str, columns: list[str] | None = None, filters=None,
         version: str = VERSION) -> pd.DataFrame:
    """One table as a DataFrame, unfiltered; `filters` as in pyarrow (e.g. [("split", "==", "dev")])."""
    return pd.read_parquet(path(source, table, version), columns=columns, filters=filters)


def windows(acc: pd.DataFrame) -> np.ndarray:
    """(n, 375, 3) float32 from the x, y, z list columns of an `accelerometer` frame. For variable-length frames
    (`snippets`, `shl_stream` chunks) use `arrays`."""
    return np.stack([np.stack(acc[c].to_numpy()) for c in ("x", "y", "z")], axis=-1).astype(np.float32)


def arrays(df: pd.DataFrame) -> list[np.ndarray]:
    """One (n_i, 3) float32 array per row of a variable-length frame (x, y, z list columns)."""
    return [np.column_stack([r.x, r.y, r.z]).astype(np.float32) for r in df[["x", "y", "z"]].itertuples()]


def load_windows(source: str, keep: pd.DataFrame, version: str = VERSION) -> tuple[pd.DataFrame, np.ndarray]:
    """Windows of the (user, ts) rows of `keep`, streamed from the `accelerometer` table: (keys merged with keep,
    (n, 375, 3) float32), in table order. No reserved-user filter: the caller chooses the rows (from experiment 05)."""
    import pyarrow.parquet as pq
    want = set(zip(keep.user, keep.ts))
    keys, arrs = [], []
    for b in pq.ParquetFile(path(source, "accelerometer", version)).iter_batches(1 << 14,
                                                                               columns=["user", "ts", "x", "y", "z"]):
        u, t = b.column("user").to_numpy(zero_copy_only=False), b.column("ts").to_numpy()
        m = np.fromiter(((a, c) in want for a, c in zip(u, t)), bool, len(t))
        if not m.any():
            continue
        xyz = [b.column(c).values.to_numpy().reshape(-1, WIN)[m] for c in ("x", "y", "z")]
        arrs.append(np.stack(xyz, -1))
        keys.append(pd.DataFrame({"user": u[m], "ts": t[m]}))
    k = pd.concat(keys, ignore_index=True)
    return k.merge(keep, on=["user", "ts"], how="left", validate="1:1"), np.concatenate(arrs)


# ---------------------------------------------------------------- reserved test users

def reserved_users(source: str, version: str = VERSION) -> set[str]:
    """Users of `source` in the reserved final test of a benchmark version (v1: `tmd.splits.reserved_test(2)`;
    v2: `reserved_test(3)`, the current set, which adds NOR-TMD installations). Collecty has none."""
    from tmd.splits import reserved_test
    r = reserved_test(RESERVED_BY_VERSION[version])
    if source == "extrasensory":
        return set(r["extrasensory"].uuid)
    if source == "nor_tmd":
        return set(r["nor_tmd"].user)
    if source == "collecty":
        return set()
    return set(r["shl"].user)


def drop_reserved(df: pd.DataFrame, source: str, version: str = VERSION) -> pd.DataFrame:
    return df[~df.user.isin(reserved_users(source, version))].reset_index(drop=True)


def _guard(df: pd.DataFrame, source: str, include_reserved: bool, version: str = VERSION) -> pd.DataFrame:
    """Drop the reserved test users (unless include_reserved) and, always, the excluded users."""
    from tmd.readers.extrasensory import EXCLUDED_USERS
    df = df if include_reserved else drop_reserved(df, source, version)
    return df[~df.user.isin(EXCLUDED_USERS)].reset_index(drop=True)


# ---------------------------------------------------------------- official set

def examples(source: str, columns: list[str] | None = None, version: str = VERSION) -> pd.DataFrame:
    """Label rows of the official example set (`official`). The reserved test users are included (column `split`)."""
    lab = read(source, "labels", None if columns is None else list(dict.fromkeys(KEY + columns + ["official"])),
               version=version)
    return lab[lab.official].reset_index(drop=True)


# ---------------------------------------------------------------- pre-training and extras (reserved users excluded)

def pretraining_minutes(source: str, columns: list[str] | None = None, include_reserved: bool = False,
                        version: str = VERSION) -> pd.DataFrame:
    """Label rows of every minute with a window, labelled or not, for self-supervised pre-training."""
    lab = read(source, "labels", None if columns is None else list(dict.fromkeys(KEY + columns + ["has_acc"])),
               version=version)
    return _guard(lab[lab.has_acc], source, include_reserved, version)


def unlabelled_minutes(source: str, columns: list[str] | None = None, include_reserved: bool = False,
                       version: str = VERSION) -> pd.DataFrame:
    """Label rows of minutes with a window that are not in the official set (unlabelled, dropped, capped, ...)."""
    off = examples(source, version=version)[KEY]
    lab = pretraining_minutes(source, columns, include_reserved, version)
    m = lab.merge(off.assign(_off=True), on=KEY, how="left")
    return m[m._off.isna()].drop(columns="_off").reset_index(drop=True)


def snippets(filters=None, include_reserved: bool = False, version: str = VERSION) -> pd.DataFrame:
    """ExtraSensory: every resampled snippet in full (variable length), incl. short and time-repaired ones."""
    return _guard(read("extrasensory", "snippets", filters=filters, version=version), "extrasensory",
                  include_reserved, version)


def shl_stream(source: str, filters=None, include_reserved: bool = False, version: str = VERSION) -> pd.DataFrame:
    """SHL: continuous 25 Hz hips stream, one row per 25 Hz sample (t_ms, x, y, z, coarse label)."""
    return _guard(read(source, "stream", filters=filters, version=version), source, include_reserved, version)


def stream(source: str, filters=None, include_reserved: bool = False, version: str = VERSION) -> pd.DataFrame:
    """Any source with a `stream` table (SHL; in v2 also NOR-TMD and Collecty): the continuous 25 Hz stream of every
    gap-free stretch, one row per sample. Same as `shl_stream`."""
    return shl_stream(source, filters, include_reserved, version)


def shl_other_positions(include_reserved: bool = False, version: str = VERSION,
                        source: str = "shl_user1") -> pd.DataFrame:
    """SHL User 1 (4phones days): windows of the hand, torso and bag phones. Never in the official set. In v2 also
    `source="shl_user23"` (the rail check's other positions; reserved, so only with include_reserved=True)."""
    return _guard(read(source, "accelerometer_other_positions", version=version), source, include_reserved, version)


def shl_gps_1hz(source: str, include_reserved: bool = False, version: str = VERSION) -> pd.DataFrame:
    """SHL: the full 1 Hz location track (incl. altitude) with the speed / bearing derived from it."""
    return _guard(read(source, "gps_1hz", version=version), source, include_reserved, version)


def shl_gps_unblanked(source: str, include_reserved: bool = False, version: str = VERSION) -> pd.DataFrame:
    """SHL: the true speed / bearing of every kept fix of both GPS variants, before blanking."""
    return _guard(read(source, "gps_unblanked", version=version), source, include_reserved, version)


PRETRAINING_ACCESSORS = ("pretraining_minutes", "unlabelled_minutes", "snippets", "shl_stream", "stream",
                         "shl_other_positions", "shl_gps_1hz", "shl_gps_unblanked")


# ---------------------------------------------------------------- manifest

def manifest(version: str = VERSION) -> dict:
    return json.loads((root(version) / "manifest.json").read_text())


def reference_manifest(version: str = VERSION) -> dict:
    """The committed copy of the manifest: the reference for verifying the data on disk."""
    from tmd.paths import PROJECT
    return json.loads((PROJECT / MANIFEST_DIR[version] / f"{version}.json").read_text())


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify(version: str = VERSION) -> list[str]:
    """Problems against the committed reference manifest (empty list = intact): data files that are missing, extra
    or have another sha256, and an on-disk manifest.json that differs from the reference."""
    ref = reference_manifest(version)["files"]
    on_disk = {str(p.relative_to(root(version))) for p in root(version).rglob("*.parquet")}
    bad = [f"missing: {f}" for f in sorted(set(ref) - on_disk)] + [f"extra: {f}" for f in sorted(on_disk - set(ref))]
    bad += [f for f in sorted(set(ref) & on_disk) if sha256(root(version) / f) != ref[f]["sha256"]]
    if manifest(version) != reference_manifest(version):
        bad.append("manifest.json differs from the committed reference")
    return bad
