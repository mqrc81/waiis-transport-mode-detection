"""Window tables per dataset (README: Plan, checks 1-5, 7-9), from the converted Parquet events.

    uv run python build_windows.py collecty [--workers 6]
    uv run python build_windows.py nor_tmd

Writes <data>/processed/<dataset>_windows_25hz_w15s/:
    minutes.parquet        one row per (trip, clock minute): v1 grid, status, rates, flags, sensors, cls
    accelerometer.parquet  user, trip, ts, x, y, z (375 floats each, m/s2, gravity included) for status 'ok'
    tiles.parquet          back-to-back 15 s windows per trip (clean hours), no arrays
    timing.parquet         per trip and sensor (tmd.candidates.timing)
    legs.parquet           runs of one label per trip
    clipping.parquet       per trip: max |acc|, per-axis max and events at it
Rules: tmd.candidates. Full log in logs/build_windows_<dataset>.log.
"""
import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tmd import candidates as q
from tmd.paths import PROCESSED
from tmd.progress import Progress
from tmd.resample import to_ms2

HERE = Path(__file__).resolve().parent
EXP = HERE.name
log = logging.getLogger("build_windows")


# ---------------------------------------------------------------- Collecty

def collecty_jobs() -> list:
    from tmd.readers import collecty
    f = collecty.files()
    return [(int(u), int(t)) for u, t in zip(f.user, f.traj)]


def collecty_trip(job):
    """Per trajectory: 12 h wraps repaired, pieces split at other backward steps (tmd.readers.collecty.repair_time); each
    piece is one qc_trip, and windows overlapping another piece's time span get status 'time_overlap'."""
    from tmd.readers import collecty
    user, traj = job
    a, g, m = (collecty.stream(user, traj, k) for k in ("acc", "gyro", "mag"))
    ta, piece, wraps, back = collecty.repair_time(a.t_ms.to_numpy())
    tg, _, _, _ = collecty.repair_time(g.t_ms.to_numpy())
    tm, _, _, _ = collecty.repair_time(m.t_ms.to_numpy())
    xyz, lab = a[["ax", "ay", "az"]].to_numpy(), a.label.to_numpy()
    spans = [(ta[piece == p].min(), ta[piece == p].max()) for p in np.unique(piece)]
    out = []
    for p, (lo, hi) in enumerate(spans):
        k = piece == p
        if k.sum() < 2:
            continue
        inside = lambda t: t[(t >= lo) & (t <= hi)]
        r = q.qc_trip(ta[k], xyz[k], lab[k], inside(tg), inside(tm))
        others = [s for i, s in enumerate(spans) if i != p]
        for name in ("minutes", "tiles"):
            df = r[name]
            if len(df) and others:
                s0 = df.t_start.to_numpy() * 1e3
                ov = np.zeros(len(df), bool)
                for o0, o1 in others:
                    ov |= (s0 < o1) & (s0 + q.WIN_S * 1e3 > o0)
                was_ok = df.status.eq("ok").to_numpy()
                if name == "minutes" and (ov & was_ok).any():
                    r["wins"] = r["wins"][~ov[was_ok]]
                df.loc[ov, "status"] = "time_overlap"
        meta = {"user": str(user), "trip": f"{traj}.{p}", "platform": "android", "device": str(user), "model": "",
                "position": "", "acc_unit": "m/s2", "clock_wraps": wraps, "clock_back_steps": back}
        r["timing"], r["legs"] = r["timing"].iloc[:0], r["legs"].iloc[:0]     # per file, below
        out.append((meta, r))
    # timing and legs per raw trajectory (all pieces), legs in file order; carried by the first piece
    if out:
        tim = [{"sensor": "acc", **q.timing(ta)}, {"sensor": "gyro", **q.timing(tg)}, {"sensor": "mag", **q.timing(tm)}]
        out[0][1]["timing"] = pd.DataFrame(tim)
        out[0][1]["legs"] = q.legs(ta, lab, ordered=True)
        out[0][1]["clipping"] = q.clipping(xyz)
        for _, r in out[1:]:
            r["clipping"] = {}
    return out


# ---------------------------------------------------------------- NOR-TMD

# From the conversion's counts table (README, Results). Android TYPE_ACCELEROMETER and iOS CMAccelerometerData both
# include gravity (iOS in g: converted per trip by tmd.resample.to_ms2).
ACC_SENSORS = ("accelerometer", "CMAccelerometerData")
GYRO_SENSORS = ("gyroscope", "gyroscope_uncalibrated", "CMGyroData", "CMDeviceMotion.rotationRate")
MAG_SENSORS = ("magnetic_field", "magnetic_field_uncalibrated", "CMMagnetometerData")
PRESS_SENSORS = ("pressure", "CMAltitudeData")
BOOT_CLOCK_BEFORE_MS = 946_684_800_000          # 2000-01-01


def nor_tmd_jobs() -> list:
    from tmd.readers import nor_tmd
    return nor_tmd.users()


def _pick(d: pd.DataFrame, names) -> pd.DataFrame:
    """The densest of the named sensors in a trip (e.g. gyroscope vs gyroscope_uncalibrated)."""
    s = d[d.sensor.str.lower().isin([n.lower() for n in names])]
    if s.empty:
        return s
    best = s.sensor.value_counts().idxmax()
    return s[s.sensor == best]


def nor_tmd_user(user):
    from tmd.readers import nor_tmd
    d = nor_tmd.stream(user, list(ACC_SENSORS + GYRO_SENSORS + MAG_SENSORS + PRESS_SENSORS))
    d["sensor"] = d.sensor.astype(str)
    tr = nor_tmd.trips()
    tr = tr[tr.user == user]
    out = []
    for (j, k), e in d.groupby(["journey", "trip"], observed=True):
        e = e.assign(label=e.label.astype(str), position=e.position.astype(str))
        info = tr[(tr.journey == j) & (tr.trip == k)].iloc[0] if ((tr.journey == j) & (tr.trip == k)).any() else None
        # guard: times before 2000 would be a second (non-epoch) clock; a trip with both clocks is split into pieces
        # (none expected once timestamps parse correctly; counted in the report as mixed_clocks)
        clock = np.where(e.t_ms < BOOT_CLOCK_BEFORE_MS, "boot", "wall")
        clocks = sorted(set(clock))
        for c in clocks:
            ec = e[clock == c]
            a = _pick(ec, ACC_SENSORS)
            meta = {"user": user, "trip": f"{j}_{k}" + (f".{c}" if len(clocks) > 1 else ""), "clock": c,
                    "platform": (str(info.os).lower() if info is not None else ""), "device": user,
                    "model": (f"{info.manufacturer} {info.model}" if info is not None else ""),
                    "position": "|".join(sorted(ec.position.dropna().unique())),
                    "acc_sensor": a.sensor.iloc[0] if len(a) else "", "mixed_clocks": len(clocks) > 1}
            if a.empty:
                out.append((meta, None))
                continue
            xyz, unit = to_ms2(a[["x", "y", "z"]].to_numpy(float))      # per piece: median |acc| < 3 -> g (as v1)
            meta["acc_unit"] = unit
            g, m, p = _pick(ec, GYRO_SENSORS), _pick(ec, MAG_SENSORS), _pick(ec, PRESS_SENSORS)
            r = q.qc_trip(a.t_ms, xyz, a.label.str.lower(), g.t_ms if len(g) else None, m.t_ms if len(m) else None,
                          p.t_ms if len(p) else None)
            out.append((meta, r))
    return out


DATASETS = {"collecty": (collecty_jobs, collecty_trip), "nor_tmd": (nor_tmd_jobs, nor_tmd_user)}


def to_v1(dataset: str):
    from tmd.readers import collecty
    from tmd.readers import nor_tmd
    return (lambda s: collecty.TO_V1.get(s, "")) if dataset == "collecty" else (lambda s: nor_tmd.TO_V1.get(s, ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(DATASETS))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    (HERE / "logs").mkdir(exist_ok=True)
    logging.basicConfig(filename=HERE / "logs" / f"build_windows_{a.dataset}.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    out = PROCESSED / f"{a.dataset}_windows_25hz_w15s"
    out.mkdir(parents=True, exist_ok=True)
    jobs_fn, work = DATASETS[a.dataset]
    jobs = jobs_fn()
    prog = Progress(EXP, f"build_windows_{a.dataset}", total=len(jobs), log=log)
    t0 = time.time()
    tabs = {k: [] for k in ("minutes", "tiles", "timing", "legs", "clipping")}
    wins = []
    with ProcessPoolExecutor(a.workers) as ex:
        for job, res in zip(jobs, ex.map(work, jobs)):
            for meta, r in (res if isinstance(res, list) else [res]):
                if r is None:
                    log.info(f"no accelerometer: {meta}")
                    continue
                for k in ("minutes", "tiles", "timing", "legs"):
                    if len(r[k]):
                        tabs[k].append(r[k].assign(**meta))
                if r["clipping"]:
                    tabs["clipping"].append(pd.DataFrame([{**meta, **r["clipping"]}]))
                if len(r["wins"]):
                    ok = r["minutes"][r["minutes"].status == "ok"]
                    wins.append((ok.assign(**meta)[["user", "trip", "ts"]], r["wins"]))
            prog.step(str(job))
    cls = to_v1(a.dataset)
    for k, v in tabs.items():
        df = pd.concat(v, ignore_index=True) if v else pd.DataFrame()
        if k in ("minutes", "tiles") and len(df):
            df["cls"] = df.label.map(cls).fillna("")
            q.flag(df)
        if k == "legs" and len(df):
            df["cls"] = df.label.map(cls).fillna("")
        df.to_parquet(out / f"{k}.parquet", index=False)
        tabs[k] = df
    meta = pd.concat([m for m, _ in wins], ignore_index=True) if wins else pd.DataFrame(columns=["user", "trip", "ts"])
    w = np.concatenate([x for _, x in wins]) if wins else np.zeros((0, q.WIN, 3), np.float32)
    arr = lambda i: pa.FixedSizeListArray.from_arrays(pa.array(np.ascontiguousarray(w[:, :, i]).ravel(), pa.float32()), q.WIN)
    pq.write_table(pa.table({"user": meta.user.astype(str), "trip": meta.trip.astype(str), "ts": meta.ts.astype(np.int64),
                             "x": arr(0), "y": arr(1), "z": arr(2)}), out / "accelerometer.parquet", compression="zstd")
    m, t = tabs["minutes"], tabs["tiles"]
    msg = (f"{a.dataset}: {len(jobs)} jobs, {len(m)} minutes ({int(m.clean.sum())} clean, "
           f"{int((m.clean & m.cls.ne('')).sum())} clean with a v1 class), clean hours {t.clean.sum() * 15 / 3600:.1f}; "
           f"{time.time() - t0:.0f} s -> {out}")
    log.info(msg)
    print(msg)


if __name__ == "__main__":
    main()
