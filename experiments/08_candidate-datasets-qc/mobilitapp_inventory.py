"""MobilitApp inventory (README: stop-point 1 answers, item 4): checks 1, 3, 8 and 9 on the raw files, no windows.

    uv run python mobilitapp_inventory.py [--workers 6]

Writes results/tables/mobilitapp_{files,trips,energy}.csv; full log in logs/mobilitapp_inventory.log.
- files: one row per file (tmd.mobilitapp.files) + samples, md5 (duplicates), bad cells.
- trips: one row per (id, stamp): mode, files, samples, timed (every file epoch ms), and for timed trips
  tmd.candidates.timing of the accelerometer.
- energy (check 8 flags, timed trips): back-to-back 15 s tiles of the raw stream; std |linear acc| per tile.
"""
import argparse
import hashlib
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from tmd import candidates as q
from tmd.progress import Progress

import mobilitapp as mob             # this folder: not used beyond 08

HERE = Path(__file__).resolve().parent
EXP = HERE.name
TABLES = HERE / "results" / "tables"
log = logging.getLogger("mobilitapp_inventory")


def trip_job(args):
    trip, names, segs, timed = args
    out_files, parts = [], []
    for n, s in zip(names, segs):
        raw = (mob.DIR / n).read_bytes()
        d = mob.read(n)
        out_files.append({"name": n, "samples": len(d), "md5": hashlib.md5(raw).hexdigest(),
                          "acc_md5": hashlib.md5(d[["ACCX", "ACCY", "ACCZ"]].to_numpy().tobytes()).hexdigest()})
        parts.append(d.assign(seg=s))
    d = pd.concat(parts, ignore_index=True)
    res = {"trip": trip, "files": out_files, "samples": len(d)}
    if timed:
        t = d.t_ms.to_numpy(np.int64)
        res["timing"] = q.timing(t)
        o = np.argsort(t, kind="stable")
        t, a = t[o] / 1e3, np.linalg.norm(d[["ACCX", "ACCY", "ACCZ"]].to_numpy(float)[o], axis=1)
        tiles = []
        for lo in t[0] + np.arange(int((t[-1] - t[0]) // q.WIN_S)) * q.WIN_S:
            i, j = np.searchsorted(t, [lo, lo + q.WIN_S])
            if j - i >= 2:
                tiles.append({"t_start": lo, "n": j - i, "lin_median": float(np.median(a[i:j])),
                              "lin_std": float(a[i:j].std())})
        res["energy"] = tiles
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=HERE / "logs" / "mobilitapp_inventory.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    t0 = time.time()
    f = mob.files()
    acc = f[f.sensors == "ACC-MAG-GYR"]
    kinds = acc.groupby("trip").time_kind.agg(lambda s: "+".join(sorted(set(s))))
    jobs = [(tr, g.sort_values("seg").name.tolist(), g.sort_values("seg").seg.tolist(), kinds[tr] == "epoch_ms")
            for tr, g in acc.groupby("trip")]
    prog = Progress(EXP, "mobilitapp_inventory", total=len(jobs), log=log)
    fr, trips, timing, energy = [], [], [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(trip_job, jobs, chunksize=8):
            fr += r["files"]
            trips.append({"trip": r["trip"], "samples": r["samples"]})
            if "timing" in r:
                timing.append({"trip": r["trip"], **r["timing"]})
                energy += [{"trip": r["trip"], **e} for e in r["energy"]]
            prog.step(r["trip"])
    files = f.merge(pd.DataFrame(fr), on="name", how="left")
    tg = acc.groupby("trip")
    tr = pd.DataFrame({"id": tg.id.first(), "stamp": tg.stamp.first(), "mode_raw": tg.mode_raw.first(),
                       "mode": tg["mode"].first(), "cls": tg.cls.first(), "modes_in_trip": tg["mode"].nunique(),
                       "files": tg.size(), "time_kind": kinds}).reset_index()
    tr = tr.merge(pd.DataFrame(trips), on="trip").merge(pd.DataFrame(timing), on="trip", how="left")
    tr["timed"] = tr.time_kind.eq("epoch_ms")
    files.to_csv(TABLES / "mobilitapp_files.csv", index=False)
    tr.to_csv(TABLES / "mobilitapp_trips.csv", index=False)
    pd.DataFrame(energy).to_csv(TABLES / "mobilitapp_energy.csv", index=False)
    msg = (f"mobilitapp: {len(files)} files, {len(tr)} trips ({int(tr.timed.sum())} timed, "
           f"{tr[tr.timed].span_s.sum() / 3600:.1f} h timed), {len(energy)} energy tiles; {time.time() - t0:.0f} s")
    log.info(msg)
    print(msg)


if __name__ == "__main__":
    main()
