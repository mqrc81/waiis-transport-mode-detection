"""Step 1 (SHL): data audit, one row per recording x position.

Usage: uv run python audit_shl.py [--workers 6]
Writes results/audit/shl_recordings.csv, results/audit/shl_duplicates.csv; full log in logs/audit_shl.log.
"""
import argparse
import hashlib
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from tmd import shl
from tmd.labels import CLASSES

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "audit"
SESSIONS = ["User1", "User1_4phones", "User2_4phones", "User3_4phones"]
ROWS_PER_H = 3600 * shl.FS


def _runs(mask):
    """Lengths of the True runs in a boolean array."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def audit_one(rec: shl.Recording, pos: str) -> dict:
    t, x = shl.read_motion(rec, pos, ("acc", "gyro", "mag", "pressure"))
    tl, y = shl.read_labels(rec)
    assert len(tl) == len(t) and (tl == t).all(), f"{rec.rec} {pos}: Label/Motion time mismatch"
    acc, gyro, mag, pres = x[:, 0:3], x[:, 3:6], x[:, 6:9], x[:, 9]
    dt = np.diff(t)
    ok = ~np.isnan(acc).any(1) & ~np.isnan(gyro).any(1)
    gaps = _runs(~ok)
    still = ok & (y == 1)
    an, gn = np.linalg.norm(acc[still], axis=1), np.linalg.norm(gyro[still], axis=1)
    rep = (np.diff(acc, axis=0) == 0).all(1)      # identical consecutive acc samples: original rate < 100 Hz
    inf = shl.info(rec)
    row = {
        "session": rec.session, "user": rec.user, "rec": rec.rec, "day": rec.day, "position": pos,
        "recdate": inf.get("recdate"), "timemsmin": int(t[0]), "has_00inf": bool(inf), "n_rows": len(t),
        "hours": len(t) / ROWS_PER_H,
        "dt_median_ms": float(np.median(dt)), "dt_min_ms": int(dt.min()), "dt_max_ms": int(dt.max()),
        "n_dt_not_10": int((dt != 10).sum()),
        "frac_acc_nan": float(np.isnan(acc).any(1).mean()), "frac_gyro_nan": float(np.isnan(gyro).any(1).mean()),
        "frac_mag_nan": float(np.isnan(mag).any(1).mean()),
        "frac_pressure_missing": float((np.isnan(pres) | (pres == 0)).mean()),
        "n_gaps_accgyro": len(gaps), "n_gaps_over_1s": int((gaps > shl.FS).sum()),
        "max_gap_s": float(gaps.max() / shl.FS) if len(gaps) else 0.0,
        "frac_acc_repeat": float(rep[ok[1:] & ok[:-1]].mean()),
        "acc_norm_still_median": float(np.median(an)) if len(an) else np.nan,
        "gyro_norm_still_median": float(np.median(gn)) if len(gn) else np.nan,
        "hours_null": float((ok & (y == 0)).sum() / ROWS_PER_H),
        "hours_no_accgyro": float((~ok).sum() / ROWS_PER_H),
    }
    cnt = np.bincount(y[ok], minlength=9)
    row.update({f"h_{c}": cnt[i + 1] / ROWS_PER_H for i, c in enumerate(CLASSES)})
    return row


def head_hash(p: Path, n=50_000_000) -> str:
    with open(p, "rb") as f:
        return hashlib.sha1(f.read(n)).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=HERE / "logs" / "audit_shl.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    log = logging.getLogger()

    recs = [r for s in SESSIONS for r in shl.recordings(s)]
    for s in SESSIONS:
        missing = [p.name for p in (shl.SHL_RAW / s).iterdir() if p.is_dir() and not (p / "Label.txt").exists()]
        log.info("%s: %d recordings, without Label.txt: %s", s, sum(r.session == s for r in recs), missing)
        print(f"{s}: {sum(r.session == s for r in recs)} recordings; without Label.txt/Motion: {missing}")
    jobs = [(r, p) for r in recs for p in r.positions]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(audit_one, r, p): (r, p) for r, p in jobs}
        for i, f in enumerate(futs, 1):
            r, p = futs[f]
            try:
                rows.append(f.result())
                log.info("[%d/%d] %s %s %s ok", i, len(jobs), r.session, r.rec, p)
            except Exception as e:  # noqa: BLE001
                log.exception("[%d/%d] %s %s %s FAILED", i, len(jobs), r.session, r.rec, p)
                print(f"FAILED {r.session} {r.rec} {p}: {e}")
    df = pd.DataFrame(rows).sort_values(["session", "rec", "position"])
    df.to_csv(OUT / "shl_recordings.csv", index=False, float_format="%.4g")

    # Days present in more than one session folder of the same user.
    dup = []
    by_day = df.groupby(["user", "day"])
    for (user, day), g in by_day:
        if g.session.nunique() > 1:
            for rec in g[["session", "rec"]].drop_duplicates().itertuples():
                r = next(x for x in recs if x.session == rec.session and x.rec == rec.rec)
                m = r.path / "Hips_Motion.txt"
                dup.append({"user": user, "day": day, "session": rec.session, "rec": rec.rec,
                            "timemsmin": int(shl.info(r)["timemsmin"]), "hips_motion_bytes": m.stat().st_size,
                            "hips_motion_head_sha1": head_hash(m),
                            "label_bytes": (r.path / "Label.txt").stat().st_size})
    pd.DataFrame(dup).to_csv(OUT / "shl_duplicates.csv", index=False)
    print(f"{len(df)} recording x position rows -> {OUT / 'shl_recordings.csv'}")
    print(f"days in >1 session: {sorted({d['day'] for d in dup})}")


if __name__ == "__main__":
    main()
