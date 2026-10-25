"""Step 1 (ExtraSensory): data audit, one row per user-minute.

Usage: uv run python audit_es.py [--workers 6]
Writes <data>/processed/es_minutes_audit.parquet (per minute, not in git) and in results/audit/:
es_users.csv (per user), es_original_labels.csv (other transport labels), es_train_minutes.csv.
Full log in logs/audit_es.log.
"""
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from tmd import extrasensory as es
from tmd.labels import CLASSES
from tmd.paths import PROCESSED

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "audit"
# Original labels scanned for transport / locomotion meaning not covered by the priority list.
SCAN = es.DROP + es.CHECK + ["ON_A_TRAIN", "ON_A_BUS", "IN_A_CAR", "STROLLING", "HIKING", "STANDING_AND_MOVING",
                  "STANDING_IN_PLACE", "WALKING", "RUNNING", "BICYCLING", "LIMPING", "JUMPING",
                  "PHONE_AWAY_FROM_ME", "PHONE_STRAPPED"]


def snippet_stats(a: np.ndarray | None, prefix: str) -> dict:
    if a is None:
        return {f"{prefix}_n": 0}
    t, v = a[:, 0], a[:, 1:]
    good = ~np.isnan(v).any(1)
    dt = np.diff(t)
    return {
        f"{prefix}_n": len(a), f"{prefix}_nan": int((~good).sum()),
        f"{prefix}_dur_s": float(t[-1] - t[0]),
        f"{prefix}_dt_med_ms": float(np.median(dt) * 1e3), f"{prefix}_dt_p95_ms": float(np.percentile(dt, 95) * 1e3),
        f"{prefix}_dt_max_ms": float(dt.max() * 1e3), f"{prefix}_dt_nonpos": int((dt <= 0).sum()),
        f"{prefix}_repeat": float((np.diff(v[good], axis=0) == 0).all(1).mean()) if good.sum() > 1 else np.nan,
        f"{prefix}_norm_med": float(np.median(np.linalg.norm(v[good], axis=1))) if good.any() else np.nan,
    }


def audit_user(uuid: str) -> tuple[pd.DataFrame, dict]:
    clean, orig = es.read_cleaned_labels(uuid), es.read_original_labels(uuid)
    m = es.minute_classes(clean, orig)
    o = orig.reindex(clean.index)
    for lab in SCAN:
        m[f"orig_{lab}"] = (o[lab].to_numpy() == 1) if lab in o else False
    # What the cleaned CSV says on ON_A_TRAIN minutes (cleaning dropped the label).
    still_cols = [c for c in ["SITTING", "OR_standing", "LYING_DOWN", "SLEEPING"] if c in clean]
    m["clean_still"] = (clean[still_cols].to_numpy() == 1).any(1)
    stats = []
    for ts in clean.index.astype(int):
        row = {}
        for s, pre in [("acc", "acc"), ("gyro", "gyro")]:
            p = es.snippet_path(uuid, ts, s)
            row[f"{pre}_file"] = p.exists()
            row.update(snippet_stats(es.read_snippet(p), pre))
        stats.append(row)
    m = pd.concat([m, pd.DataFrame(stats, index=m.index)], axis=1)
    m.insert(0, "uuid", uuid)
    extra = {}
    for s in ["acc", "gyro"]:
        d = es.ES_RAW / es.SENSORS[s] / uuid
        files = {int(f.name.split(".")[0]) for f in d.glob("*.dat")} if d.exists() else set()
        extra[f"{s}_dir"] = d.exists()
        extra[f"{s}_files_without_csv_row"] = len(files - set(clean.index.astype(int)))
    extra["orig_rows_not_in_clean"] = len(set(orig.index) - set(clean.index))
    return m.reset_index(), extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=HERE / "logs" / "audit_es.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    log = logging.getLogger()
    plat = es.platforms()
    users = es.uuids()
    parts, extras = [], {}
    with ProcessPoolExecutor(a.workers) as ex:
        for u, (m, extra) in zip(users, ex.map(audit_user, users)):
            parts.append(m)
            extras[u] = extra
            log.info("%s: %d minutes %s", u, len(m), extra)
    mins = pd.concat(parts, ignore_index=True)
    mins["platform"] = mins.uuid.map(plat)
    mins.to_parquet(PROCESSED / "es_minutes_audit.parquet")

    # Per user: platform, minutes, raw availability, rates, units, minutes per class.
    g = mins.groupby("uuid")
    acc_ok = mins.acc_n > 0
    gyro_ok = mins.gyro_n > 0
    users_df = pd.DataFrame({
        "platform": g.platform.first(),
        "minutes": g.size(),
        "acc_dir": pd.Series({u: e["acc_dir"] for u, e in extras.items()}),
        "gyro_dir": pd.Series({u: e["gyro_dir"] for u, e in extras.items()}),
        "acc_file_missing": (~mins.acc_file).groupby(mins.uuid).sum(),
        "acc_file_unusable": (mins.acc_file & ~acc_ok).groupby(mins.uuid).sum(),
        "gyro_file_missing": (~mins.gyro_file).groupby(mins.uuid).sum(),
        "gyro_file_unusable": (mins.gyro_file & ~gyro_ok).groupby(mins.uuid).sum(),
        "acc_and_gyro_ok": (acc_ok & gyro_ok).groupby(mins.uuid).sum(),
        "acc_files_without_row": pd.Series({u: e["acc_files_without_csv_row"] for u, e in extras.items()}),
        "acc_n_med": mins[acc_ok].groupby("uuid").acc_n.median(),
        "acc_dur_med_s": mins[acc_ok].groupby("uuid").acc_dur_s.median(),
        "acc_dt_med_ms": mins[acc_ok].groupby("uuid").acc_dt_med_ms.median(),
        "acc_dt_p95_ms": mins[acc_ok].groupby("uuid").acc_dt_p95_ms.median(),
        "acc_dt_max_ms_p99": mins[acc_ok].groupby("uuid").acc_dt_max_ms.quantile(0.99),
        "acc_dt_med_ms_p10": mins[acc_ok].groupby("uuid").acc_dt_med_ms.quantile(0.1),
        "acc_dt_med_ms_p90": mins[acc_ok].groupby("uuid").acc_dt_med_ms.quantile(0.9),
        "acc_repeat_med": mins[acc_ok].groupby("uuid").acc_repeat.median(),
        "frac_acc_in_g": (mins[acc_ok].acc_norm_med < 3).groupby(mins[acc_ok].uuid).mean(),
        "gyro_dt_med_ms": mins[gyro_ok].groupby("uuid").gyro_dt_med_ms.median(),
        "acc_norm_still_med": mins[acc_ok & (mins.cls == "still")].groupby("uuid").acc_norm_med.median(),
        "acc_norm_all_med": mins[acc_ok].groupby("uuid").acc_norm_med.median(),
        "gyro_norm_still_med": mins[gyro_ok & (mins.cls == "still")].groupby("uuid").gyro_norm_med.median(),
    })
    usable = mins[(mins.reason == "ok") & acc_ok & gyro_ok]
    cnt = usable.groupby(["uuid", "cls"]).size().unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0)
    users_df = users_df.join(cnt.add_prefix("n_")).fillna({f"n_{c}": 0 for c in CLASSES})
    lab = mins[mins.reason == "ok"].groupby(["uuid", "cls"]).size().unstack(fill_value=0)
    users_df = users_df.join(lab.reindex(columns=CLASSES, fill_value=0).add_prefix("lab_")).fillna(0)
    users_df.index.name = "uuid"
    users_df.to_csv(OUT / "es_users.csv", float_format="%.4g")

    # Original labels outside the priority list: minutes and users, and what our mapping gave them.
    rows = []
    for lab in SCAN:
        sel = mins[mins[f"orig_{lab}"]]
        rows.append({"label": lab, "minutes": len(sel), "users": sel.uuid.nunique(),
                     "in_drop_list": lab in es.DROP,
                     "mapped_to": "; ".join(f"{k or '-'}:{v}" for k, v in sel.cls.value_counts().items()),
                     "reason": "; ".join(f"{k}:{v}" for k, v in sel.reason.value_counts().items())})
    pd.DataFrame(rows).to_csv(OUT / "es_original_labels.csv", index=False)

    tr = mins[mins.orig_ON_A_TRAIN]
    tr.groupby("uuid").agg(minutes=("cls", "size"), clean_still=("clean_still", "sum"),
                           mapped_train=("y", lambda y: int((y == 7).sum())),
                           acc_gyro_ok=("gyro_n", lambda s: int(((s > 0) & (tr.loc[s.index, "acc_n"] > 0)).sum())),
                           platform=("platform", "first")).to_csv(OUT / "es_train_minutes.csv")

    print(f"{len(users)} users, {len(mins)} minutes; reasons {mins.reason.value_counts().to_dict()}")
    print(f"usable (class ok + acc + gyro): {len(usable)}; per class {usable.cls.value_counts().to_dict()}")
    print(f"ON_A_TRAIN minutes {len(tr)} in {tr.uuid.nunique()} users")


if __name__ == "__main__":
    main()
