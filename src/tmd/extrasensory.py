"""ExtraSensory readers and the minute -> class mapping (brief_block1_experiments.md, Labels).

    per_uuid_features_labels/<uuid>.features_labels.csv/<uuid>.features_labels.csv   cleaned labels
    per_uuid_original_labels/<uuid>.original_labels.csv.gz                            original labels
    raw_acc/<uuid>/<timestamp>.m_raw_acc.dat, proc_gyro/..., raw_magnet/...           one snippet per minute
    per_uuid_absolute_location/<uuid>.absolute_locations.csv                          lat / lon per minute
    cv_5_folds/fold_*_{train,test}_{iphone,android}_uuids.txt                         platform per user

The motion summary features in the cleaned CSV are never read: only timestamp, label:*, label_source and (for
experiment 04's GPS) location:* / location_quick_features:*.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from tmd.labels import CLASS_ID
from tmd.paths import ES_RAW

SENSORS = {"acc": "raw_acc", "gyro": "proc_gyro", "mag": "raw_magnet"}

# Priority order of the brief: the first class whose label is set wins. Keys are column names
# without prefix; 'orig:' means the original (uncleaned) label file, otherwise the cleaned CSV.
PRIORITY = [
    ("train", ["orig:ON_A_TRAIN"]),
    ("bus", ["ON_A_BUS"]),
    ("car", ["IN_A_CAR", "DRIVE_-_I_M_THE_DRIVER", "DRIVE_-_I_M_A_PASSENGER"]),   # passenger: see PASSENGER
    ("bike", ["BICYCLING"]),
    ("run", ["FIX_running"]),
    ("walk", ["FIX_walking", "STAIRS_-_GOING_UP", "STAIRS_-_GOING_DOWN"]),
    ("still", ["SITTING", "OR_standing", "LYING_DOWN", "SLEEPING"]),
]
MOVING = ["train", "bus", "car", "bike", "run", "walk"]   # two of these in one minute = conflict
# "I'm a passenger" doesn't name the vehicle: with bus or train it is that vehicle, not a car
# (approved 2026-10-25). Only IN_A_CAR or I'M_THE_DRIVER with bus/train is a conflict.
PASSENGER = "DRIVE_-_I_M_A_PASSENGER"
# Minutes with any of these (original labels) are dropped: ELEVATOR (brief) and transport or
# locomotion labels that are not clearly one of our classes (approved in experiment 02's audit).
DROP = ["ELEVATOR", "ON_A_PLANE", "ON_A_BOAT", "RIDING_AN_ANIMAL", "SKATEBOARDING",
        "STATIONARY_BIKE", "TREADMILL", "ELLIPTICAL_MACHINE", "WHEELCHAIR", "USING_CRUTCHES"]
# Kept out of training but labelled, as a separate check of what models call them (reason 'check:<label>').
CHECK = ["MOTORBIKE"]
POSITION_LABELS = ["PHONE_IN_POCKET", "PHONE_IN_HAND", "PHONE_IN_BAG", "PHONE_ON_TABLE"]
# Phone not carried: on a table or away from the user (02 label review, 2026-10-25). For walk / run / bike the
# label then describes the person, not the phone, so those minutes are dropped ('drop:not_carried'). Vehicles keep
# them (the phone still rides along). In still they are capped per user, see `cap_resting_still`.
RESTING_LABELS = ["PHONE_ON_TABLE", "orig:PHONE_AWAY_FROM_ME"]
FOOT_BIKE = ["walk", "run", "bike"]
RESTING_STILL_MAX_SHARE = 0.2


def uuids() -> list[str]:
    return sorted(p.name.split(".")[0] for p in (ES_RAW / "per_uuid_features_labels").glob("*.features_labels.csv"))


def platforms() -> dict[str, str]:
    """uuid -> 'iphone' | 'android', from the dataset's own cv_5_folds test lists."""
    out = {}
    for f in (ES_RAW / "cv_5_folds").glob("fold_*_test_*_uuids.txt"):
        plat = f.name.split("_")[3]
        out.update({u.strip(): plat for u in f.read_text().split() if u.strip()})
    return out


def _unzipped(p: Path) -> Path:
    return p / p.name if p.is_dir() else p      # the local unzip left one folder per file


def _cleaned_path(uuid: str) -> Path:
    return _unzipped(ES_RAW / "per_uuid_features_labels" / f"{uuid}.features_labels.csv")


def read_cleaned_labels(uuid: str) -> pd.DataFrame:
    """Index timestamp [s]; columns label names without prefix (1.0 / 0.0 / NaN = not reported) + label_source."""
    df = pd.read_csv(_cleaned_path(uuid), usecols=lambda c: c == "timestamp" or c.startswith("label"))
    df = df.set_index("timestamp")
    return df.rename(columns=lambda c: c.removeprefix("label:"))


def read_original_labels(uuid: str) -> pd.DataFrame:
    """Index timestamp [s]; columns original label names without prefix + label_source."""
    with gzip.open(ES_RAW / "per_uuid_original_labels" / f"{uuid}.original_labels.csv.gz") as f:
        df = pd.read_csv(f).set_index("timestamp")
    return df.rename(columns=lambda c: c.removeprefix("original_label:"))


def minute_classes(clean: pd.DataFrame, orig: pd.DataFrame) -> pd.DataFrame:
    """One row per minute: class (name or ''), reason ('ok', 'unlabelled', 'conflict', 'drop:<label>',
    'check:<label>'),
    the set of moving classes reported, and the phone position labels.

    A NaN label means "not reported" and never counts as positive.
    """
    orig = orig.reindex(clean.index)

    def pos(col: str) -> np.ndarray:
        src, name = (orig, col[5:]) if col.startswith("orig:") else (clean, col)
        if name not in src:
            return np.zeros(len(clean), bool)
        return (src[name].to_numpy() == 1)

    hit = {cls: np.any([pos(c) for c in cols], axis=0) for cls, cols in PRIORITY}
    car_strict = pos("IN_A_CAR") | pos("DRIVE_-_I_M_THE_DRIVER")
    hit["car"] = car_strict | (pos(PASSENGER) & ~hit["bus"] & ~hit["train"])
    n_moving = np.sum([hit[c] for c in MOVING], axis=0)
    drop = {d: pos("orig:" + d) for d in DROP}

    cls = np.full(len(clean), "", dtype=object)
    for c, _ in reversed(PRIORITY):          # higher priority overwrites
        cls[hit[c]] = c
    reason = np.where(cls == "", "unlabelled", "ok").astype(object)
    reason[n_moving > 1] = "conflict"
    for d, m in drop.items():
        reason[m] = "drop:" + d
    resting = np.any([pos(c) for c in RESTING_LABELS], axis=0)
    ok = reason == "ok"
    reason[ok & (cls == "still") & pos("SLEEPING")] = "drop:SLEEPING"          # phone on the nightstand
    reason[ok & np.isin(cls, FOOT_BIKE) & resting] = "drop:not_carried"
    for d in CHECK:
        reason[pos("orig:" + d)] = "check:" + d
    out = pd.DataFrame({"cls": cls, "reason": reason, "label_source": clean["label_source"].to_numpy()},
                       index=clean.index)
    out["resting"] = resting
    out["moving"] = ["+".join(c for c in MOVING if hit[c][i]) for i in range(len(clean))]
    for p in POSITION_LABELS:
        out[p] = pos(p) | pos("orig:" + p)
    out["position"] = [next((p.removeprefix("PHONE_").lower() for p in POSITION_LABELS if out[p].iat[i]), "")
                       for i in range(len(out))]
    out["y"] = [CLASS_ID.get(c, 0) if r == "ok" else 0 for c, r in zip(cls, reason)]
    return out


def cap_resting_still(minutes: pd.DataFrame, max_share: float = RESTING_STILL_MAX_SHARE,
                      seed: int = 0) -> np.ndarray:
    """Keep-mask over minutes (columns user, ts, cls, resting): per user, resting-phone still minutes are subsampled
    so that they are at most `max_share` of that user's still minutes. Other minutes are always kept."""
    keep = np.ones(len(minutes), bool)
    rng = np.random.default_rng(seed)
    still = (minutes.cls == "still").to_numpy()
    resting = minutes.resting.to_numpy(bool)
    users = minutes.user.to_numpy()
    for u in np.unique(users[still]):
        mine = still & (users == u)
        r = np.flatnonzero(mine & resting)
        n_keep = min(len(r), int(max_share / (1 - max_share) * (mine & ~resting).sum()))
        keep[r] = False
        if n_keep:
            keep[rng.choice(r, n_keep, replace=False)] = True
    return keep


def snippet_path(uuid: str, ts: int, sensor: str = "acc") -> Path:
    s = SENSORS[sensor]
    return ES_RAW / s / uuid / f"{ts}.m_{s}.dat"


def read_snippet(path: Path) -> np.ndarray | None:
    """(n, 4) array: time [s, device clock], x, y, z. None if the file is missing, empty or all-NaN."""
    if not path.exists():
        return None
    a = np.fromstring(path.read_text(), sep=" ")
    if a.size == 0 or a.size % 4:
        return None
    a = a.reshape(-1, 4)
    return None if np.isnan(a[:, 1:]).all() else a


def minutes_table(users) -> pd.DataFrame:
    """minute_classes for several users, with columns user and ts (one row per user-minute)."""
    parts = []
    for u in users:
        m = minute_classes(read_cleaned_labels(u), read_original_labels(u))
        parts.append(m.rename_axis("ts").reset_index().assign(user=u))
    out = pd.concat(parts, ignore_index=True)
    out["ts"] = out.ts.astype(int)
    return out


# Website: lat/lon is NaN when the minute had no location update or all updates were worse than 200 m.
MAX_ACCURACY_M = 200.0


def read_location(uuid: str) -> pd.DataFrame:
    """Index timestamp [s]; all location:* and location_quick_features:* columns of the cleaned CSV (names as in the
    CSV) plus latitude, longitude from the absolute-location file (NaN = no coordinates)."""
    loc = pd.read_csv(_cleaned_path(uuid), usecols=lambda c: c == "timestamp" or c.startswith("location"))
    ab = pd.read_csv(_unzipped(ES_RAW / "per_uuid_absolute_location" / f"{uuid}.absolute_locations.csv"))
    return loc.merge(ab, on="timestamp", how="left", validate="one_to_one").set_index("timestamp")


def location_missing_reason(loc: pd.DataFrame) -> np.ndarray:
    """Per minute: '' (coordinates present), 'no_updates' (num_valid_updates 0 / NaN), 'poor_accuracy' (updates, no
    coordinates, best accuracy > 200 m), 'withheld' (updates within 200 m, i.e. relative location present, but no
    absolute coordinates), 'unknown'. Location only updates on a significant change, so 'no_updates' also covers a
    phone that did not move; nothing is imputed."""
    has = loc.latitude.notna().to_numpy() & loc.longitude.notna().to_numpy()
    n = loc["location:num_valid_updates"].fillna(0).to_numpy()
    acc = loc["location:best_horizontal_accuracy"].to_numpy()
    out = np.full(len(loc), "unknown", dtype=object)
    out[n <= 0] = "no_updates"
    out[(n > 0) & (acc > MAX_ACCURACY_M)] = "poor_accuracy"
    out[(n > 0) & (acc <= MAX_ACCURACY_M)] = "withheld"
    out[has] = ""
    return out
