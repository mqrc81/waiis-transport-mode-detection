"""Step 1: proposed splits (for approval, not yet fixed).

Usage: uv run python propose_splits.py [--seed 0]
Reads results/audit/{shl_recordings,shl_duplicates,es_users}.csv.
Writes results/audit/proposed_{shl_day_folds,es_users}.csv and prints a short summary.
"""
import argparse
from pathlib import Path

import pandas as pd

from tmd.labels import CLASSES
from tmd.splits import balanced_folds, pick_test_groups

HERE = Path(__file__).resolve().parent
AUD = HERE / "results" / "audit"
K = 5
N_TEST = 12
MIN_MIN = 5          # a user "has" a class with >= 5 usable minutes
MIN_USERS = 4        # test users per class for bus, car, bike


def shl_folds(seed: int) -> pd.DataFrame:
    rec = pd.read_csv(AUD / "shl_recordings.csv", dtype={"rec": str, "day": str})
    u1 = rec[rec.user == "User1"]
    # A day present in User1 and User1_4phones is kept once, from User1_4phones (all 4 positions).
    dup_days = set(u1[u1.session == "User1_4phones"].day)
    u1 = u1[~((u1.session == "User1") & u1.day.isin(dup_days))]
    hips = u1[u1.position == "Hips"]                          # balance on the hips phone, one per day
    h = hips.groupby("day")[[f"h_{c}" for c in CLASSES]].sum()
    folds = balanced_folds(h, K, seed=seed)
    out = u1[["session", "rec", "day", "position"]].assign(fold=u1.day.map(folds))
    out.to_csv(AUD / "proposed_shl_day_folds.csv", index=False)
    per = h.groupby(folds).sum().round(1)
    per.columns = CLASSES
    per.insert(0, "days", folds.value_counts().sort_index())
    print("SHL User 1 day folds (hips hours per class):")
    print(per.to_string())
    return out


def es_split(seed: int) -> pd.DataFrame:
    u = pd.read_csv(AUD / "es_users.csv", index_col="uuid")
    n = u[[f"n_{c}" for c in CLASSES]]
    n.columns = CLASSES
    train_users = set(pd.read_csv(AUD / "es_train_minutes.csv", index_col="uuid").query("mapped_train > 0").index)
    has_gyro = u.index[u.acc_and_gyro_ok > 0]
    n_ios_all = (u.platform == "iphone").sum()
    ios_lo, ios_hi = int(N_TEST * n_ios_all / len(u)), -(-N_TEST * n_ios_all // len(u))

    def valid(sel):
        s = n.loc[sel]
        ok = all((s[c] >= MIN_MIN).sum() >= MIN_USERS for c in ["bus", "car", "bike"])
        ios = (u.loc[sel, "platform"] == "iphone").sum()
        ok &= ios_lo <= ios <= ios_hi
        if len(train_users) >= 2:
            ok &= 1 <= len(train_users & set(sel)) < len(train_users)
        # dev keeps >= 50% of every class
        ok &= bool((n.drop(index=sel).sum() >= 0.5 * n.sum()).all())
        return ok

    test = pick_test_groups(n, N_TEST, valid, seed=seed, candidates=has_gyro)
    dev = n.index.difference(test)
    folds = balanced_folds(n.loc[dev], K, seed=seed, strata=u.loc[dev, "platform"])
    out = u[["platform"]].assign(partition="dev", fold=-1)
    out.loc[test, "partition"] = "test"
    out.loc[dev, "fold"] = folds
    out = out.join(n)
    out.to_csv(AUD / "proposed_es_users.csv")
    print(f"\nExtraSensory: {len(test)} test / {len(dev)} dev users; train users (ON_A_TRAIN mapped): {sorted(train_users)}")
    print(out.groupby(["partition", "fold"]).agg(users=("platform", "size"),
                                                  iphone=("platform", lambda p: int((p == "iphone").sum())),
                                                  **{c: (c, "sum") for c in CLASSES}).to_string())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    shl_folds(a.seed)
    es_split(a.seed)


if __name__ == "__main__":
    main()
