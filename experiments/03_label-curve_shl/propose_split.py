"""Stop-point 1: proposed day split of SHL User 1 into training / validation / test (60 / 20 / 20 % of hips hours).

Usage: uv run python propose_split.py [--seed 0]
Reads 02's cached window table (<data>/processed/shl_u1_25hz_w5s_s2.5s_feat-v1_all.parquet, User 1 only) and
writes splits/shl_u1_days.csv (day, partition) plus results/split_{hours,days}.csv. Prints a short summary.

Method: tmd.splits.pick_test_groups twice on whole calendar days (hips hours per class): first the test days, then
the validation days from the rest, each N_DAYS days with per-class hours closest to 20 % of the total. Constraints:
every class on >= MIN_DAYS days (> 3 min each) in the picked set; test holds at least one User1_4phones day (the
other-position read-out). Days without labelled hips windows are not split.
"""
import argparse
from pathlib import Path

import pandas as pd

from tmd import shl
from tmd.labels import CLASSES
from tmd.paths import PROCESSED
from tmd.splits import pick_test_groups

HERE = Path(__file__).resolve().parent
TABLE = PROCESSED / "shl_u1_25hz_w5s_s2.5s_feat-v1_all.parquet"
STRIDE_H = 2.5 / 3600          # one window per 2.5 s step -> labelled hours
MIN_DAY_H = 3 / 60             # a day "has" a class with > 3 min (as in 02)
N_DAYS = 15                    # of 74 days with labels: ~20 %
MIN_DAYS = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    w = pd.read_parquet(TABLE, columns=["user", "day", "position", "y"])
    assert set(w.user) == {"User1"}
    h = w[(w.position == "Hips") & (w.y > 0)]
    hours = h.groupby(["day", "y"]).size().unstack(fill_value=0).reindex(columns=range(1, 9), fill_value=0)
    hours = hours * STRIDE_H
    hours.columns = CLASSES
    four = {r.day for r in shl.recordings("User1_4phones")}
    has = hours > MIN_DAY_H

    def spread(sel):
        return bool((has.loc[sel].sum() >= MIN_DAYS).all())

    test = pick_test_groups(hours, N_DAYS, lambda sel: spread(sel) and bool(four & set(sel)), seed=a.seed)
    rest = hours.drop(index=test)
    # target share of the rest = N_DAYS / len(rest), i.e. ~20 % of the total
    val = pick_test_groups(rest, N_DAYS, spread, seed=a.seed)
    part = pd.Series("training", index=hours.index, name="partition")
    part[test], part[val] = "test", "validation"
    split = pd.DataFrame({"user": "User1", "day": part.index, "partition": part.values,
                          "four_phones": part.index.isin(four)}).sort_values("day")
    split.to_csv(HERE / "splits" / "shl_u1_days.csv", index=False)

    order = ["training", "validation", "test"]
    hp = hours.groupby(part).sum().loc[order]
    hp.insert(0, "days", part.value_counts().loc[order])
    hp["total"] = hp[CLASSES].sum(axis=1)
    share = (hp[CLASSES + ["total"]] / hp[CLASSES + ["total"]].sum()).round(3)
    dp = has.groupby(part).sum().loc[order]
    hp.round(2).to_csv(HERE / "results" / "split_hours.csv")
    dp.to_csv(HERE / "results" / "split_days.csv")
    print(f"seed {a.seed}; 4-phone days: "
          f"{split[split.four_phones][['day', 'partition']].values.tolist()}")
    print("hips hours\n" + hp.round(1).to_string())
    print("share of hours\n" + share.to_string())
    print(f"days with > 3 min\n{dp.to_string()}")


if __name__ == "__main__":
    main()
