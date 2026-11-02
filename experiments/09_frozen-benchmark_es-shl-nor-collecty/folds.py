"""Write splits/folds_nor_tmd_collecty.csv: NOR-TMD and Collecty dev users in the 5 dev folds (approved method,
README stop-point 1 answers, item 4).

    uv run python folds.py          # refuses to overwrite

Per dataset, the dev users (NOR-TMD: Android installations not in splits/reserved_test_v3.json; Collecty: all
users) with >= 5 clean windows of a class count as "having" it; tmd.splits.balanced_folds deals them into 5 folds
balancing users per class (seed 0). Counts from 08's window tables, as in the dry run. Users without any clean
labelled window are not listed; the build gives them fold -1 (never scored).
"""
import json
from pathlib import Path

import pandas as pd

from tmd.paths import PROJECT
from tmd.splits import balanced_folds

from dry_run import MIN_USER_WINDOWS, candidate

HERE = Path(__file__).resolve().parent
OUT = HERE / "splits" / "folds_nor_tmd_collecty.csv"
SEED, K = 0, 5


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; folds are fixed once")
    test = {r["user"] for r in json.loads((PROJECT / "splits" / "reserved_test_v3.json").read_text())["nor_tmd"]}
    rows, per = [], []
    for ds in ("nor_tmd", "collecty"):
        w = candidate(ds)
        U = w.pivot_table(index="user", columns="fine", values="ts", aggfunc="size", fill_value=0)
        U = U.drop(index=[u for u in U.index if u in test])
        has = (U >= MIN_USER_WINDOWS).astype(int)
        has = has.loc[:, has.sum() > 0]
        f = balanced_folds(has, K, seed=SEED, n_iter=20000)
        rows.append(pd.DataFrame({"dataset": ds, "user": f.index.astype(str), "fold": f.to_numpy()}))
        p = has.groupby(f).sum()
        p.insert(0, "dataset", ds)
        per.append(p.rename_axis("fold").reset_index())
    pd.concat(rows).to_csv(OUT, index=False)
    t = pd.concat(per)
    t.to_csv(HERE / "results" / "tables" / "folds_users_per_class.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
