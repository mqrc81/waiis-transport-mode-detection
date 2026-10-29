"""Re-pick of the 12 ExtraSensory test users (approved by the supervisor on 2026-10-28).

The approved pick was computed on the development build of commit b1e2a9c (decision register, History);
rerunning it on waiis_bench_v1 gives different counts, and the script never overwrites the approved file.

Usage: uv run python propose_test_v2.py [--seed 0] [--draws 3000000]
Reads the official examples (tmd.bench.examples). Writes
- splits/reserved_test_v2.json (project root), unless it is already approved; SHL Users 2-3 unchanged;
- splits/es_users_v2.csv (this folder): uuid, platform, partition, fold (dev folds rebuilt from the other users);
- results/tables/test_pick_{old_vs_new,users,dev_folds}.csv.
The old files (splits/reserved_test.json, 02's es_users.csv) are not touched. Approved 2026-10-28: the new files are
now the reference, and this script no longer overwrites an approved reserved_test_v2.json.

Rule (Marc, 2026-10-28):
- eligible: every user except those with implausible accelerometer data (> MAX_IMPLAUSIBLE of their windows);
- 6-7 iPhone users; iPhone share of test examples and location coverage of test examples within TOL of the
  overall figures (eligible users);
- users with >= MIN_EX examples of a class: bus >= 5, bike >= 4, run >= 3, train 1-2, still / walk / car >= 5;
- among the subsets that pass, the one whose per-class test share is closest to 12/59 wins (sum of squared
  deviations); fixed seed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tmd import bench
from tmd.labels import CLASSES
from tmd.paths import PROJECT
from tmd.splits import balanced_folds, reserved_test

HERE = Path(__file__).resolve().parent
RES = HERE / "results" / "tables"
VERSION = bench.VERSION
N_TEST, K = 12, 5
MIN_EX = 5                       # a user "has" a class with >= 5 examples
MAX_IMPLAUSIBLE = 0.10           # users with more implausible windows are not eligible
TOL = 0.03                       # iPhone share and location coverage of test examples vs overall
MIN_USERS = {"still": 5, "walk": 5, "car": 5, "bus": 5, "bike": 4, "run": 3, "train": 1}
MAX_USERS = {"train": 2}
C7 = [c for c in CLASSES if c != "subway"]


def user_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    lab = bench.read("extrasensory", "labels", columns=["user", "ts", "platform", "has_acc", "acc_implausible"],
                     version=VERSION)
    win = lab[lab.has_acc]
    imp = win.groupby("user").acc_implausible.mean()
    ex = bench.examples("extrasensory", columns=["platform", "cls", "has_location", "split"], version=VERSION)
    n = ex.groupby(["user", "cls"]).size().unstack(fill_value=0).reindex(columns=C7, fill_value=0)
    u = pd.DataFrame({"platform": lab.groupby("user").platform.first(), "implausible_share": imp,
                      "examples": ex.groupby("user").size(), "with_location": ex.groupby("user").has_location.sum(),
                      "old_split": ex.groupby("user").split.first()}).join(n).fillna(0)
    return u, ex


def search(u: pd.DataFrame, seed: int, draws: int) -> tuple[pd.Index, int]:
    el = u[u.implausible_share <= MAX_IMPLAUSIBLE]
    c = el[C7].to_numpy(float)
    has = (c >= MIN_EX)
    ios = (el.platform == "iphone").to_numpy()
    exn = el.examples.to_numpy(float)
    locn = el.with_location.to_numpy(float)
    ios_share = (exn * ios).sum() / exn.sum()
    loc_share = locn.sum() / exn.sum()
    tot = c.sum(0)
    target = N_TEST / len(el)
    n_ios = ios.sum() * N_TEST / len(el)
    lo_ios, hi_ios = int(np.floor(n_ios)), int(np.ceil(n_ios))
    mins = np.array([MIN_USERS.get(k, 0) for k in C7])
    maxs = np.array([MAX_USERS.get(k, N_TEST) for k in C7])
    rng = np.random.default_rng(seed)
    best, best_score, n_valid = None, np.inf, 0
    for _ in range(draws // 100_000):
        sel = np.argsort(rng.random((100_000, len(el))), axis=1)[:, :N_TEST]
        S = np.zeros((len(sel), len(el)), bool)
        np.put_along_axis(S, sel, True, axis=1)
        users = S @ has.astype(int)
        e = S @ exn
        ok = (users >= mins).all(1) & (users <= maxs).all(1)
        k_ios = S @ ios.astype(int)
        ok &= (k_ios >= lo_ios) & (k_ios <= hi_ios)
        ok &= np.abs((S @ (exn * ios)) / e - ios_share) <= TOL
        ok &= np.abs((S @ locn) / e - loc_share) <= TOL
        n_valid += int(ok.sum())
        if not ok.any():
            continue
        share = (S[ok] @ c) / tot
        score = ((share - target) ** 2).sum(1)
        i = score.argmin()
        if score[i] < best_score:
            best_score, best = score[i], S[ok][i]
    if best is None:
        raise ValueError("no subset satisfies the constraints")
    return el.index[best].sort_values(), n_valid


def describe(u: pd.DataFrame, ex: pd.DataFrame, test: pd.Index, name: str) -> pd.Series:
    t = ex[ex.user.isin(test)]
    out = {"users": len(test), "iphone_users": int((u.loc[test, "platform"] == "iphone").sum()),
           "examples": len(t), "iphone_share_examples": round((t.platform == "iphone").mean(), 3),
           "location_coverage": round(t.has_location.mean(), 3)}
    for c in C7:
        out[f"{c}: examples"] = int((t.cls == c).sum())
        out[f"{c}: share of class"] = round((t.cls == c).sum() / (ex.cls == c).sum(), 3)
        out[f"{c}: users >= {MIN_EX}"] = int((u.loc[test, c] >= MIN_EX).sum())
    return pd.Series(out, name=name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=3_000_000)
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    u, ex = user_table()
    ineligible = u.index[u.implausible_share > MAX_IMPLAUSIBLE]
    print(f"ineligible (> {MAX_IMPLAUSIBLE:.0%} implausible windows): "
          f"{[(x[:8], round(u.implausible_share[x], 2)) for x in ineligible]}")
    old = pd.Index(sorted(reserved_test(1)["extrasensory"].uuid))
    new, n_valid = search(u, a.seed, a.draws)
    print(f"{a.draws:,} draws, {n_valid:,} valid subsets")
    allu = u.index
    cmp = pd.concat([describe(u, ex, allu, "all 60 users"), describe(u, ex, old, "old test (v1 pick)"),
                     describe(u, ex, new, "new test (v2 proposal)")], axis=1)
    cmp.to_csv(RES / "test_pick_old_vs_new.csv")
    print(cmp.to_string())
    users = u.loc[old.union(new), ["platform", "examples", "with_location"] + C7].assign(
        old=lambda d: d.index.isin(old), new=lambda d: d.index.isin(new))
    users.to_csv(RES / "test_pick_users.csv")
    print(users.rename(index=lambda x: x[:8]).to_string())
    # dev folds from the remaining users, as 02 (balanced_folds, platform strata, seed)
    dev = u.index.difference(new)
    folds = balanced_folds(u.loc[dev, C7], K, seed=a.seed, strata=u.loc[dev, "platform"])
    out = u[["platform"]].assign(partition="dev", fold=-1)
    out.loc[new, "partition"] = "test"
    out.loc[dev, "fold"] = folds
    out = out.rename_axis("uuid").reset_index()
    (HERE / "splits").mkdir(exist_ok=True)
    out.to_csv(HERE / "splits" / "es_users_v2.csv", index=False)
    f = ex[ex.user.isin(dev)].assign(fold=lambda d: d.user.map(folds)).groupby(["fold", "cls"]).size().unstack(
        fill_value=0)[C7]
    f.insert(0, "users", pd.Series(folds).value_counts().sort_index())
    f.to_csv(RES / "test_pick_dev_folds.csv")
    print(f.to_string())
    old_json = json.loads((PROJECT / "splits" / "reserved_test.json").read_text())
    new_json = {
        "description": "PROPOSED 2026-10-28, NOT IN USE: pending the supervisor's approval. Re-pick of the 12 "
                       "ExtraSensory test users (experiment 04). SHL Users 2-3 "
                       "unchanged. The reference stays splits/reserved_test.json until this is approved.",
        "source": "experiments/04_frozen-benchmark_es-shl/propose_test_v2.py --seed "
                  f"{a.seed} --draws {a.draws}; dev folds in experiments/04_frozen-benchmark_es-shl/splits/"
                  "es_users_v2.csv",
        "shl": old_json["shl"],
        "extrasensory": [{"uuid": x, "platform": u.platform[x]} for x in new]}
    p = PROJECT / "splits" / "reserved_test_v2.json"
    if p.exists() and json.loads(p.read_text())["description"].startswith("APPROVED"):
        print(f"{p.name} is approved and not overwritten")
        return
    p.write_text(json.dumps(new_json, indent=1) + "\n")


if __name__ == "__main__":
    main()
