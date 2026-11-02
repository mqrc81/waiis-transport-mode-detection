"""Write splits/reserved_test_v3.json (README: stop-point 1 follow-up, item 3; approved 2026-11-02).

    uv run python pick_test.py          # refuses to overwrite an existing reserved_test_v3.json

- ExtraSensory's 12 test users and SHL Users 2-3: copied unchanged from splits/reserved_test_v2.json.
- NOR-TMD: whole device models as groups (linkage guard: the data have no person field), from models without any
  bike installation; >= 5 held-out installations with >= 5 clean windows for bus, subway, train, tram and >= 3 for
  car; tmd.splits.pick_test_groups, 8 models, seed 0, 200,000 draws. Counts from 08's window tables (Android, clean,
  fine classes), as in the table approved at stop-point 1.
"""
import json
from pathlib import Path

import pandas as pd

from tmd.paths import PROJECT
from tmd.splits import pick_test_groups

from dry_run import candidate

HERE = Path(__file__).resolve().parent
OUT = PROJECT / "splits" / "reserved_test_v3.json"
SEED, N_MODELS, N_ITER = 0, 8, 200000
CLASSES = ["bike", "bus", "car", "subway", "train", "tram"]
MIN_USERS = {"bus": 5, "subway": 5, "train": 5, "tram": 5, "car": 3}
MIN_WINDOWS = 5


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; the reserved set is fixed once")
    w = candidate("nor_tmd")
    U = w.pivot_table(index="user", columns="fine", values="ts", aggfunc="size", fill_value=0).reindex(
        columns=CLASSES, fill_value=0)
    model = pd.read_csv(HERE / "results" / "tables" / "nor_tmd_installations.csv", index_col=0).model
    U["model"] = model.reindex(U.index)
    members = U.groupby("model").apply(lambda x: list(x.index))
    G = U.groupby("model")[CLASSES].sum()
    cand = G.index[G.bike == 0]

    def valid(sel):
        us = [u for g in sel for u in members[g]]
        return all((U.loc[us, c] >= MIN_WINDOWS).sum() >= k for c, k in MIN_USERS.items())

    sel = pick_test_groups(G, N_MODELS, valid, seed=SEED, n_iter=N_ITER, candidates=cand)
    test = sorted(u for g in sel for u in members[g])
    v2 = json.loads((PROJECT / "splits" / "reserved_test_v2.json").read_text())
    t, d = U.loc[test], U.drop(index=test)
    summary = {c: {"held_out_users_ge5": int((t[c] >= MIN_WINDOWS).sum()), "left_users_ge5": int((d[c] >= MIN_WINDOWS).sum()),
                   "held_out_windows": int(t[c].sum()), "left_windows": int(d[c].sum())} for c in CLASSES}
    out = {
        "description": "APPROVED 2026-11-02 (experiment 09). ExtraSensory's 12 test users and SHL Users 2-3 unchanged "
                       "from reserved_test_v2.json; NOR-TMD test installations added (whole device models, scored on "
                       "bus, subway, train, tram, car). The reserved set of waiis_bench_v2.",
        "source": "experiments/09_frozen-benchmark_es-shl-nor-collecty/pick_test.py (method, seed, constraints); "
                  "README, stop-point 1 follow-up, item 3",
        "shl": v2["shl"],
        "extrasensory": v2["extrasensory"],
        "nor_tmd": [{"user": u, "model": U.loc[u, "model"]} for u in test],
        "nor_tmd_scored_classes": ["bus", "subway", "train", "tram", "car"],
        "nor_tmd_pick": {"seed": SEED, "models": N_MODELS, "draws": N_ITER, "min_users": MIN_USERS,
                         "min_windows_per_user": MIN_WINDOWS, "models_picked": sorted(sel), "counts": summary},
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{OUT}: {len(test)} NOR-TMD installations from {len(sel)} models; {summary}")


if __name__ == "__main__":
    main()
