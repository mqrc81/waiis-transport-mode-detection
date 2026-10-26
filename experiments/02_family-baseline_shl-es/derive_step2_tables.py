"""Rebuild the step-2 feature tables (labelled windows only) from the all-window tables of step 3.

Usage: uv run python derive_step2_tables.py [--check]
  shl_u1_<tag>.parquet  = shl_u1_<tag>_all.parquet with y > 0
  es_dev_<tag>.parquet  = es_dev_<tag>_all.parquet for minutes that were 'ok' under the step-1 label rules, i.e. reason
                          ok / check:* / drop:SLEEPING / drop:not_carried under the current (v2) rules
--check compares against the existing step-2 tables instead of writing (verified identical on 2026-10-26).
(Needs the _all tables: uv run python make_features.py --all.)
"""
import argparse

import numpy as np
import pandas as pd

from tmd import extrasensory as es_mod
from tmd.features import FEATURES_VERSION, NAMES
from tmd.labels import CLASS_ID
from tmd.paths import PROCESSED

TAG = f"25hz_w5s_s2.5s_feat-{FEATURES_VERSION}"
V1_OK = ["ok", "drop:SLEEPING", "drop:not_carried"]


def derive() -> dict[str, pd.DataFrame]:
    shl = pd.read_parquet(PROCESSED / f"shl_u1_{TAG}_all.parquet")
    es = pd.read_parquet(PROCESSED / f"es_dev_{TAG}_all.parquet")
    es = es[es.reason.isin(V1_OK) | es.reason.str.startswith("check:")].copy()
    # minutes dropped by v2 carry y = 0 in the _all table; step 1 had their class
    cls = es_mod.minutes_table(es.user.unique())[["user", "ts", "cls"]]
    y1 = es[["user", "ts"]].merge(cls, on=["user", "ts"], how="left").cls.map(CLASS_ID).fillna(0)
    v2_drop = es.reason.isin(["drop:SLEEPING", "drop:not_carried"]).to_numpy()
    es.loc[v2_drop, "y"] = y1.to_numpy()[v2_drop].astype(np.int8)
    es["reason"] = np.where(es.reason.str.startswith("check:"), es.reason, "ok")   # step-1 reason values
    return {f"shl_u1_{TAG}.parquet": shl[shl.y > 0].reset_index(drop=True),
            f"es_dev_{TAG}.parquet": es.reset_index(drop=True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    for name, df in derive().items():
        if a.check:
            old = pd.read_parquet(PROCESSED / name)
            same = (len(old) == len(df) and list(old.columns) == list(df.columns)
                    and all(old[c].equals(df[c]) for c in old.columns if c not in NAMES)
                    and np.array_equal(old[NAMES].to_numpy(), df[NAMES].to_numpy(), equal_nan=True))
            print(name, "identical" if same else "DIFFERENT", len(old), len(df))
        else:
            df.to_parquet(PROCESSED / name)
            print("wrote", name, len(df))


if __name__ == "__main__":
    main()
