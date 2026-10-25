"""Step 1: summary tables from the audit outputs (run after audit_shl.py and audit_es.py).

Usage: uv run python summarise_audit.py
Writes results/audit/shl_hours_user_position_class.csv and es_minutes_user_class_position.csv.
"""
from pathlib import Path

import pandas as pd

from tmd.labels import CLASSES
from tmd.paths import PROCESSED

AUD = Path(__file__).resolve().parent / "results" / "audit"


def main():
    d = pd.read_csv(AUD / "shl_recordings.csv", dtype={"rec": str, "day": str})
    dup = set(pd.read_csv(AUD / "shl_duplicates.csv", dtype={"day": str}).day)
    d = d[~((d.session == "User1") & d.day.isin(dup))]           # keep the User1_4phones copy
    h = [f"h_{c}" for c in CLASSES]
    t = d.groupby(["user", "position"])[h + ["hours_null", "hours_no_accgyro", "hours"]].sum().round(1)
    t.columns = [c.removeprefix("h_") for c in t.columns]
    t.insert(0, "days", d.groupby(["user", "position"]).day.nunique())
    t.to_csv(AUD / "shl_hours_user_position_class.csv")
    print(t.to_string())

    m = pd.read_parquet(PROCESSED / "es_minutes_audit.parquet")
    ok = (m.reason == "ok") & (m.acc_n > 0) & (m.gyro_n > 0)
    m = m[ok].assign(position=m.position.replace("", "none"))
    m.groupby(["uuid", "platform", "cls", "position"]).size().rename("minutes").to_csv(
        AUD / "es_minutes_user_class_position.csv")
    print(m.groupby(["cls", "position"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
