"""Step 2 follow-up: which ExtraSensory labels make up each class (source label x phone position x label_source).

Usage: uv run python es_label_review.py
Uses labels only (all 60 users, as in the audit; no model output). Writes results/audit/es_label_review.csv and prints
a short summary.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from tmd import extrasensory as es

AUD = Path(__file__).resolve().parent / "results" / "audit"
STILL_SRC = ["SITTING", "OR_standing", "LYING_DOWN", "SLEEPING"]
POS = {"table": "PHONE_ON_TABLE", "pocket": "PHONE_IN_POCKET", "hand": "PHONE_IN_HAND", "bag": "PHONE_IN_BAG"}
ORIG_POS = {"away": "PHONE_AWAY_FROM_ME", "strapped": "PHONE_STRAPPED"}


def one(uuid: str) -> pd.DataFrame:
    c, o = es.read_cleaned_labels(uuid), es.read_original_labels(uuid)
    m = es.minute_classes(c, o)
    o = o.reindex(c.index)
    m = m[m.reason == "ok"].copy()
    cc, oo = c.loc[m.index], o.loc[m.index]
    for s in STILL_SRC:
        m[s] = (cc[s].to_numpy() == 1) if s in cc else False
    for k, col in POS.items():
        m[f"pos_{k}"] = ((cc[col].to_numpy() == 1) if col in cc else False) | (oo[col].to_numpy() == 1)
    for k, col in ORIG_POS.items():
        m[f"pos_{k}"] = oo[col].to_numpy() == 1
    m["n_pos"] = m[[f"pos_{k}" for k in [*POS, *ORIG_POS]]].sum(1)
    m["hour_utc"] = pd.to_datetime(m.index, unit="s").hour
    m.insert(0, "uuid", uuid)
    return m.reset_index()


def main():
    with ProcessPoolExecutor(6) as ex:
        df = pd.concat(ex.map(one, es.uuids()), ignore_index=True)
    df.to_parquet(es.ES_RAW.parents[1] / "processed" / "es_label_review.parquet")
    pos_cols = [f"pos_{k}" for k in [*POS, *ORIG_POS]]

    def pos_name(r):
        on = [c[4:] for c in pos_cols if r[c]]
        return "+".join(on) if on else "none"
    df["pos"] = df[pos_cols].apply(pos_name, axis=1)
    rows = []
    for cls, g in df.groupby("cls"):
        row = {"cls": cls, "minutes": len(g), "users": g.uuid.nunique()}
        row.update({f"pos={p}": int(n) for p, n in g.pos.value_counts().items()})
        row.update({f"src={s}": int(g[s].sum()) for s in STILL_SRC if cls == "still"})
        row.update({f"label_source={k}": int(n) for k, n in g.label_source.value_counts().sort_index().items()})
        rows.append(row)
    t = pd.DataFrame(rows).fillna(0)
    t.to_csv(AUD / "es_label_review.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 50)
    print(t.set_index("cls").T.to_string())
    st = df[df.cls == "still"]
    print("\nstill: source label x position (minutes)")
    src = np.select([st.SLEEPING, st.LYING_DOWN, st.SITTING, st.OR_standing], ["sleeping", "lying", "sitting", "standing"], "?")
    print(pd.crosstab(src, st.pos).to_string())
    print("\nstill per user: share on table (quantiles)", st.groupby("uuid").pos_table.mean().quantile([.1, .5, .9]).round(2).to_dict())


if __name__ == "__main__":
    main()
