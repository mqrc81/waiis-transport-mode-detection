"""Descriptive check before the Conclusion is approved (dev only, no retraining): what lies behind E1's false rail.

1. Share of `quiet` windows among labelled vehicle windows (bus, car, train, subway, tram), per source, with the
   per-user spread (official dev windows: what the models train on).
2. E1's ExtraSensory dev still minutes predicted rail (seed 0, after the HMM): users, platform, sessions, contiguous
   runs, ExtraSensory's context tags; quiet and not-quiet separately, each next to all still minutes of its kind.
3. For the not-quiet ones: vehicle labels (cls car / bus / train, or the original ON_A_TRAIN / ON_A_BUS / IN_A_CAR /
   DRIVE tags) of the same user within ±10 / ±30 min, and on the same day (UTC), against the same rate for every
   not-quiet still minute.
4. E1's false bus on ExtraSensory dev (six-class bus predicted on other classes), A0 alongside: by true class and
   with / without GPS.

    uv run python diag_quiet.py         # results/tables/diag_quiet_*.csv; logs/diag_quiet.log
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import pipe
from tmd import bench

OUT = pipe.HERE / "results" / "tables"
VEH = ["bus", "car", "train", "subway", "tram"]
VEH_TAGS = ["orig:ON_A_TRAIN", "orig:ON_A_BUS", "orig:IN_A_CAR", "orig:DRIVE_-_I_M_THE_DRIVER",
            "orig:DRIVE_-_I_M_A_PASSENGER", "orig:MOTORBIKE", "orig:ON_A_BOAT", "orig:ON_A_PLANE"]
CTX = ["PHONE_ON_TABLE", "PHONE_IN_POCKET", "PHONE_IN_HAND", "PHONE_IN_BAG", "label:LOC_home",
       "label:LOC_main_workplace", "label:OR_indoors", "label:OR_outside", "label:SITTING", "label:LYING_DOWN",
       "label:SLEEPING", "label:COMPUTER_WORK", "label:IN_CLASS", "label:IN_A_MEETING", "label:EATING",
       "label:WATCHING_TV", "label:SURFING_THE_INTERNET", "label:TALKING", "has_location"]
log = logging.getLogger()


def quiet_shares() -> pd.DataFrame:
    rows = []
    for s in pipe.SOURCES:
        e = bench.examples(s, ["fold", "split", "cls_fine", "quiet"], version=pipe.BENCH)
        e = e[(e.fold >= 0) & e.cls_fine.isin(VEH)]
        for c, d in e.groupby("cls_fine"):
            u = d.groupby("user").quiet.agg(["mean", "size"])
            u5 = u[u["size"] >= 5]["mean"]                     # users with >= 5 windows for the spread
            rows.append({"source": s, "class": c, "windows": len(d), "users": len(u), "quiet_share": d.quiet.mean(),
                         "users_ge5": len(u5), "user_median": u5.median(), "user_q25": u5.quantile(.25),
                         "user_q75": u5.quantile(.75), "user_max": u5.max(),
                         "users_any_quiet": int((u["mean"] > 0).sum())})
    return pd.DataFrame(rows)


def runs(ts: np.ndarray) -> list[int]:
    """Lengths of runs of consecutive minutes (steps of 60 s +- 30 s)."""
    if not len(ts):
        return []
    ts = np.sort(ts)
    brk = np.flatnonzero(np.abs(np.diff(ts) - 60) > 30)
    return list(np.diff(np.concatenate([[0], brk + 1, [len(ts)]])))


def main():
    logging.basicConfig(filename=pipe.HERE / "logs" / "diag_quiet.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    pd.set_option("display.width", 200)
    q = quiet_shares()
    q.round(3).to_csv(OUT / "diag_quiet_vehicle_share.csv", index=False)
    print(q.round(3).to_string(index=False), flush=True)
    log.info("\n%s", q.round(3).to_string(index=False))

    df = pipe.load()
    es = ((df.source == "extrasensory") & df.official).to_numpy() & (df.y.to_numpy() == 1)
    f = pipe.load_probs("E1", 0, df)[es].argmax(1) + 1
    d = df.loc[es, ["user", "ts", "session", "platform", "quiet", "covered"]].reset_index(drop=True)
    d["rail"] = np.isin(f, [7, 8, 9])
    d["pred"] = np.array(pipe.NAMES["fine"])[f - 1]
    lab = bench.read("extrasensory", "labels", version=pipe.BENCH)
    lab = lab[~lab.user.isin(bench.reserved_users("extrasensory", pipe.BENCH))]
    d = d.merge(lab[["user", "ts", "position"] + CTX], on=["user", "ts"], how="left", validate="1:1")

    # 2. who, where, how contiguous
    rows, per_user = [], []
    for nm, mk in (("quiet", d.quiet), ("not_quiet", ~d.quiet)):
        a, r = d[mk], d[mk & d.rail]
        rr = [x for (_, s), g in r.groupby(["user", "session"]) for x in runs(g.ts.to_numpy())]
        row = {"part": nm, "still_minutes": len(a), "false_rail": len(r), "users": r.user.nunique(),
               "sessions": r.groupby(["user", "session"]).ngroups,
               "top_user_share": r.user.value_counts(normalize=True).max() if len(r) else np.nan,
               "android_share": (r.platform == "android").mean(), "android_share_all": (a.platform == "android").mean(),
               "gps_share": r.covered.mean(), "runs": len(rr), "run_len_median": np.median(rr) if rr else np.nan,
               "run_len_max": max(rr) if rr else 0, "in_runs_ge5": sum(x for x in rr if x >= 5),
               "pred_subway": int((r.pred == "subway").sum()), "pred_train": int((r.pred == "train").sum())}
        for c in ["position"]:
            for v in ("on_table", "in_pocket", "in_hand", "in_bag", ""):
                row[f"pos_{v or 'none'}"] = (r.position == v).mean()
                row[f"pos_{v or 'none'}_all"] = (a.position == v).mean()
        for c in CTX:
            row[f"{c}"] = pd.to_numeric(r[c], errors="coerce").fillna(0).astype(float).mean()
            row[f"{c}_all"] = pd.to_numeric(a[c], errors="coerce").fillna(0).astype(float).mean()
        rows.append(row)
        u = r.groupby("user").agg(minutes=("ts", "size"), sessions=("session", "nunique"),
                                  platform=("platform", "first"))
        per_user.append(u.assign(part=nm))
    t2 = pd.DataFrame(rows).set_index("part").T
    t2.round(3).to_csv(OUT / "diag_quiet_false_rail.csv")
    pu = pd.concat(per_user).reset_index()
    pu["user"] = pu.user.str[:8]
    pu.to_csv(OUT / "diag_quiet_false_rail_users.csv", index=False)
    print(t2.round(3).to_string(), "\n", pu.sort_values("minutes", ascending=False).to_string(index=False), flush=True)
    log.info("\n%s\n%s", t2.round(3).to_string(), pu.to_string(index=False))

    # 3. vehicle labels nearby (same user), for the not-quiet false rail vs every not-quiet still minute
    veh = lab.cls.isin(["car", "bus", "train"]) | (lab[VEH_TAGS].fillna(0).astype(float) > 0).any(axis=1)
    vt = lab[veh][["user", "ts"]]
    vby = {u: np.sort(g.ts.to_numpy()) for u, g in vt.groupby("user")}
    day = lambda t: (np.asarray(t) // 86400)

    def near(user, ts, w):
        v = vby.get(user)
        if v is None:
            return False
        i = np.searchsorted(v, ts - w)
        return i < len(v) and v[i] <= ts + w

    out = []
    nq = d[~d.quiet].copy()
    for w in (600, 1800):
        nq[f"veh_{w // 60}min"] = [near(u, t, w) for u, t in zip(nq.user, nq.ts)]
    nq["veh_same_day"] = [bool(len(vby.get(u, [])) and (day(vby[u]) == day(t)).any()) for u, t in zip(nq.user, nq.ts)]
    nq["own_vehicle_tag"] = (lab.set_index(["user", "ts"]).loc[pd.MultiIndex.from_frame(nq[["user", "ts"]]), VEH_TAGS]
                             .fillna(0).astype(float) > 0).any(axis=1).to_numpy()
    sets = (("false_rail", nq.rail), ("false_rail_no_gps", nq.rail & ~nq.covered),
            ("false_rail_gps", nq.rail & nq.covered), ("all_not_quiet_still", np.ones(len(nq), bool)),
            ("all_not_quiet_still_no_gps", ~nq.covered))
    for nm, mk in sets:
        g = nq[mk]
        rr = [x for _, h in g.groupby(["user", "session"]) for x in runs(h.ts.to_numpy())]
        out.append({"set": nm, "minutes": len(g), "users": g.user.nunique(), "sessions": g.groupby(["user", "session"]).ngroups,
                    "runs": len(rr), "in_runs_ge5": sum(x for x in rr if x >= 5),
                    "top_user": g.user.value_counts().iloc[0] if len(g) else 0,
                    "android_share": (g.platform == "android").mean(), "sitting": g["label:SITTING"].astype(float).mean(),
                    "talking": g["label:TALKING"].astype(float).mean(), "indoors": g["label:OR_indoors"].astype(float).mean(),
                    **{c: g[c].mean() for c in ("veh_10min", "veh_30min", "veh_same_day", "own_vehicle_tag")}})
    t3 = pd.DataFrame(out).set_index("set")
    # which tags, near the false-rail minutes
    fr = nq[nq.rail & ~nq.covered]                         # the 56 not-quiet false-rail minutes without GPS
    tags = {}
    for c in VEH_TAGS + ["cls_car", "cls_bus", "cls_train"]:
        if c.startswith("cls_"):
            vv = lab[lab.cls == c[4:]][["user", "ts"]]
        else:
            vv = lab[lab[c].fillna(0).astype(float) > 0][["user", "ts"]]
        by = {u: np.sort(g.ts.to_numpy()) for u, g in vv.groupby("user")}
        tags[c] = int(sum(bool(len(by.get(u, [])) and (np.abs(by[u] - t) <= 1800).any()) for u, t in zip(fr.user, fr.ts)))
    t3.round(3).to_csv(OUT / "diag_quiet_nearby_vehicle.csv")
    pd.Series(tags, name="false_rail_not_quiet_no_gps_minutes_within_30min").to_csv(OUT / "diag_quiet_nearby_tags.csv")
    print(t3.round(3).to_string(), "\n", tags, flush=True)
    log.info("\n%s\n%s", t3.round(3).to_string(), tags)

    t4 = false_bus(df)
    t4.round(3).to_csv(OUT / "diag_quiet_false_bus.csv")
    print(t4.round(3).to_string(), flush=True)
    log.info("\n%s", t4.round(3).to_string())


def false_bus(df) -> pd.DataFrame:
    es = ((df.source == "extrasensory") & df.official).to_numpy()
    y6 = pipe.truth(df.y.to_numpy())["six"][es]
    cov = df.covered.to_numpy()[es]
    rows = []
    for r in ("A0", "E1"):
        pr = pipe.preds(pipe.load_probs(r, 0, df)[es])["six"]
        fb = (pr == 5) & (y6 != 5)
        row = {"row": r, "pred_bus": int((pr == 5).sum()), "false_bus": int(fb.sum()),
               "false_bus_gps": int((fb & cov).sum()), "false_bus_gps_share": float(cov[fb].mean()),
               "es_official_gps_share": float(cov.mean())}
        for i, c in enumerate(pipe.NAMES["six"], start=1):
            if c != "bus":
                row[f"from_{c}"] = int((fb & (y6 == i)).sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("row")


if __name__ == "__main__":
    main()
