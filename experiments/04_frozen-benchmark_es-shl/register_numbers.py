"""Numbers for decision_register.md, from es_bench_v1 and 02's audit (read-only).

Usage: uv run python register_numbers.py > logs/register_numbers.out
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from functools import partial

from tmd import bench
from tmd.paths import PROCESSED, PROJECT

V1 = "es_bench_v1"                 # the register describes v1
read = partial(bench.read, version=V1)
path = partial(bench.path, version=V1)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
C7 = ["still", "walk", "run", "bike", "car", "bus", "train"]


def h(t):
    print(f"\n######## {t}")


L = read("extrasensory", "labels")
A = pq.read_table(path("extrasensory", "accelerometer"),
                  columns=["user", "ts", "unit", "raw_n", "raw_rate_hz", "span_s", "start_offset_s"]).to_pandas()
AUD = pd.read_parquet(PROCESSED / "es_minutes_audit.parquet", columns=["uuid", "timestamp", "acc_norm_med", "acc_file",
                                                                       "acc_dur_s", "acc_dt_med_ms"])
AUD = AUD.rename(columns={"uuid": "user", "timestamp": "ts"})
L["plat"] = L.platform
cls = L.cls.replace("", "(none)")

h("labels: reason x platform x cls")
print(pd.crosstab([L.reason], [cls], margins=True))
print(pd.crosstab(L.reason, L.platform))

h("users affected per reason")
print(L.groupby("reason").user.nunique())

h("top users per reason (minutes)")
for r in ["drop:SLEEPING", "drop:not_carried", "conflict", "drop:ON_A_PLANE", "drop:ELEVATOR", "unlabelled"]:
    s = L[L.reason == r].user.str[:8].value_counts()
    print(r, "users", len(s), "top3", s.head(3).to_dict(), "top3 share", round(s.head(3).sum() / s.sum(), 2))

h("conflicts: moving sets")
print(L[L.reason == "conflict"].groupby(["moving", "platform"]).size())

h("not_carried by cls x platform")
nc = L[L.reason == "drop:not_carried"]
print(pd.crosstab(nc.cls, nc.platform, margins=True))

h("cap: still ok minutes, resting share, capped per platform")
ok = L[L.reason == "ok"]
st = ok[ok.cls == "still"]
print("still ok", len(st), "resting", st.resting.sum(), "capped", st.capped.sum())
print(pd.crosstab(st.platform, [st.resting, st.capped]))
u = st.groupby("user").agg(n=("ts", "size"), resting=("resting", "sum"), capped=("capped", "sum"),
                          split=("split", "first"), plat=("platform", "first"))
u["share_capped"] = (u.capped / u.n).round(2)
print(u.sort_values("capped", ascending=False).head(8).rename(index=lambda x: x[:8]))
print("users with >50% of still capped:", (u.share_capped > .5).sum(), "of", len(u))
inc = L[L.included_v2 & L.has_acc]
s2 = inc[inc.cls == "still"]
print("examples: still resting share", round(s2.resting.mean(), 3), "unknown position share",
      round((s2.position == "").mean(), 3), "carried (pocket/hand/bag)", round(s2.position.isin(["in_pocket", "in_hand", "in_bag"]).mean(), 3))
print("still examples by position x platform\n", pd.crosstab(s2.position.replace("", "(none)"), s2.platform))
print("test vs dev still resting share", s2.groupby("split").resting.mean().round(3).to_dict())

h("label composition per class (included_v2)")
iv = L[L.included_v2]
def p(c):
    return iv[c].fillna(0).to_numpy() == 1
comp = {
    "car: IN_A_CAR": p("label:IN_A_CAR"), "car: DRIVER": p("label:DRIVE_-_I_M_THE_DRIVER"),
    "car: PASSENGER": p("label:DRIVE_-_I_M_A_PASSENGER"),
    "bus: +PASSENGER": p("label:ON_A_BUS") & p("label:DRIVE_-_I_M_A_PASSENGER"),
    "walk: FIX_walking": p("label:FIX_walking"), "walk: STAIRS": p("label:STAIRS_-_GOING_UP") | p("label:STAIRS_-_GOING_DOWN"),
    "still: SITTING": p("label:SITTING"), "still: OR_standing": p("label:OR_standing"), "still: LYING_DOWN": p("label:LYING_DOWN"),
    "train: clean SITTING": p("label:SITTING"),
}
for k, m in comp.items():
    c = k.split(":")[0]
    mm = m & (iv.cls == c).to_numpy()
    print(f"{k:24s} {mm.sum():7d} of {int((iv.cls == c).sum())}")
w = iv[iv.cls == "walk"]
print("walk via STAIRS only (no FIX_walking):", int(((w["label:FIX_walking"].fillna(0) != 1) & ((w["label:STAIRS_-_GOING_UP"].fillna(0) == 1) | (w["label:STAIRS_-_GOING_DOWN"].fillna(0) == 1))).sum()))
c = iv[iv.cls == "car"]
print("car via PASSENGER only:", int(((c["label:IN_A_CAR"].fillna(0) != 1) & (c["label:DRIVE_-_I_M_THE_DRIVER"].fillna(0) != 1)).sum()))
print("still minutes where a moving class also... n/a (moving wins)")
print("label_source by class (included_v2)\n", pd.crosstab(iv.cls, iv.label_source, normalize="index").round(2))
print("cleaned ELEVATOR not in orig:", int(((L["label:ELEVATOR"].fillna(0) == 1) & (L["orig:ELEVATOR"].fillna(0) != 1)).sum()) if "label:ELEVATOR" in L else "no col")
for d in ["ELEVATOR", "ON_A_PLANE", "SKATEBOARDING", "STATIONARY_BIKE", "TREADMILL", "ELLIPTICAL_MACHINE", "ON_A_BOAT", "WHEELCHAIR", "MOTORBIKE"]:
    m = L[L.reason.str.endswith(d)]
    print(d, "minutes", len(m), "users", m.user.nunique(), "would-be cls", m.cls.replace("", "(none)").value_counts().to_dict(),
          "plat", m.platform.value_counts().to_dict())
print("cleaned-only labels for orig STROLLING etc not dropped: STROLLING->", L[L["orig:STROLLING"].fillna(0) == 1].cls.value_counts().to_dict() if "orig:STROLLING" in L else "")
print("unlabelled by platform", L[L.reason == "unlabelled"].platform.value_counts().to_dict(), "with window", int((L.reason.eq("unlabelled") & L.has_acc).sum()))
print("orig labels that might be transport, minutes:",
      {c: int((L[c].fillna(0) == 1).sum()) for c in L.columns if c.startswith("orig:") and any(k in c for k in ["TRAM", "SUBWAY", "METRO", "SCOOTER", "TAXI", "UBER", "SHUTTLE", "TROLLEY", "LIGHT_RAIL", "TRAIN", "BUS", "CAR", "MOTOR", "DRIV"])})

h("units")
a = A.merge(L[["user", "ts", "platform", "cls", "included_v2"]], on=["user", "ts"])
print(pd.crosstab(a.platform, a.unit))
g = a[a.unit == "g"]
print("android g by user", g[g.platform == "android"].user.str[:8].value_counts().to_dict())
au = AUD.merge(L[["user", "ts", "platform"]], on=["user", "ts"])
nm = au.acc_norm_med
print("raw median |acc| in [1.5, 6] (ambiguous zone):", int(((nm > 1.5) & (nm < 6)).sum()), "of", int(nm.notna().sum()),
      "| in [2.5, 3.5]:", int(((nm > 2.5) & (nm < 3.5)).sum()))
print("raw |acc| median quantiles per platform\n", au.groupby("platform").acc_norm_med.quantile([.001, .01, .5, .99, .999]).unstack().round(2))

h("rate / edge margin (windows)")
a["n25"] = np.floor(a.span_s * 25).astype(int) + 1
a["margin"] = (a.n25 - 375) // 2
print(a.groupby("platform").raw_rate_hz.describe(percentiles=[.01, .05, .5, .95, .99]).round(1))
print("rate < 30 Hz:", int((a.raw_rate_hz < 30).sum()))
print("margin < 10 samples (0.4 s, inside FIR edge):", pd.crosstab(a.margin < 10, a.platform))
ex = a[a.included_v2]
print("examples with margin<10 by cls", ex[ex.margin < 10].cls.value_counts().to_dict())
print("discarded seconds per window (span-15): ", a.assign(d=a.span_s - 15).groupby("platform").d.describe().round(1))

h("cleaning & 15s per class x platform (included_v2), and all minutes")
iv2 = L[L.included_v2]
print(pd.crosstab([iv2.acc_status], [iv2.platform, iv2.cls]))
print(pd.crosstab(L.acc_status, L.platform, margins=True))
for r in ["time_not_increasing", "hole_over_0.5s", "span_over_40s", "no_file"]:
    s = L[L.acc_status == r].user.str[:8].value_counts()
    print(r, "users", len(s), "top3", s.head(3).to_dict())
print("15 s loss by split x cls", pd.crosstab(iv2[iv2.acc_status == "under_15s"].split, iv2[iv2.acc_status == "under_15s"].cls))
print("unlabelled minutes lost to 15s", int((L.reason.eq("unlabelled") & L.acc_status.eq("under_15s")).sum()))

h("splits")
ex = L[L.included_v2 & L.has_acc]
sp = pd.crosstab(ex.split, ex.cls)[C7]
print(sp, "\n", (sp / sp.sum()).round(3))
t = ex[ex.split == "test"]
print("test users with >=5 examples per class:", {c: int((t[t.cls == c].groupby("user").size() >= 5).sum()) for c in C7})
d = ex[ex.split == "dev"]
print("dev users with >=5 per class:", {c: int((d[d.cls == c].groupby("user").size() >= 5).sum()) for c in C7})
print("platform users per split", L.groupby("split").apply(lambda g: g.groupby("user").platform.first().value_counts().to_dict()))
print("examples per split x platform", pd.crosstab(ex.split, ex.platform))
gl = ex[ex.user.str[:8].isin(["61359772", "CCAF77F0", "F50235E0"])]
print("gyro-less users' examples", pd.crosstab(gl.user.str[:8], gl.cls), gl.groupby("user").fold.first().to_dict())
hi_rate = ["81536B0A", "1DBB0F6F", "CCAF77F0", "F50235E0", "27E04243", "96A358A0"]
print("high-rate users in test:", [u for u in hi_rate if (L[L.user.str[:8] == u].split == "test").any()])
ua = pd.read_csv(PROJECT / "experiments/02_family-baseline_shl-es/splits/es_users.csv")
print("pick-time counts (es_users.csv in 02 audit?)", ua.columns.tolist())
aud = pd.read_csv(PROJECT / "experiments/02_family-baseline_shl-es/results/audit/es_users.csv", index_col="uuid")
nn = aud[[f"n_{c}" for c in C7]]
nn.columns = C7
tu = set(ua.uuid[ua.partition == "test"])
print("pick-time test share", (nn.loc[nn.index.isin(tu)].sum() / nn.sum()).round(3).to_dict())
print("pick-time total", nn.sum().to_dict())

h("GPS ES")
G = read("extrasensory", "gps_per_minute")
F = read("extrasensory", "location_features")
ex = L[L.included_v2 & L.has_acc]
print("has_location by cls x platform\n", ex.pivot_table(index="cls", columns="platform", values="has_location", aggfunc="mean").round(3).reindex(C7))
print("by split", ex.groupby("split").has_location.mean().round(3).to_dict())
gg = G.merge(L[["user", "ts", "cls", "platform", "included_v2", "has_acc"]], on=["user", "ts"])
ge = gg[gg.included_v2 & gg.has_acc]
print("speed -1 by cls x platform\n", ge.pivot_table(index="cls", columns="platform", values="speed", aggfunc=lambda s: (s < 0).mean()).round(3).reindex(C7))
print("accuracy p50 by platform", ge[ge.accuracy >= 0].groupby("platform").accuracy.median().to_dict())
print("min_speed available but max_speed NaN:", int((F["location:min_speed"].notna() & F["location:max_speed"].isna()).sum()))
print("max_speed > 60 m/s:", int((F["location:max_speed"] > 60).sum()), "> 45:", int((F["location:max_speed"] > 45).sum()))
print("ES sessions", read("extrasensory", "sessions").minutes.describe().round(1).to_dict())
print("missing reason by cls\n", pd.crosstab(L.cls.replace("", "(none)"), L.location_missing_reason.replace("", "present")))

h("SHL")
for s in ["shl_user1", "shl_user23"]:
    S = read(s, "labels")
    print(s, pd.crosstab(S.label_status, S.acc_status, margins=True))
    mx = S[S.label_status == "mixed"]
    print(" mixed majority cls", mx.majority_cls.value_counts().to_dict(), "majority_share p50", round(mx.majority_share.median(), 2))
    print(" mixed with majority >= 0.8:", int((mx.majority_share >= .8).sum()))
    W = read(s, "gps_waiis_cadence"); M = read(s, "gps_per_minute")
    print(" waiis fixes", len(W), "blanked", int(W.speed_blanked.sum()), "speed -1 natural", int(((W.speed < 0) & ~W.speed_blanked).sum()),
          "| per-minute", len(M), "blanked", int(M.speed_blanked.sum()))
    e = S[S.included & S.has_acc]
    print(" examples with per-minute fix by cls", e.groupby("cls").has_location.mean().round(3).to_dict())
print("SHL user1 duplicates removed (3 days, hips) - see 02 audit")
dup = pd.read_csv(PROJECT / "experiments/02_family-baseline_shl-es/results/audit/shl_duplicates.csv")
print(dup.head())
