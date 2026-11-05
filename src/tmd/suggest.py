"""Leg suggestions for a WAIIS day (experiment 11): reader -> minute windows -> features -> saved model -> fusion ->
HMM -> legs (`tmd.legs.predict_legs`) + GPS-only legs. Models are 10's final models as saved (no training here).

GPS input (11, rule C1, approved 2026-11-04): fixes with accuracy <= 500 m, assigned to the session whose
accelerometer span +- 150 s contains them, one fix per minute nearest the minute centre, gps-v2 features.
Leg rules (11, approved 2026-11-04): E2 s_veh 0, s_other 3, g 10, l_min 2, c 0.5; E1 raw HMM legs, c 0.5.
Moving GPS gaps (12, approved 2026-11-05): `Spec.gap` (a `tmd.legs.GapRule`; None = off), for both models gap
>= 5 min, speed across 2-25 m/s, k = 2. Gaps are found on the day's good fixes (accuracy <= 500 m). R2 boosts the
raw train and subway probabilities of model minutes in a moving gap by k before the HMM; R1 adds GPS-only gap legs
(unknown, hint underground_rail).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from tmd import legs as L
from tmd.features import acc as af
from tmd.features import gps as gf
from tmd.gps import gaps, minute_gaps, nearest_per_minute
from tmd.models import fusion, minutes
from tmd.paths import PROCESSED
from tmd.readers import waiis

MODELS10 = PROCESSED / "10_gps-acc-baseline_waiis-bench-v2" / "models"
GPS_VERSION = "gps-v2"
RULES_VERSION = "legs-v2 (12)"


GAP_RULE = L.GapRule(g_min_s=300.0, v_lo=2.0, v_hi=25.0, k=2.0)   # 12, approved 2026-11-05 (both models)


@dataclass(frozen=True)
class Spec:
    id: str
    version: str
    acc: Path
    gps: Path
    hmm: Path
    rules: L.Rules
    gap: L.GapRule | None = None


SPECS = {"E2": Spec("E2", "10:test_E2_seed0", MODELS10 / "test_E2_seed0.txt", MODELS10 / "test_gps_seed0.txt",
                    MODELS10 / "test_E2_seed0_hmm.npz", L.Rules(s_veh=0, s_other=3, g=10, l_min=2, c=0.5),
                    GAP_RULE),
         "E1": Spec("E1", "10:test_E1_seed0", MODELS10 / "test_E1_seed0.txt", MODELS10 / "test_gps_seed0.txt",
                    MODELS10 / "test_E1_seed0_hmm.npz", L.Rules(c=0.5), GAP_RULE)}


@dataclass
class DayInput:
    day: waiis.Day
    minutes: pd.DataFrame        # user, session, ts, t_start, covered
    x_acc: np.ndarray
    x_gps: np.ndarray
    fixes: pd.DataFrame          # usable fixes (accuracy <= 500 m): t, lat, lon, accuracy_m, speed_mps, session
    n_samples: int
    gaps: pd.DataFrame | None = None     # gaps of the day's usable fixes (tmd.gps.gaps); minutes get gap_dt, gap_v


def day_input(day: waiis.Day) -> DayInput:
    t, xyz = waiis.read_acc(day)
    w, x = waiis.minute_windows(t, xyz)
    sess = waiis.session_ids(t)
    span = pd.DataFrame({"s": sess, "t": t}).groupby("s").t.agg(["min", "max"])
    g = waiis.read_gps(day)
    g = g[g.accuracy_m <= waiis.COARSE_M].reset_index(drop=True)
    g["session"] = -1
    for s, (lo, hi) in span.iterrows():
        m = (g.session < 0) & (g.t >= lo - gf.HALF_WINDOW_S) & (g.t <= hi + gf.HALF_WINDOW_S)
        g.loc[m, "session"] = s
    fx = g[g.session >= 0].assign(user=day.user, ts=(np.floor(g.t / 60) * 60).astype(np.int64), time=g.t)
    mins = w[["ts", "t_start", "session"]].assign(user=day.user)
    x_gps = gf.features(mins[["user", "session", "ts"]], nearest_per_minute(fx), GPS_VERSION) if len(fx) else \
        np.full((len(mins), len(gf.names(GPS_VERSION))), np.nan, np.float32)
    mins["covered"] = gf.covered(x_gps)
    gp = gaps(g.rename(columns={"t": "time"}), [])
    mg = minute_gaps(mins, gp, [])
    mins["gap_dt"], mins["gap_v"] = mg.gap_dt.to_numpy(), mg.gap_v.to_numpy()
    return DayInput(day, mins, af.features(x, sub=True, grav=False), x_gps, g, len(t), gp)


def in_gap(spec: Spec, d: DayInput) -> np.ndarray:
    """Model minutes in a moving gap under spec.gap (all False when the rule is off)."""
    if spec.gap is None or "gap_dt" not in d.minutes:
        return np.zeros(len(d.minutes), bool)
    return spec.gap.moving(d.minutes.gap_dt.to_numpy(), d.minutes.gap_v.to_numpy())


def minute_probs(spec: Spec, d: DayInput) -> tuple[np.ndarray, np.ndarray]:
    """(p after HMM, 9 columns; raw fused) in d.minutes' order."""
    if len(d.minutes) == 0:
        return np.zeros((0, 9)), np.zeros((0, 9))
    acc = lgb.Booster(model_file=str(spec.acc))
    gps = lgb.Booster(model_file=str(spec.gps))
    p_acc = acc.predict(d.x_acc, num_threads=1)
    p_gps = gps.predict(d.x_gps, num_threads=1)
    raw = fusion.product(p_acc, p_gps, d.minutes.covered.to_numpy())
    if spec.gap is not None:
        raw = L.boost_rail(raw, in_gap(spec, d), spec.gap.k)
    h = np.load(spec.hmm)
    df = d.minutes.assign(source="waiis")
    p = minutes.smooth(df, raw, h["a"], h["pi"], "fb")
    pad = lambda z: z if z.shape[1] == 9 else np.column_stack([z, np.zeros(len(z))])
    return pad(p), pad(raw)


def suggest(spec: Spec, d: DayInput) -> pd.DataFrame:
    """Model legs + GPS-only legs of one day, sorted by start; `minutes` attribute from predict_legs."""
    p, _ = minute_probs(spec, d)
    m = d.minutes
    lg = L.predict_legs(m.session.to_numpy(), m.ts.to_numpy(), p, m.covered.to_numpy(), spec.rules,
                        in_gap(spec, d))
    lo = pd.Timestamp(d.day.date, tz="UTC").timestamp()
    go = L.gps_only_legs(lg, d.fixes, lo, lo + 86400, spec.gap)
    mins = lg.attrs["minutes"]
    out = pd.concat([lg, go], ignore_index=True).sort_values("start").reset_index(drop=True)
    out.attrs["minutes"] = mins
    return out
