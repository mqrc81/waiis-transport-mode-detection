"""Experiment 11, dev pieces: 10's feature frame, saved out-of-fold probabilities and fold models, and E1 / E2's
fusion + HMM exactly as 10's run.py (no training here).

E2: acc `E2_seed<k>_fold<f>`, 8 classes, HMM from ExtraSensory + SHL User 1 + Collecty (tram minutes unlabelled).
E1: acc `E1_seed<k>_fold<f>`, 9 classes, HMM from ExtraSensory + SHL User 1 (9 states).
Both: GPS model `A0_gps_seed<k>_fold<f>` (gps-v2), product fusion on GPS-covered minutes (E1: vehicle-mean tram
factor), HMM forward-backward per fold (`tmd.models.minutes.smooth_cv`).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from tmd.evaluate import cm_metrics, six_pred
from tmd.features import acc as af
from tmd.features import gps as gf
from tmd.labels import to_score6
from tmd.models import fusion, minutes
from tmd.paths import PROCESSED

BENCH = "waiis_bench_v2"
HERE = Path(__file__).resolve().parent
EXP = HERE.name
OUT = PROCESSED / EXP
P10 = PROCESSED / "10_gps-acc-baseline_waiis-bench-v2"
MODELS10, OOF10 = P10 / "models", P10 / "oof"
ACC = af.names(True, False)                     # 05's B: 580
GPS_VERSION = "gps-v2"
GPS = gf.names(GPS_VERSION)                     # 13
K = 5
SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class Model:
    name: str
    n: int
    hmm_sources: tuple[str, ...]
    gps: str = "A0_gps"


MODELS = {"E2": Model("E2", 8, ("extrasensory", "shl_user1", "collecty")),
          "E1": Model("E1", 9, ("extrasensory", "shl_user1"))}


def frame(source: str, gps: bool = True) -> pd.DataFrame:
    """10's dev feature frame of one source (rows in 10's order), acc features + 10's per-minute GPS features."""
    a = pd.read_parquet(P10 / f"{source}_dev_25hz_w15s_acc-v1-B.parquet")
    if gps:
        g = pd.read_parquet(P10 / f"{source}_dev_gps-v2_w2min.parquet")
        assert (a.user.to_numpy() == g.user.to_numpy()).all() and (a.ts.to_numpy() == g.ts.to_numpy()).all()
        a = pd.concat([a, g[GPS]], axis=1)
    a = a.assign(source=source)
    a["y"], a["fold"] = a.y.fillna(0).astype(int), a.fold.astype(int)
    a["official"] = a.official.fillna(False).astype(bool)
    if gps:
        a["covered"] = gf.covered(a[GPS].to_numpy(np.float32))
    return a


def saved(model: str, seed: int, df: pd.DataFrame, key: str) -> np.ndarray:
    """10's saved out-of-fold array (acc, gps, raw, p) for the rows of df (one source, 10's row order)."""
    z = np.load(OOF10 / f"{model}_seed{seed}.npz", allow_pickle=True)
    src = df.source.iat[0]
    sel = z["source"] == src
    assert (z["user"][sel] == df.user.to_numpy(str)).all() and (z["ts"][sel] == df.ts.to_numpy()).all(), "row order"
    return z[key][sel]


def fold_models(name: str, seed: int) -> list[lgb.Booster]:
    return [lgb.Booster(model_file=str(MODELS10 / f"{name}_seed{seed}_fold{f}.txt")) for f in range(K)]


def predict_oof(models: list[lgb.Booster], x: np.ndarray, fold: np.ndarray) -> np.ndarray:
    out = None
    for f, m in enumerate(models):
        sel = fold == f
        if not sel.any():
            continue
        p = m.predict(x[sel])
        out = np.full((len(x), p.shape[1]), np.nan) if out is None else out
        out[sel] = p
    return out


def labelled(sources: tuple[str, ...], n: int) -> pd.DataFrame:
    """The HMM's training minutes, as 10's pipe.labelled."""
    lab = pd.concat([minutes.labelled_minutes(s, BENCH, fine=True) for s in sources], ignore_index=True)
    return lab[lab.y8 <= n].reset_index(drop=True)


def pad9(p: np.ndarray) -> np.ndarray:
    return p if p.shape[1] == 9 else np.column_stack([p, np.zeros(len(p))])


def fuse_smooth(model: Model, df: pd.DataFrame, p_acc: np.ndarray, p_gps: np.ndarray, covered: np.ndarray,
                lab: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(raw, p): product fusion where covered, then the HMM per fold; p has 9 columns."""
    lab = labelled(model.hmm_sources, model.n) if lab is None else lab
    raw = fusion.product(p_acc, p_gps, covered)
    return raw, pad9(minutes.smooth_cv(df, raw, lab, "fb"))


def six_f1(y: np.ndarray, p9: np.ndarray) -> float:
    y6, q6 = to_score6(y), six_pred(p9)
    cm = np.zeros((6, 6), np.int64)
    np.add.at(cm, (y6 - 1, q6 - 1), 1)
    return cm_metrics(cm, np.isin(np.arange(1, 7), np.unique(y6)))["macro_f1"]


# ---------------------------------------------------------------- legs on dev

@dataclass
class LegData:
    """One source's minutes sorted by (sequence, ts), with top-level probabilities and truth (top id, -1 = none)."""
    order: np.ndarray
    seq: np.ndarray
    ts: np.ndarray
    truth: np.ndarray
    day: np.ndarray
    official: np.ndarray


def leg_data(df: pd.DataFrame) -> LegData:
    """Truth = every labelled minute of the source (`tmd.models.minutes.labelled_minutes`, fine, v2 rules), mapped
    to the top level; sequences = source / user / session."""
    from tmd.legs import fine_to_top
    src = df.source.iat[0]
    lab = minutes.labelled_minutes(src, BENCH, fine=True)[["user", "ts", "y8"]]
    y = df[["user", "ts"]].merge(lab, on=["user", "ts"], how="left", validate="1:1").y8.fillna(0).astype(int)
    seq = minutes.seq_ids(df)
    order = np.lexsort((df.ts.to_numpy(), seq))
    # day key: SHL's recording day; ExtraSensory (empty `day`): user + UTC date
    ud = df.user.astype(str) + "/" + pd.to_datetime(df.ts, unit="s").dt.date.astype(str)
    day = df.day.astype(str).where(df.day.astype(str) != "", ud).to_numpy() if "day" in df else ud.to_numpy()
    return LegData(order, seq[order], df.ts.to_numpy(np.int64)[order], fine_to_top(y.to_numpy())[order],
                   day[order], df.official.to_numpy()[order])


def leg_eval(d: LegData, p9: np.ndarray, rules=None):
    """Build legs from 9-column minute probabilities (frame order) and score them against the truth legs."""
    from tmd import legs as L
    q = L.top_probs(p9[d.order])
    lg = L.build(d.seq, d.ts, q, rules or L.RAW)
    tl = L.truth_legs(d.seq, d.ts, d.truth)
    return lg, L.score(lg, tl, len(q), q, d.day)
