"""Label-budget sampling: k windows per class, spread over days and trips (experiment 03).

A trip is a maximal run of consecutive pure windows of one class in one recording (one 2.5 s step apart). Per (seed,
class) the windows get one order: days shuffled, trips within a day shuffled, windows within a trip shuffled; a day's
queue takes one window from each trip in turn, and the class order takes one window from each day's queue in turn.
The first k of that order is the k-set, so sets are nested in k.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_STEP_MS = 2600            # strides are 2480 / 2520 ms at 25 Hz


def trips(w: pd.DataFrame) -> pd.Series:
    """Trip id per row of w (columns rec, t_start_ms, y; pure windows only), same index as w."""
    s = w.sort_values(["rec", "t_start_ms"])
    new = ((s.rec != s.rec.shift()) | (s.y != s.y.shift()) | (s.t_start_ms.diff() > MAX_STEP_MS))
    return new.cumsum().rename("trip").reindex(w.index)


def class_order(w: pd.DataFrame, seed: int, class_id: int) -> pd.Index:
    """Round-robin order (days, then trips) of the pure windows of one class.

    w: columns day, rec, t_start_ms, y, trip (pure windows; other classes are ignored). Returns w's index in order.
    """
    rng = np.random.default_rng([seed, class_id])
    c = w[w.y == class_id].sort_values(["day", "rec", "t_start_ms"])
    keys = []
    days = np.array(sorted(c.day.unique()))
    for d_pos, day in enumerate(rng.permutation(days)):
        cd = c[c.day == day]
        tids = np.array(sorted(cd.trip.unique()))
        q = []                                   # (position in trip, trip position, index)
        for t_pos, tid in enumerate(rng.permutation(tids)):
            idx = rng.permutation(cd.index[cd.trip == tid].to_numpy())
            q += [(j, t_pos, i) for j, i in enumerate(idx)]
        q.sort()
        keys += [(qi, d_pos, i) for qi, (_, _, i) in enumerate(q)]
    keys.sort()
    return pd.Index([i for _, _, i in keys])


def ranks(w: pd.DataFrame, seed: int, classes: list[int]) -> pd.Series:
    """Rank of every pure window of the given classes in its class order (0 = first), index of w."""
    out = pd.Series(-1, index=w.index, name="rank")
    for c in classes:
        o = class_order(w, seed, c)
        out[o] = np.arange(len(o))
    return out
