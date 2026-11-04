"""Stop-point 1 inventory beyond app_days/: per WAIIS day, the minutes by data coverage (accelerometer window
ok / GPS fix / both), GPS-only minutes by GPS state, minute windows by status, and the session count for candidate
burst-joining pauses. Prints summaries only, never coordinates.

Minute windows as waiis_bench_v2 (`tmd.candidates.trip_windows`, one 15 s window per clock minute, 25 Hz, the whole
day as one trip). A GPS minute has at least one fix with accuracy <= 500 m in the clock minute. GPS states (still /
walk / 2.5-4 / vehicle / unknown) from `tmd.readers.waiis.fix_states` (60 s baseline).

    uv run python inventory.py [<data root of the WAIIS days>]     default: <data>/raw/waiis
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tmd.candidates import trip_windows
from tmd.readers import waiis

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else waiis.ROOT
COARSE_M = waiis.COARSE_M
JOIN_S = (10, 30, 120, 600)          # candidate pauses [s] that still join two bursts into one session


def main():
    rows = []
    for d in waiis.days(ROOT):
        day = d.date
        a = waiis.read_imu(d)
        t = a.t_ms.to_numpy() / 1e3
        w, _ = trip_windows(t, a[["ax", "ay", "az"]].to_numpy(float), t, np.full(len(t), "u"))
        ok_min = set(w.ts[w.status == "ok"].astype(np.int64))
        ev_min = set((np.floor(t / 60) * 60).astype(np.int64))
        g = waiis.fix_states(waiis.read_gps(d))
        g = g[g.accuracy_m <= COARSE_M]
        g["minute"] = (np.floor(g.t / 60) * 60).astype(np.int64)
        st = g.groupby("minute").state.agg(lambda s: s.value_counts().index[0])
        gps_min = set(st.index)
        pauses = np.diff(t)
        n_sess = {f"sessions_join<={j}s": int((pauses > j).sum() + 1) for j in JOIN_S}
        only_gps = sorted(gps_min - ev_min)
        r = {"day": day, "imu_h": round((t[-1] - t[0] - pauses[pauses > 10].sum()) / 3600, 2),
             "minutes_with_acc_events": len(ev_min), "minute_windows_ok": len(ok_min),
             "gps_minutes": len(gps_min), "both": len(ok_min & gps_min), "acc_only": len(ok_min - gps_min),
             "gps_only_no_acc_events": len(only_gps), **n_sess}
        r.update({f"gps_only_{k}": int(v) for k, v in st.loc[only_gps].value_counts().items()})
        rows.append(r)
        print(day, "window status:", w.status.value_counts().to_dict())
    out = pd.DataFrame(rows).set_index("day").T.fillna(0)
    print(out.to_string())


if __name__ == "__main__":
    main()
