"""Code move check (12, approved 2026-11-05): 11's dev helpers moved to `tmd.models.e10`. Run once with 11's dev.py
before the move (`old`) and once with `tmd.models.e10` after (`new`); the leg scores must be identical.

For E1 and E2, seed 0: SHL User 1 hips with C1 at 20 s recomputed end to end (thinned track -> C1 features -> A0_gps
fold models -> fusion + HMM per fold), compared to 11's saved p; leg scores (11's approved rules and raw legs) on SHL
User 1 and ExtraSensory dev (10's saved p).

    uv run python check_move.py old|new      # writes logs/check_move_<which>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from tmd import gps as tgps
from tmd import legs as L
from tmd import suggest as S
from tmd.features import gps as gf

HERE = Path(__file__).resolve().parent
which = sys.argv[1]
if which == "old":
    sys.path.insert(0, str(HERE.parent / "11_leg-suggestions_waiis-app-days"))
    import dev as E
    from gps_candidates import thinned_track
else:
    from tmd.models import e10 as E
    thinned_track = E.shl_thinned_track
P11 = E.PROCESSED / "11_leg-suggestions_waiis-app-days"

shl, es = E.frame("shl_user1"), E.frame("extrasensory")
x = gf.features(shl[["user", "session", "ts"]], tgps.nearest_per_minute(thinned_track(shl)), E.GPS_VERSION)
assert np.allclose(x, np.load(P11 / "shl_user1_gps_C1.npy"), equal_nan=True), "C1 features differ from 11's"
out = {}
for mid, spec in S.SPECS.items():
    p_gps = E.predict_oof(E.fold_models("A0_gps", 0), x, shl.fold.to_numpy())
    _, p = E.fuse_smooth(E.MODELS[mid], shl, E.saved(mid, 0, shl, "acc"), p_gps, gf.covered(x))
    d = float(np.abs(p - np.load(P11 / f"shl_user1_{mid}_seed0_C1_p.npy")).max())
    out[f"{mid}_p_vs_11_max_abs"] = d
    for name, df, pp in (("shl", shl, p), ("es", es, E.saved(mid, 0, es, "p"))):
        ld = E.leg_data(df)
        for rn, rules in (("approved", spec.rules), ("raw", L.RAW)):
            _, sc = E.leg_eval(ld, pp, rules)
            out.update({f"{mid}_{name}_{rn}_{k}": float(sc.metrics[k]) for k in
                        ("edits_per_h", "leg_f1", "time_acc", "wrong_unflagged_time", "pred_legs", "truth_legs")})
(HERE / "logs" / f"check_move_{which}.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if "p_vs" in k or "edits" in k}, indent=1))
