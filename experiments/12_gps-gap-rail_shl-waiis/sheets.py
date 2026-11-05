"""Review sheets for the three WAIIS days: diary vs E1 + R1 (engine as in tmd.suggest.SPECS: k = 1, R1 gap legs),
E2 (11's setting, no gap rule) and the package's own predictions; 11's `review.sheet`, every lane
clipped to the diary's span +- 30 min. Personal output under
<processed>/12_.../waiis/review/ (never in git).

    uv run python sheets.py
"""
import dataclasses
import sys

import numpy as np

from tmd import legs as L
from tmd import suggest as S
from tmd.models.e10 import PROCESSED
from tmd.readers import waiis

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "11_leg-suggestions_waiis-app-days"))
import review as R  # noqa: E402

PAD_S = 1800.0
OUT = PROCESSED / "12_gps-gap-rail_shl-waiis" / "waiis" / "review"
SPECS = {"E1 + R1": S.SPECS["E1"], "E2": dataclasses.replace(S.SPECS["E2"], gap=None)}


def named(lg):
    return lg.assign(mode=["unknown" if m < 0 else L.MODES[m] for m in lg["mode"]], conf=lg.p1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for day in waiis.days():
        dia = R.diary(day.date)
        d_in = S.day_input(day)
        lanes = {"diary": dia.assign(hint=dia.sub_mode, conf=np.nan, source="diary")}
        for name, spec in SPECS.items():
            lanes[name] = named(S.suggest(spec, d_in))
        lanes["package"] = R.package_legs(day.path).assign(conf=np.nan)
        lo, hi = dia.start.min() - PAD_S, dia.end.max() + PAD_S          # the diary's span +- 30 min
        lanes = {k: v[(v.end > lo) & (v.start < hi)].assign(start=lambda d: d.start.clip(lo, hi),
                                                             end=lambda d: d.end.clip(lo, hi)) for k, v in lanes.items()}
        g = waiis.fix_states(waiis.read_gps(day))
        g = g[g.accuracy_m <= waiis.COARSE_M]
        mins = d_in.minutes.assign(kind=np.where(d_in.minutes.covered, "acc+gps", "acc"))
        n_gap = int((lanes["E1 + R1"].source == "rule:gps_gap_moving").sum())
        R.sheet(OUT / f"{day.date}_sheet.png", day.date, lanes, mins, g,
                f"WAIIS {day.date}: diary vs E1 + R1 (hatched unknown 'underground_rail' = GPS gap leg) vs E2 (11) "
                f"vs the package's {R.PKG_MODEL} (labels: mode, hint, confidence)")
        print(day.date, {k: len(v) for k, v in lanes.items()}, "gap legs", n_gap)


if __name__ == "__main__":
    main()
