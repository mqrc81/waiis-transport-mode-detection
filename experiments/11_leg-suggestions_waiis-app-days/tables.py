"""Final Results tables of experiment 11, generated from the exported runs (`results/runs.csv`) only.

    uv run python tables.py > results/tables_from_runs.md
"""
from pathlib import Path

import pandas as pd

r = pd.read_csv(Path(__file__).resolve().parent / "results" / "runs.csv")
md = lambda d: "```\n" + d.round(3).to_string() + "\n```"       # noqa: E731

g = r[r.step == "gps_candidates"]
t = g.pivot_table(index="gps_rule", columns="seed", values="six_f1")
t["mean"] = t.mean(1)
t = t.join(g.groupby("gps_rule")[["e1_six_f1", "six_f1_before_hmm", "gps_alone_six_f1_covered", "covered_official",
                                  "rawlegs_edits_per_h"]].mean())
print("## GPS candidates (SHL User 1, 20 s; six-class F1 after fusion + HMM)\n\n" + md(t) + "\n")

lg = r[r.step == "leg_grid"].set_index("model")
rows = []
for m in ("E2", "E1"):
    for data in ("shl", "es"):
        for legs in ("raw", "chosen"):
            k = f"{data}_{legs}_seed0_"
            rows.append({"model": m, "data": data, "legs": legs, **{c: lg.at[m, k + c] for c in
                         ("edits_per_h", "leg_f1", "leg_f1_top1", "time_acc", "wrong_unflagged_time",
                          "flagged_leg_share", "range_top1", "range_top2", "range_ece")}})
print("## Leg grid, seed 0 (raw HMM legs vs the objective's choice)\n\n" + md(pd.DataFrame(rows).set_index(
    ["model", "data", "legs"])) + "\n")

rf = r[r.step == "refine"].set_index("model")[["minute_boundary_median_min", "refined_boundary_median_min",
                                               "minute_edits_per_h", "refined_edits_per_h", "moved"]]
print("## 20 s boundary refinement (SHL User 1, 1 s truth)\n\n" + md(rf) + "\n")

w = r[r.step == "waiis_days"].set_index(["date", "model"]).sort_index()[
    ["minutes", "legs_model", "legs_gps_only_still", "legs_gps_only_moving", "legs_unknown_low_confidence",
     "diary_matched", "diary_legs", "extra_legs", "run_seconds_total"]]
print("## WAIIS days (strict matching, IoU >= 0.5)\n\n" + md(w) + "\n")

le = r[r.step == "lenient_diary_comparison"].copy()
le["model"] = le.model.str.split(" ").str[0]
cols = [f"{g}_{c}" for g in ("vehicle", "other") for c in ("legs", "right", "found", "wrong", "no_data",
                                                          "core_minutes_with_data", "core_minute_accuracy")]
print("## WAIIS days, lenient comparison (core minutes)\n\n" + md(le.set_index("model")[cols + ["extra_vehicle_legs"]].T)
      + "\n")
pm = [f"{md_}_{c}" for md_ in ("bike", "car", "bus", "rail", "walking", "running") for c in ("legs", "right", "no_data")]
print("### per mode\n\n" + md(le.set_index("model")[pm].T) + "\n")

s = r[r.step == "shl23_once"].set_index("model")
c = [x for x in r.columns if x.startswith(("minute_", "c1_")) and x.split("_")[1] in ("hips", "bag", "torso", "hand")
     and x.endswith(("_leg_f1", "_edits_per_h", "_wrong_unflagged_time", "_leg_recall_rail"))]
print("## SHL Users 2-3 (once)\n\n" + md(s[sorted(c)].T) + "\n")
