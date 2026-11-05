"""Final Results tables of experiment 12, generated from the exported runs (`results/runs.csv`) only.
(`k_per_class` in the export is tracking's numeric copy of the param `k`, here the boost factor.)

    uv run python tables.py > results/tables_from_runs.md
"""
from pathlib import Path

import pandas as pd

r = pd.read_csv(Path(__file__).resolve().parent / "results" / "runs.csv")
md = lambda d: "```\n" + d.round(4).to_string() + "\n```"
FR = ["false_rail_still", "false_rail_walk", "false_rail_car", "false_rail_bus"]

d = r[r.step == "dev_k"]
c = ["pooled_" + x for x in ("edits_per_h", "time_acc", "wrong_unflagged_time", "leg_f1", "subway_leg_recall",
                             "subway_hint_metro", "rail_leg_recall", "rail_leg_precision", "extra_rail_legs_per_h",
                             *FR)]
print("## k grid, SHL User 1 all positions pooled, seed 0 (the choice)\n\n" +
      md(d[d.seed == 0].set_index(["model", "k"])[c + ["es_false_rail_nonrail"]]) + "\n")
print("## k grid, mean of seeds 0-2\n\n" + md(d.groupby(["model", "k"])[c + ["es_false_rail_nonrail",
                                                                             "es_false_rail_min"]].mean()) + "\n")
print("### edits per hour by seed\n\n" + md(d.pivot_table(index=["model", "seed"], columns="k",
                                                          values="pooled_edits_per_h")) + "\n")
pc = [f"{p}_{x}" for p in ("hips", "hand", "torso", "bag") for x in ("edits_per_h", "subway_leg_recall", "time_acc")]
print("### per position, seed 0\n\n" + md(d[d.seed == 0].set_index(["model", "k"])[pc].T) + "\n")

s = r[r.step == "sensitivity"]
print("## Gap-setting sensitivity at k = 2, seed 0\n\n" + md(s.set_index(["model", "gap_g_min_s", "gap_v_lo", "gap_v_hi"])[
    ["pooled_edits_per_h", "pooled_subway_leg_recall", "pooled_extra_rail_legs_per_h", "pooled_false_rail_car",
     "pooled_false_rail_bus", "es_false_rail_nonrail"]]) + "\n")

h = r[r.step == "sp3_shl23"].set_index("model")
rows = []
for m in h.index:
    for pos in ("pooled", "hips", "hand", "torso", "bag"):
        for v in ("off", "on"):
            rows.append({"model": m, "position": pos, "rule": v, **{x: h.at[m, f"{v}_{pos}_{x}"] for x in
                         ("edits_per_h", "time_acc", "leg_f1", "subway_leg_recall", "rail_leg_recall",
                          "rail_leg_precision", "extra_rail_legs_per_h", *FR, "in_gap_minutes")}})
print("## SHL Users 2-3 (once), rule off / on\n\n" + md(pd.DataFrame(rows).set_index(["model", "position", "rule"]))
      + "\n")

e = r[r.step == "sp3_es_test_guard"].set_index("model")
rows = [{"model": m, "part": p, "rule": v, **{x: e.at[m, f"{v}_{p}_{x}"] for x in
         ("false_rail_nonrail", "false_rail_min", *FR, "extra_rail_legs_per_h", "edits_per_h", "in_gap_minutes")}}
        for m in e.index for p in ("all", "iphone", "android") for v in ("off", "on")]
print("## ExtraSensory test users (once): false rail, rule off / on\n\n" +
      md(pd.DataFrame(rows).set_index(["model", "part", "rule"])) + "\n")

w = r[r.step == "sp3_waiis_lenient"].set_index(["model", "rule", "gap_rule"]).sort_index()
wc = [f"{g}_{x}" for g in ("vehicle", "other") for x in ("legs", "right", "found", "wrong", "no_data",
                                                        "core_minute_accuracy")]
wc += ["rail_legs", "rail_right", "rail_no_data", "rail_no_data_with_gap_hint", "rail_no_data_gap_leg_share_mean",
       "rail_with_gap_hint", "extra_vehicle_legs", "extra_rail_legs", "extra_gap_legs", "run_seconds_per_day_mean",
       "run_seconds_per_day_max"]
print("## WAIIS days, lenient comparison (11's), rule off / on\n\n" + md(w[wc].T) + "\n")
