"""Stop-point 2, step 2: the leg-rule grid per model (E2, E1), dev only.

Minute probabilities: SHL User 1 with the chosen 20 s GPS rule (gps_candidates.py), ExtraSensory dev with its own
per-minute GPS (10's saved out-of-fold `p`). Grid (11's Plan): s_veh {0,2,5,10} x s_other {0,1,2,3} x g {2,10} x
l_min {1,2,3,5} x c {off,0.5,...,0.9} = 768. Objective: fewest edits per hour on SHL User 1 (seed 0) subject to
(a) SHL time with a wrong, unflagged mode <= 5% and (b) ExtraSensory leg F1 and time-weighted accuracy (top-1 modes)
not more than 0.01 below the raw HMM legs. Ties within 1% of the fewest edits: smaller s_veh, s_other, l_min, g, then
the higher c. Then tau2: the highest of {0.3, 0.2, 0.1} that shows the true mode for >= 50% of SHL legs whose top-1
is wrong. The chosen rules are then scored on seeds 1-2.

    uv run python leg_grid.py           # log: logs/leg_grid.log; MLflow one run per model
"""
import itertools
import json
import logging
import time

import mlflow
import numpy as np
import pandas as pd

import dev
from tmd import legs as L
from tmd.progress import Progress
from tmd.tracking import DataInput, run

log = logging.getLogger()
GRID = {"s_veh": [0, 2, 5, 10], "s_other": [0, 1, 2, 3], "g": [2, 10], "l_min": [1, 2, 3, 5]}
CS = [None, 0.5, 0.6, 0.7, 0.8, 0.9]
TAUS = [0.3, 0.2, 0.1]
A_MAX, B_TOL, TIE = 0.05, 0.01, 0.01
TABLES = dev.HERE / "results" / "tables"
KEEP = ["leg_f1", "leg_f1_top1", "time_acc", "time_acc_top1", "edits_per_h", "edits_per_day", "merges", "splits",
        "mode_changes", "boundary_moves", "boundary_median_min", "boundary_p90_min", "extra_legs_per_h",
        "missed_legs_per_h", "wrong_unflagged_time", "wrong_unflagged_legs", "flagged_leg_share",
        "flagged_time_share", "pred_legs", "truth_legs", "short_leg_recall", "range_top1", "range_top2", "range_ece",
        "range_brier", "top1_wrong_legs"] + [f"second_shows_truth_tau{t:g}" for t in TAUS] + \
    [f"second_shown_share_tau{t:g}" for t in TAUS]


def probs(model: str, seed: int, shl: pd.DataFrame, es: pd.DataFrame, rule: str) -> dict:
    return {"shl": np.load(dev.OUT / f"shl_user1_{model}_seed{seed}_{rule}_p.npy").astype(np.float64),
            "es": dev.saved(model, seed, es, "p").astype(np.float64)}


def evaluate(d: dev.LegData, p9: np.ndarray, base: L.Rules) -> dict:
    """Metrics for every threshold c on one structural rule set (c only relabels legs)."""
    q = L.top_probs(p9[d.order])
    lg = L.build(d.seq, d.ts, q, base)
    tl = L.truth_legs(d.seq, d.ts, d.truth)
    out = {}
    for c in CS:
        lc = lg.copy()
        if c is not None:
            lc["mode"] = np.where(lc.p1 < c, L.UNKNOWN, lc.top1)
        sc = L.score(lc, tl, len(q), q, d.day)
        out[c] = sc
    return out


def choose(grid: pd.DataFrame, raw_es: dict) -> pd.Series:
    ok = (grid.shl_wrong_unflagged_time <= A_MAX) & \
         (grid.es_leg_f1_top1 >= raw_es["leg_f1_top1"] - B_TOL) & \
         (grid.es_time_acc_top1 >= raw_es["time_acc_top1"] - B_TOL)
    grid = grid.assign(ok_a=grid.shl_wrong_unflagged_time <= A_MAX, ok=ok)
    feas = grid[ok]
    if feas.empty:
        return None
    best = feas.shl_edits_per_h.min()
    near = feas[feas.shl_edits_per_h <= best * (1 + TIE)].copy()
    near["c_key"] = -near.c.fillna(0.0)
    return near.sort_values(["s_veh", "s_other", "l_min", "g", "c_key"]).iloc[0]


def main():
    logging.basicConfig(filename=dev.HERE / "logs" / "leg_grid.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    rule = json.loads((TABLES / "gps_choice.json").read_text())["chosen"]
    shl, es = dev.frame("shl_user1", gps=False), dev.frame("extrasensory", gps=False)
    data = {"shl": dev.leg_data(shl), "es": dev.leg_data(es)}
    combos = list(itertools.product(*GRID.values()))
    prog = Progress(dev.EXP, "leg_grid", len(dev.MODELS) * (len(combos) + 2), log)
    summary = {}
    for name in dev.MODELS:
        t0 = time.time()
        p0 = probs(name, 0, shl, es, rule)
        raw = {k: L.score(*_raw(data[k], p0[k])).metrics for k in data}
        rows = []
        for combo in combos:
            base = L.Rules(**dict(zip(GRID, combo)))
            res = {k: evaluate(data[k], p0[k], base) for k in data}
            for c in CS:
                r = {**dict(zip(GRID, combo)), "c": c}
                for k in data:
                    r.update({f"{k}_{m}": res[k][c].metrics[m] for m in KEEP})
                rows.append(r)
            prog.step(f"{name} {base.name()}")
        grid = pd.DataFrame(rows)
        TABLES.mkdir(parents=True, exist_ok=True)
        grid.to_csv(TABLES / f"leg_grid_{name}.csv", index=False)
        best = choose(grid, raw["es"])
        if best is None:
            log.info("%s: no combination meets (a) and (b)", name)
            print(f"{name}: no combination meets (a) and (b)")
            summary[name] = None
            continue
        rules = L.Rules(int(best.s_veh), int(best.s_other), int(best.g), int(best.l_min),
                        None if pd.isna(best.c) else float(best.c))
        tau = next((t for t in TAUS if best[f"shl_second_shows_truth_tau{t:g}"] >= 0.5), None)
        # chosen rules and raw legs on every seed, with per-mode tables
        per_seed, modes = [], []
        for seed in dev.SEEDS:
            p = p0 if seed == 0 else probs(name, seed, shl, es, rule)
            for tag, rr in (("raw", L.RAW), ("chosen", rules)):
                for k in data:
                    lg, sc = dev.leg_eval(data[k], p[k], rr)
                    per_seed.append({"model": name, "seed": seed, "legs": tag, "data": k, **sc.metrics})
                    if seed == 0:
                        modes.append(sc.per_mode.assign(model=name, legs=tag, data=k))
            if seed:
                prog.step(f"{name} seed {seed}")
        ps = pd.DataFrame(per_seed)
        ps.to_csv(TABLES / f"leg_chosen_seeds_{name}.csv", index=False)
        pm = pd.concat(modes, ignore_index=True)
        pm.to_csv(TABLES / f"leg_per_mode_{name}.csv", index=False)
        n_ok_a, n_ok = int((grid.shl_wrong_unflagged_time <= A_MAX).sum()), int(len(grid))
        summary[name] = {"rules": rules.name(), "tau2": tau, "gps_rule": rule, "grid_rows": n_ok,
                         "grid_rows_meeting_a": n_ok_a, "raw_es": raw["es"], "raw_shl": raw["shl"],
                         "chosen": best.to_dict()}
        prm = {"model": name, "dataset": "shl_user1 (hips, day folds) + extrasensory dev", "rate_hz": 25,
               "class_set": "top level (still, walking, running, bike, car, bus, rail)", "seed": 0,
               "gps_rule_shl": rule, "grid": json.dumps({**GRID, "c": CS}), "objective": "min SHL edits/h",
               "constraint_a": f"SHL wrong unflagged time <= {A_MAX}",
               "constraint_b": f"ES leg F1 / time acc (top-1) >= raw - {B_TOL}",
               "chosen_rules": rules.name(), "tau2": str(tau), "step": "leg_grid"}
        kw = {"label_rules": "bench-v2 (9 fine classes) -> tool top level", "window_s": 15, "stride_s": 60,
              "oof_eval": True}
        ds = [DataInput("shl_user1_dev", "evaluation", features=f"acc-v1-B + gps-v2 ({rule}, 20 s)", **kw),
              DataInput("extrasensory_dev_v2", "evaluation", features="acc-v1-B + gps-v2 (per minute)", **kw)]
        with run(dev.EXP, prm, run_name=f"leg_grid_{name}", datasets=ds):
            met = {}
            for _, r in ps.iterrows():
                for m in KEEP:
                    v = r.get(m)
                    if v == v and v is not None:
                        met[f"{r['data']}_{r['legs']}_seed{r['seed']}_{m}"] = float(v)
            met["grid_rows_meeting_a"] = n_ok_a
            met["run_seconds"] = time.time() - t0
            mlflow.log_metrics(met)
            mlflow.log_artifact(str(TABLES / f"leg_grid_{name}.csv"))
            mlflow.log_artifact(str(TABLES / f"leg_per_mode_{name}.csv"))
        log.info("%s chosen %s tau2 %s; %s", name, rules.name(), tau, json.dumps(
            {k: round(v, 4) for k, v in best.items() if isinstance(v, float)}))
        show = ps[ps.seed == 0].set_index(["data", "legs"])[["leg_f1", "leg_f1_top1", "time_acc", "time_acc_top1",
                                                              "edits_per_h", "wrong_unflagged_time",
                                                              "flagged_leg_share", "boundary_median_min"]]
        print(f"{name}: chosen {rules.name()}, tau2 {tau}; grid rows meeting (a): {n_ok_a} of {n_ok}")
        print(show.round(4).to_string())
    (TABLES / "leg_choice.json").write_text(json.dumps(summary, indent=1, default=float))


def _raw(d: dev.LegData, p9: np.ndarray):
    q = L.top_probs(p9[d.order])
    lg = L.build(d.seq, d.ts, q, L.RAW)
    return lg, L.truth_legs(d.seq, d.ts, d.truth), len(q), q, d.day


if __name__ == "__main__":
    main()
