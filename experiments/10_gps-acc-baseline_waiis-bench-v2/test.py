"""Stop-point 3: the test, once (protocol in the README, "Test protocol"). Per seed 0-2: final models on all dev data
(rounds = the median of the seed's 5 out-of-fold early-stopping counts), scored on the reserved users.

- E2 (final) and A0 (reference): ExtraSensory test (overall, iPhone, Android), NOR-TMD test, SHL Users 2-3 per position.
- E1 (descriptive; can't change the final configuration): NOR-TMD test and SHL Users 2-3 only.

    uv run python make_features.py --part test      # once, after the protocol is committed
    uv run python test.py                           # logs/test.log; results/tables/test/
Refuses to run if any test_* run exists in MLflow.
"""
from __future__ import annotations

import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")

import json
import logging

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

import pipe
from tmd.evaluate import model_size_bytes
from tmd.labels import FINE_CLASSES
from tmd.models import fusion, lgbm, minutes
from tmd.progress import Progress
from tmd.tracking import TRACKING_URI, DataInput, run

ES, SHL, NOR, COL = "extrasensory", "shl_user1", "nor_tmd", "collecty"
CONFIGS = {"A0": pipe.Row("A0", (ES, SHL), 8),                      # first: E2's and E1's paired differences
           "E2": pipe.Row("E2", (ES, SHL, COL), 8),
           "E1": pipe.Row("E1", (ES, SHL, NOR), 9, hmm_sources=(ES, SHL), stop_no_tram=True)}
SCORED = {"E2": ("es", "nor", "shl23"), "A0": ("es", "nor", "shl23"), "E1": ("nor", "shl23")}
SEEDS = (0, 1, 2)
TEST_DIR = pipe.HERE / "results" / "tables" / "test"
log = logging.getLogger()


def rounds(name: str, seed: int) -> int:
    rows = json.loads((pipe.MODELS / f"{name}_seed{seed}_folds.json").read_text())
    return int(np.median([r["rounds"] for r in rows]))


def train_final(dev, cols, trainable, sw, n, lr, seed, r) -> lgb.Booster:
    prm = {**lgbm.PARAMS, "learning_rate": lr, "num_class": n, "seed": seed}
    tr = trainable & (dev.fold.to_numpy() >= 0)
    return lgb.train(prm, lgbm.dataset(dev[cols].to_numpy(np.float32)[tr], dev.y.to_numpy()[tr], sw[tr], n=n), r)


def support(d: pd.DataFrame) -> dict:
    return {c: {"users": int(d.user[d.y == i + 1].nunique()), "windows": int((d.y == i + 1).sum())}
            for i, c in enumerate(FINE_CLASSES) if (d.y == i + 1).any()}


def score(name, test, p, p_raw, base, parts) -> tuple[dict, dict, dict]:
    """Every test set of the protocol. base: A0's p for the paired differences (None for A0)."""
    m, tab, figs = {}, {}, {}
    src, off, users = test.source.to_numpy(), test.official.to_numpy(), test.user.to_numpy()
    tr, pr = pipe.truth(test.y.to_numpy()), pipe.preds(p)
    bpr = pipe.preds(base) if base is not None else None
    bases = (lambda sel, lvl: {"A0": bpr[lvl][sel]}) if base is not None else (lambda sel, lvl: None)
    if "es" in parts:
        for plat in ("all", "iphone", "android"):
            sel = (src == ES) & off & ((test.platform.to_numpy() == plat) if plat != "all" else True)
            for lvl in pipe.LEVELS:
                pipe.block(m, tab, figs, f"test_es_{plat}", tr[lvl][sel], pr[lvl][sel], lvl, users[sel],
                           bases(sel, lvl), fig=plat == "all")
            for part, mk in (("covered", test.covered.to_numpy()), ("uncovered", ~test.covered.to_numpy())):
                s2 = sel & mk
                pipe.block(m, tab, figs, f"test_es_{plat}_{part}", tr["six"][s2], pr["six"][s2], "six", ci=False,
                           fig=False)
        tab["test_es_support"] = support(test[(src == ES) & off])
        walk = (src == ES) & off & (test.y.to_numpy() == 2)
        m["test_es_walk_to_still"] = float((pr["fine"][walk] == 1).sum())
        m["test_es_tram_pred_usable_n"] = float((pr["fine"][src == ES] == pipe.TRAM).sum())
    if "nor" in parts:
        sel = (src == NOR) & off
        g = sel & np.isin(tr["six"], [4, 5, 6])
        pipe.block(m, tab, figs, "test_nor_guard", tr["six"][g], pr["six"][g], "six", users[g], bases(g, "six"),
                   present=np.isin(np.arange(1, 7), [4, 5, 6]))
        pipe.block(m, tab, figs, "test_nor", tr["fine"][sel], pr["fine"][sel], "fine", users[sel], bases(sel, "fine"))
        for c in ("bus", "subway", "train", "tram", "car"):        # tram 0 where the model has no tram class
            k = f"test_nor_fine_recall_{c}"
            m.setdefault(k, 0.0)
        tab["test_nor_support"] = support(test[sel])
        tram = sel & (test.y.to_numpy() == pipe.TRAM)
        tab["test_nor_tram_predicted_as"] = pd.Series(np.array(FINE_CLASSES)[pr["fine"][tram] - 1]).value_counts() \
            .to_dict()
        for c, v in tab["test_nor_tram_predicted_as"].items():
            m[f"test_nor_tram_as_{c}"] = float(v)
    if "shl23" in parts:
        for pos in ("hips", "hand", "torso", "bag"):
            s = "shl_user23" if pos == "hips" else pipe.TEST_OTHER
            sel = (src == s) & off & (test.position.to_numpy() == pos)
            for lvl in pipe.LEVELS:
                pipe.block(m, tab, figs, f"test_shl23_{pos}", tr[lvl][sel], pr[lvl][sel], lvl, ci=False,
                           base_prs=None if base is None else {"A0": bpr[lvl][sel]}, fig=False)
            pre = pipe.preds(p_raw)["fine"][sel]                   # before the HMM
            for i, c in ((6, "train"), (7, "subway")):
                y = tr["fine"][sel] == i + 1
                if y.any():
                    m[f"test_shl23_{pos}_fine_recall_{c}_before_hmm"] = float((pre[y] == i + 1).mean())
        m["test_shl23_tram_pred_usable_n"] = float((pr["fine"][np.isin(src, ["shl_user23", pipe.TEST_OTHER])]
                                                     == pipe.TRAM).sum())
    return m, tab, figs


def main():
    (pipe.HERE / "logs").mkdir(exist_ok=True)
    logging.basicConfig(filename=pipe.HERE / "logs" / "test.log", filemode="a", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    mlflow.set_tracking_uri(TRACKING_URI)
    done = mlflow.search_runs(experiment_names=[pipe.EXP], filter_string="tags.mlflow.runName LIKE 'test_%'")
    if len(done):
        raise SystemExit(f"test runs exist already ({len(done)}): the test is scored once")
    dev, test = pipe.load("dev"), pipe.load("test")
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    prog = Progress(pipe.EXP, "test", len(SEEDS) * (1 + len(CONFIGS)), log)
    summary = []
    for seed in SEEDS:
        a0 = CONFIGS["A0"]
        tr_g, sw_g = pipe.gps_trainable(dev, a0)
        r_gps = rounds("A0_gps", seed)
        gps = train_final(dev, pipe.GPS, tr_g, sw_g, 8, pipe.GPS_FIT[0], seed, r_gps)
        gps.save_model(str(pipe.MODELS / f"test_gps_seed{seed}.txt"))
        p_gps = gps.predict(test[pipe.GPS].to_numpy(np.float32))
        prog.step(f"seed {seed} GPS model ({r_gps} rounds)")
        probs = {}
        for name, row in CONFIGS.items():
            tr_a, sw_a = pipe.acc_trainable(dev, row)
            r_acc = rounds(name, seed)
            acc = train_final(dev, pipe.ACC, tr_a, sw_a, row.n, pipe.ACC_FIT[0], seed, r_acc)
            acc.save_model(str(pipe.MODELS / f"test_{name}_seed{seed}.txt"))
            lab = pipe.labelled(row.hmm(), row.n)
            a, pi = minutes.hmm_params(lab[lab.fold >= 0], row.n)
            np.savez(pipe.MODELS / f"test_{name}_seed{seed}_hmm.npz", a=a, pi=pi)
            raw = fusion.product(acc.predict(test[pipe.ACC].to_numpy(np.float32)), p_gps, test.covered.to_numpy())
            p = pipe.pad9(minutes.smooth(test, raw, a, pi, "fb"))
            probs[name] = p
            pipe.save(f"test_{name}", seed, test, raw=raw, p=p)
            m, tab, figs = score(name, test, p, pipe.pad9(raw), probs.get("A0") if name != "A0" else None,
                                 SCORED[name])
            m.update({"acc_rounds": r_acc, "gps_rounds": r_gps, "acc_model_bytes": model_size_bytes(acc),
                      "gps_model_bytes": model_size_bytes(gps)})
            if name in ("E2", "A0"):
                pool = fusion.session_pool(test[test.source == ES].assign(fold=0), seed, version=pipe.BENCH)
                m.update(fusion.latency(pool, acc, gps, a, pi, pipe.GPS_VERSION))
            kw = {"label_rules": "bench-v2 (9 fine classes)", "features": pipe.datasets(row)[0].features,
                  "rate_hz": 25, "window_s": 15, "stride_s": 60}
            names = {ES: "extrasensory_dev_v2", SHL: "shl_user1_dev", NOR: "nor_tmd_dev", COL: "collecty_dev"}
            inputs = [DataInput(names[s], "training", **kw) for s in row.sources]
            inputs += [DataInput({"es": "extrasensory_test_v3", "nor": "nor_tmd_test_v3",
                                  "shl23": "shl_user23_test_v3"}[k], "evaluation", **kw) for k in SCORED[name]]
            prm = {"model": "acc lightgbm x gps-v2 lightgbm, product, hmm-fb", "row": f"test_{name}", "seed": seed,
                   "rate_hz": 25, "dataset": "+".join(row.sources), "class_set": f"{row.n}class (six-class primary)",
                   "config": name, "role": {"E2": "final", "A0": "reference", "E1": "descriptive"}[name],
                   "acc_rounds": r_acc, "gps_rounds": r_gps, "hmm_sources": ",".join(row.hmm()),
                   "scored": ",".join(SCORED[name]), "benchmark": pipe.BENCH}
            with run(pipe.EXP, prm, run_name=f"test_{name}_seed{seed}", datasets=inputs):
                mlflow.log_metrics({k: float(v) for k, v in m.items() if v == v})
                for fn, fig in figs.items():
                    mlflow.log_figure(fig, fn)
                    plt.close(fig)
                mlflow.log_dict(pipe.dump(tab), "tables.json")
            (TEST_DIR / f"test_{name}_seed{seed}.json").write_text(json.dumps(pipe.dump(tab), indent=1))
            log.info("test %s seed %s %s", name, seed, json.dumps(m, default=float))
            summary.append({"config": name, "seed": seed, **{k: v for k, v in m.items() if not k.endswith(("_lo", "_hi"))}})
            prog.step(f"seed {seed} {name}")
            keys = ["test_es_all_six_f1", "test_es_iphone_six_f1", "test_es_android_six_f1", "test_es_all_six_f1_diff_A0",
                    "test_nor_guard_six_f1", "test_nor_guard_six_f1_diff_A0", "test_shl23_hips_six_f1"]
            print(name, seed, {k: round(m[k], 4) for k in keys if k in m}, flush=True)
    s = pd.DataFrame(summary)
    s.to_csv(TEST_DIR / "test_per_seed.csv", index=False)
    s.groupby("config").mean(numeric_only=True).T.to_csv(TEST_DIR / "test_mean_of_seeds.csv")


if __name__ == "__main__":
    main()
