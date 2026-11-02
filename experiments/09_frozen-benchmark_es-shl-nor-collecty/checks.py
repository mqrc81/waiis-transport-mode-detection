"""Stop-point 2 checks on waiis_bench_v2 (README: Plan, Checks and Comparability; stop-point 1 answers, items 2, 8).

    uv run python checks.py counts      # class x dataset x platform (x SHL position), users, top-user shares, flags
    uv run python checks.py identify    # dataset identification within class (05's 580 features), dev users only
    uv run python checks.py platform    # sign conversion leaves 05's features unchanged; platform AUC within class
    uv run python checks.py rescore     # 05 / 06 final configurations on the six-class score (logged CMs, no retraining)
    uv run python checks.py run         # ExtraSensory run labels (descriptive)

Tables to results/tables/checks/; full log in logs/checks_<name>.log; stdout gets a short summary. Reserved test users
are read only where a check is about them (counts; rescore and run on test use the logged confusion matrices and v1's
own test labels), never for any fitting.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tmd import acc_features as af
from tmd import bench
from tmd.labels import CLASSES, FINE_CLASSES, SCORE6, SCORE6_OF
from tmd.paths import PROCESSED

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "tables" / "checks"
V1, V2 = "waiis_bench_v1", "waiis_bench_v2"
SEED = 0
log = logging.getLogger("checks")


def table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv")
    log.info(f"\n== {name}\n{df.to_string()}")
    return df


def official(source: str, columns=None) -> pd.DataFrame:
    """Official examples of a v2 source (hips only for SHL by construction), with the fine class."""
    cols = ["dataset", "source_platform", "split", "fold", "cls_fine", "position"] + (columns or [])
    return bench.examples(source, cols, version=V2)


# ---------------------------------------------------------------- counts

def counts() -> None:
    rows, users = [], []
    for s in bench.sources(V2):
        e = official(s)
        for (sp, pl, c), g in e.groupby(["split", "source_platform", "cls_fine"]):
            rows.append({"source": s, "split": sp, "platform": pl, "class": c, "windows": len(g),
                         "users": g.user.nunique()})
        u = e.groupby(["cls_fine", "user"]).size()
        for c, g in u.groupby(level=0):
            users.append({"source": s, "class": c, "users": len(g), "users_ge5": int((g >= 5).sum()),
                          "top_user_share": round(g.max() / g.sum(), 3)})
    ct = pd.DataFrame(rows)
    table(ct, "counts_long")
    piv = ct.pivot_table(index=["source", "split", "platform"], columns="class", values="windows", aggfunc="sum",
                         fill_value=0).reindex(columns=FINE_CLASSES, fill_value=0)
    piv["all"] = piv.sum(axis=1)
    table(piv, "counts_class_source_platform")
    table(pd.DataFrame(users).pivot_table(index=["source", "class"], values=["users", "users_ge5", "top_user_share"]),
          "users_per_class")
    # SHL per position (official hips + other positions with a single-label window), Users 2-3 per user
    pos = []
    for s in ("shl_user1", "shl_user23"):
        lab = official(s)
        o = bench.read(s, "accelerometer_other_positions", columns=["user", "ts", "position", "cls", "acc_implausible"],
                       version=V2)
        o = o[o.cls.ne("") & ~o.acc_implausible]
        both = pd.concat([lab[["user", "ts", "cls_fine"]].assign(position="hips"),
                          o.rename(columns={"cls": "cls_fine"})[["user", "ts", "cls_fine", "position"]]],
                         ignore_index=True)
        both["position"] = both.position.str.lower()
        pos.append(pd.crosstab([both.position], both.cls_fine).assign(source=s).reset_index())
    table(pd.concat(pos).set_index(["source", "position"]), "shl_positions")
    # top-user shares before / after caps (new sources: from the build's cap tables)
    caps = pd.concat([pd.read_csv(HERE / "results" / "tables" / "bench_v2" / f"{s}_caps.csv") for s in ("nor_tmd", "collecty")])
    table(caps.set_index(["dataset", "class"]), "caps")
    # plausibility flag rates per class and source (on windows with a class, before the flags)
    fl = []
    for s in bench.sources(V2):
        lab = bench.read(s, "labels", columns=["cls_fine", "has_acc", "acc_implausible", "quiet", "label_implausible",
                                               "split"], version=V2)
        lab = lab[lab.has_acc & lab.cls_fine.ne("")]
        g = lab.groupby("cls_fine")[["acc_implausible", "quiet", "label_implausible"]].mean().mul(100).round(2)
        fl.append(g.assign(source=s, windows=lab.groupby("cls_fine").size()).reset_index())
    table(pd.concat(fl).set_index(["source", "cls_fine"]), "flag_rates")
    print(piv.to_string())


# ---------------------------------------------------------------- dataset / platform identification

def _features(source: str, keep: pd.DataFrame, version: str = V2) -> tuple[pd.DataFrame, np.ndarray]:
    k, w = bench.load_windows(source, keep[["user", "ts"]].assign(_k=1), version)
    return k, af.features(w, sub=True, grav=False)


def _sample(e: pd.DataFrame, n: int, rng) -> pd.DataFrame:
    """At most n windows, spread over users (round-robin over a seeded user order)."""
    if len(e) <= n:
        return e
    e = e.sample(frac=1, random_state=int(rng.integers(1 << 31)))
    return e.groupby("user").head(max(1, n // e.user.nunique())).head(n)


def _group_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, float, int]:
    """Cross-validated identification by group (StratifiedGroupKFold, LightGBM; groups = users, SHL User 1's days):
    macro one-vs-rest AUC and balanced accuracy; folds = min(5, groups of the smallest dataset)."""
    import lightgbm as lgb
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    k = int(min(5, pd.Series(groups).groupby(y).nunique().min()))
    if k < 2:
        return np.nan, np.nan, k
    classes = np.unique(y)
    p = np.zeros((len(y), len(classes)))
    for tr, te in StratifiedGroupKFold(k, shuffle=True, random_state=SEED).split(X, y, groups):
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, class_weight="balanced",
                               random_state=SEED, verbose=-1, n_jobs=8)
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])
    auc = roc_auc_score(y, p, multi_class="ovr", average="macro") if len(classes) > 2 else roc_auc_score(y, p[:, 1])
    return float(auc), float(balanced_accuracy_score(y, classes[p.argmax(1)])), k


DATASET_CHECKS = {"bus": ["extrasensory", "nor_tmd", "collecty", "shl_user1"],
                  "car": ["extrasensory", "nor_tmd", "collecty", "shl_user1"],
                  "train": ["shl_user1", "nor_tmd", "collecty", "extrasensory"],
                  "subway": ["shl_user1", "nor_tmd"],
                  "tram": ["nor_tmd", "collecty"]}
PER_SOURCE = 2000


def identify() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for cls, srcs in DATASET_CHECKS.items():
        parts = []
        for s in srcs:
            e = official(s, ["day"] if s.startswith("shl") else None)
            e = e[(e.split == "dev") & (e.cls_fine == cls)]
            # groups for the cross-validation: users; SHL User 1 is one person, so its recording days (its day folds)
            e["group"] = e.user if not s.startswith("shl") else "day" + e.day.astype(str)
            if e.group.nunique() < 2:
                log.info(f"{cls}/{s}: {e.group.nunique()} groups, left out")
                continue
            e = _sample(e, PER_SOURCE, rng)
            k, X = _features(s, e)
            k = k.merge(e[["user", "ts", "group"]], on=["user", "ts"])
            parts.append((s, k.group.to_numpy(), X))
        if len(parts) < 2:
            continue
        X = np.concatenate([p[2] for p in parts])
        y = np.concatenate([np.full(len(p[2]), i) for i, p in enumerate(parts)])
        g = np.concatenate([[f"{p[0]}:{u}" for u in p[1]] for p in parts])
        auc, bacc, k = _group_auc(X, y, g)
        rows.append({"class": cls, "datasets": " vs ".join(p[0] for p in parts),
                     "windows": " / ".join(str(len(p[2])) for p in parts),
                     "groups": " / ".join(str(len(set(p[1]))) for p in parts), "cv_folds_by_group": k,
                     "macro_auc": round(auc, 3), "balanced_accuracy": round(bacc, 3),
                     "chance_bacc": round(1 / len(parts), 3)})
        log.info(rows[-1])
    print(table(pd.DataFrame(rows).set_index("class"), "dataset_identification").to_string())


def platform() -> None:
    """ExtraSensory: v2 windows (iPhone sign-converted) give 05's features identical to v1's; platform identification
    within class (still, walk, car, bike), dev users, v1 vs v2 features."""
    rng = np.random.default_rng(SEED)
    rows = []
    for cls in ("still", "walk", "car", "bike"):
        e = official("extrasensory", ["platform"])
        e = e[(e.split == "dev") & (e.cls_fine == cls)]
        e = pd.concat([_sample(g, PER_SOURCE, rng) for _, g in e.groupby("platform")])
        k2, X2 = _features("extrasensory", e, V2)
        k1, X1 = _features("extrasensory", e, V1)
        assert (k1.user.to_numpy() == k2.user.to_numpy()).all() and (k1.ts.to_numpy() == k2.ts.to_numpy()).all()
        same = bool(np.array_equal(np.nan_to_num(X1, nan=-9e9), np.nan_to_num(X2, nan=-9e9)))
        y = (k2.merge(e[["user", "ts", "platform"]], on=["user", "ts"]).platform == "iphone").to_numpy().astype(int)
        auc2, bacc2, kk = _group_auc(X2, y, k2.user.to_numpy())
        auc1, _, _ = _group_auc(X1, y, k1.user.to_numpy())
        rows.append({"class": cls, "windows": len(y), "iphone_share": round(y.mean(), 3), "features_identical": same,
                     "platform_auc_v1": round(auc1, 3), "platform_auc_v2": round(auc2, 3), "cv_folds_by_user": kk})
        log.info(rows[-1])
    print(table(pd.DataFrame(rows).set_index("class"), "platform_identification").to_string())


# ---------------------------------------------------------------- rescoring 05 / 06 on six classes

RUNS = {"05": ("05_acc-baseline_waiis-bench", "F_fb", "test"), "06": ("06_gps-acc-baseline_waiis-bench", "H1_v2", "test")}
CM_KEYS = {"dev ES pooled OOF": "cm_es_class", "dev SHL User 1 (day folds)": "cm_shl_class",
           "test ES all": "cm_test_es_all_class", "test ES iPhone": "cm_test_es_iphone_class",
           "test ES Android": "cm_test_es_android_class", "test SHL Users 2-3": "cm_test_shl23_class"}


def _merge_cm(cm: np.ndarray, classes: list[str], groups: list[str], of: dict) -> tuple[np.ndarray, list[str]]:
    present = [g for g in groups if any(of[c] == g for c in classes)]
    M = np.zeros((len(present), len(present)))
    for i, a in enumerate(classes):
        for j, b in enumerate(classes):
            M[present.index(of[a]), present.index(of[b])] += cm[i][j]
    return M, present


def _macro_f1(M: np.ndarray) -> float:
    """Macro-F1 over the classes with support (rows > 0) of a square confusion matrix (rows true, columns predicted)."""
    M = np.asarray(M, float)
    tp = np.diag(M)
    prec = np.divide(tp, M.sum(0), out=np.zeros_like(tp), where=M.sum(0) > 0)
    rec = np.divide(tp, M.sum(1), out=np.zeros_like(tp), where=M.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1[M.sum(1) > 0].mean())


def rescore() -> None:
    import mlflow
    from tmd.paths import MLFLOW_DB
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    client = mlflow.MlflowClient()
    rows = []
    for tag, (exp, dev_row, test_row) in RUNS.items():
        e = client.get_experiment_by_name(exp)
        for seed in (0, 1, 2):
            for row in (dev_row, test_row):
                r = mlflow.search_runs([e.experiment_id], filter_string=f"tags.mlflow.runName = '{row}_seed{seed}'",
                                       output_format="list")
                if not r:
                    continue
                t = json.load(open(client.download_artifacts(r[0].info.run_id, "tables.json")))
                for name, key in CM_KEYS.items():
                    if key not in t or (row == dev_row) != name.startswith("dev"):
                        continue
                    classes = t[f"{key[3:]}_classes"].split(",")
                    cm = np.asarray(t[key], float)
                    m6, c6 = _merge_cm(cm, classes, SCORE6, SCORE6_OF)
                    rows.append({"model": tag, "seed": seed, "set": name, "six_class_f1": _macro_f1(m6),
                                 "eight_class_f1": _macro_f1(cm), "classes_6": ",".join(c6),
                                 **{f"recall_{c}": (m6[i, i] / m6[i].sum() if m6[i].sum() else np.nan)
                                    for i, c in enumerate(c6)}})
    r = pd.DataFrame(rows)
    table(r, "rescore_per_seed")
    summ = r.groupby(["model", "set"]).agg(six_class_f1=("six_class_f1", "mean"), eight_class_f1=("eight_class_f1", "mean"),
                                           seeds=("seed", "nunique"))
    # dev from the saved out-of-fold predictions (06's dev runs logged no confusion matrix as data, only figures):
    # seeds 0-2, ES dev and SHL User 1 official examples, user-bootstrap CI for ES (seed 0)
    from tmd.labels import to_score6
    sets = {"dev ES pooled OOF": ("extrasensory", "dev"), "dev SHL User 1 (day folds)": ("shl_user1", "dev")}
    oof = []
    for tag, (exp, dev_row, _) in RUNS.items():
        for seed in (0, 1, 2):
            a = np.load(PROCESSED / exp / "oof" / f"{dev_row}_seed{seed}.npz", allow_pickle=True)
            p = pd.DataFrame({"user": a["user"], "ts": a["ts"], "pred": a["p"].argmax(1) + 1})
            for name, (src, sp) in sets.items():
                ex = bench.examples(src, ["split", "y"], version=V1)
                m = ex[ex.split == sp].merge(p, on=["user", "ts"], how="inner")
                y6, p6 = to_score6(m.y.to_numpy()), to_score6(m.pred.to_numpy())
                r = {"model": tag, "seed": seed, "set": name, "six_class_f1_oof": _macro_f1(_cm(y6, p6)),
                     "eight_class_f1_oof": _macro_f1(_cm(m.y.to_numpy(), m.pred.to_numpy())), "examples": len(m)}
                if seed == 0 and src == "extrasensory":
                    lo, hi = _boot6(y6, p6, m.user.to_numpy())
                    r["six_class_ci_seed0"] = f"[{lo:.3f}, {hi:.3f}]"
                oof.append(r)
    oof = pd.DataFrame(oof)
    table(oof, "rescore_dev_oof_per_seed")
    o = oof.groupby(["model", "set"]).agg(six_class_f1_oof=("six_class_f1_oof", "mean"),
                                          eight_class_f1_oof=("eight_class_f1_oof", "mean"),
                                          six_class_ci_seed0=("six_class_ci_seed0", "first"))
    summ = summ.join(o, how="outer")
    print(table(summ.round(4), "rescore_six_class").to_string())


def _cm(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Square confusion matrix over the union of true and predicted labels."""
    labels = sorted(set(y) | set(pred))
    return pd.crosstab(y, pred).reindex(index=labels, columns=labels, fill_value=0).to_numpy()


def _boot6(y: np.ndarray, pred: np.ndarray, users: np.ndarray, n_boot: int = 1000) -> tuple[float, float]:
    """User-bootstrap 95% CI of six-class macro-F1 (resamples lacking a class of y are redrawn, as in
    tmd.evaluate.user_resamples)."""
    from tmd.evaluate import user_resamples
    vals = [_macro_f1(_cm(y[idx], pred[idx])) for idx in user_resamples(y, users, n_boot=n_boot, seed=SEED)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------------------------------------------------------- run labels

def _cadence(w: np.ndarray) -> np.ndarray:
    """Dominant frequency [Hz] of |acc| in 0.5-4 Hz per window (step cadence for walk / run)."""
    from tmd.resample import FS
    n = np.linalg.norm(w, axis=-1)
    n = n - n.mean(1, keepdims=True)
    spec = np.abs(np.fft.rfft(n, axis=1))
    f = np.fft.rfftfreq(n.shape[1], 1 / FS)
    band = (f >= 0.5) & (f <= 4.0)
    return f[band][spec[:, band].argmax(1)]


def run() -> None:
    """ExtraSensory run (dev and test) vs SHL User 1 run: windows per user, motion energy, step cadence; where 05 / 06's
    out-of-fold predictions put dev run minutes, per user. v1's labels (the benchmark the models were scored on)."""
    rows, per_user = [], []
    for s, sp in (("extrasensory", "dev"), ("extrasensory", "test"), ("shl_user1", "dev")):
        e = bench.examples(s, ["split", "cls", "platform", "acc_std", "acc_median"], version=V1)
        e = e[(e.split == sp) & (e.cls == "run")]
        k, w = bench.load_windows(s, e[["user", "ts"]], V1)
        k = k.merge(e, on=["user", "ts"])
        k["cadence_hz"] = _cadence(w)
        k["std_acc"] = np.linalg.norm(w, axis=-1).std(1)
        for u, g in k.groupby("user"):
            per_user.append({"source": f"{s} {sp}", "user": u[:8], "platform": g.platform.iloc[0], "windows": len(g),
                             "std_acc_median": round(g.std_acc.median(), 2), "cadence_hz_median": round(g.cadence_hz.median(), 2),
                             "cadence_share_2_3.5hz": round(g.cadence_hz.between(2, 3.5).mean(), 2)})
        rows.append({"set": f"{s} {sp}", "windows": len(k), "users": k.user.nunique(),
                     "std_acc_p25_p50_p75": " / ".join(f"{v:.2f}" for v in k.std_acc.quantile([.25, .5, .75])),
                     "cadence_hz_p25_p50_p75": " / ".join(f"{v:.2f}" for v in k.cadence_hz.quantile([.25, .5, .75]))})
    table(pd.DataFrame(rows).set_index("set"), "run_energy_cadence")
    pu = pd.DataFrame(per_user)
    # where 05 / 06's out-of-fold predictions put dev run minutes, per user (after the HMM)
    e = bench.examples("extrasensory", ["split", "cls"], version=V1)
    e = e[(e.split == "dev") & (e.cls == "run")]
    for tag, (exp, row, _) in RUNS.items():
        a = np.load(PROCESSED / exp / "oof" / f"{row}_seed0.npz", allow_pickle=True)
        p = pd.DataFrame({"user": a["user"], "ts": a["ts"], "pred": np.array(CLASSES)[a["p"].argmax(1)]})
        m = e.merge(p, on=["user", "ts"])
        share = pd.crosstab(m.user.str[:8], m.pred, normalize="index").round(2)
        share.columns = [f"{tag}_{c}" for c in share.columns]
        pu = pu.merge(share, left_on="user", right_index=True, how="left")
    print(table(pu.set_index(["source", "user"]), "run_per_user").to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("check", choices=["counts", "identify", "platform", "rescore", "run"])
    a = ap.parse_args()
    (HERE / "logs").mkdir(exist_ok=True)
    logging.basicConfig(filename=HERE / "logs" / f"checks_{a.check}.log", filemode="w", level=logging.INFO,
                        format="%(message)s")
    {"counts": counts, "identify": identify, "platform": platform, "rescore": rescore, "run": run}[a.check]()


if __name__ == "__main__":
    main()
