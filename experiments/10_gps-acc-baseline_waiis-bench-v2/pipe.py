"""Experiment 10: shared pieces of the rows (frame, training, fusion, HMM, evaluation, shortcut checks, latency).

Frame rows, in a fixed order: the usable dev minutes of ExtraSensory, SHL User 1 (hips), NOR-TMD, Collecty, then SHL
User 1's other-position windows (`shl_user1_other`, one row per position and minute; scored as unseen positions,
trained on only in P). Every row of the ablation predicts every frame row out of fold (fold k's model scores fold k),
so a source a row doesn't train on gets a transfer score. Probabilities per row and seed in `oof/<row>_seed<k>.npz`
(acc = acc model before fusion, gps = GPS model, raw = fused before the HMM, p = after; p always 9 columns).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from make_features import GPS_VERSION, OUT, acc_path, gps_path
from tmd import bench
from tmd.evaluate import (bootstrap_cms, class_list, cm_metrics, family_proba, identification_auc, plot_cm,
                          six_pred)
from tmd.features import acc as af
from tmd.features import gps as gf
from tmd.labels import FAMILIES, FINE_CLASSES, SCORE6, to_family, to_score6
from tmd.models import fusion, lgbm, minutes
from tmd.tracking import DataInput

BENCH = "waiis_bench_v2"
HERE = Path(__file__).resolve().parent
EXP = HERE.name
OOF, MODELS = OUT / "oof", OUT / "models"
ACC = af.names(True, False)                     # 05's B: 580 features
GPS = gf.names(GPS_VERSION)                     # gps-v2: 13
SOURCES = ("extrasensory", "shl_user1", "nor_tmd", "collecty")
OTHER = "shl_user1_other"
GPS_SOURCES = ("extrasensory", "shl_user1")     # the GPS model's training sources (hips only for SHL)
ACC_FIT = 0.1, 1000, 30                         # 06's acc model: lr, max rounds, patience
GPS_FIT = 0.02, 10000, 150                      # 06's G_v2
N_BOOT = 1000
LEVELS = ("six", "fine", "family")
NAMES = {"six": SCORE6, "fine": FINE_CLASSES, "family": FAMILIES}
TRAM = 9


@dataclass
class Row:
    name: str
    sources: tuple[str, ...]                    # acc model's training sources; HMM transitions from the same
    n: int = 9                                  # classes (A0: 8)
    shl_weight: float = 1.0                     # SHL User 1's source weight, both models (S: 0.5)
    positions: bool = False                     # P: SHL User 1 trains on the drawn position per minute
    gps_from: str | None = None                 # reuse this row's GPS model (same GPS training data), else train
    vs: list[str] = field(default_factory=list)  # rows for paired differences (A0 and the best so far)
    hmm_sources: tuple[str, ...] | None = None  # E1 / E2: HMM transitions from these sources (default: `sources`)
    stop_no_tram: bool = False                  # E1 / E2: tram windows left out of the early-stopping fold

    def hmm(self) -> tuple[str, ...]:
        return self.hmm_sources or self.sources


# ---------------------------------------------------------------- frame

def load() -> pd.DataFrame:
    d = []
    for s in SOURCES + (OTHER,):
        a = pd.read_parquet(acc_path(s, "dev"))
        if s in GPS_SOURCES + (OTHER,):
            g = pd.read_parquet(gps_path(s, "dev"))
            assert (a.user.to_numpy() == g.user.to_numpy()).all() and (a.ts.to_numpy() == g.ts.to_numpy()).all()
            a = pd.concat([a, g[GPS]], axis=1)
        else:
            a = a.assign(**{c: np.float32(np.nan) for c in GPS})
        if s == "shl_user1":
            lab = bench.read(s, "labels", columns=["user", "ts", "position_draw"], version=BENCH)
            a = a.merge(lab, on=["user", "ts"], how="left", validate="1:1").assign(position="hips")
        elif s == OTHER:
            a = a.assign(session=a.session.astype(str) + "/" + a.position)    # one HMM sequence per position
        else:
            a = a.assign(position="", position_draw=True)
        d.append(a.assign(source=s))
    df = pd.concat(d, ignore_index=True)
    df["official"] = df.official.fillna(False).astype(bool)
    df["position_draw"] = df.position_draw.fillna(False).astype(bool)
    df["y"], df["fold"] = df.y.fillna(0).astype(int), df.fold.astype(int)
    df["covered"] = gf.covered(df[GPS].to_numpy(np.float32))
    return df


def labelled(sources, n: int = 9) -> pd.DataFrame:
    """All labelled minutes (9 fine classes) of the dev users of `sources`: the HMM's training data. n = 8: tram
    minutes are left out, i.e. unlabelled (E2 without a tram class)."""
    lab = pd.concat([minutes.labelled_minutes(s, BENCH, fine=True) for s in sources], ignore_index=True)
    return lab[lab.y8 <= n].reset_index(drop=True)


def save(name, seed, df, **arrays):
    OOF.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OOF / f"{name}_seed{seed}.npz", user=df.user.to_numpy(str), ts=df.ts.to_numpy(),
                        source=df.source.to_numpy(str), **arrays)


def load_probs(name, seed, df, key="p"):
    z = np.load(OOF / f"{name}_seed{seed}.npz", allow_pickle=True)
    assert (z["user"] == df.user.to_numpy(str)).all() and (z["ts"] == df.ts.to_numpy()).all() and \
        (z["source"] == df.source.to_numpy(str)).all(), "row order"
    return z[key]


def pad9(p: np.ndarray) -> np.ndarray:
    """8-class probabilities (A0) with a zero tram column."""
    return p if p.shape[1] == 9 else np.column_stack([p, np.zeros(len(p))])


# ---------------------------------------------------------------- training

def acc_trainable(df, row: Row) -> tuple[np.ndarray, np.ndarray]:
    """(trainable, source weight) of the acc model. P: SHL User 1 hips rows only where hips is the drawn position,
    plus the drawn other-position rows."""
    src = df.source.to_numpy()
    sources = set(row.sources) | ({OTHER} if row.positions else set())
    ok = np.isin(src, list(sources)) & df.official.to_numpy() & (df.fold.to_numpy() >= 0)
    if row.positions:
        ok &= df.position_draw.to_numpy() | ~np.isin(src, ["shl_user1", OTHER])
    sw = np.where(np.isin(src, ["shl_user1", OTHER]), row.shl_weight, 1.0)
    if row.n == 8:                              # tram minutes are unlabelled usable minutes (E2; none in A0)
        ok &= df.y.to_numpy() != TRAM
    return ok, sw


def gps_trainable(df, row: Row) -> tuple[np.ndarray, np.ndarray]:
    src = df.source.to_numpy()
    ok = np.isin(src, GPS_SOURCES) & df.official.to_numpy() & df.covered.to_numpy() & (df.fold.to_numpy() >= 0)
    return ok, np.where(src == "shl_user1", row.shl_weight, 1.0)


def fit(df, cols, trainable, sw, n, fit_cfg, seed, log, name, stop_no_tram=False, **hooks):
    lr, mr, pat = fit_cfg
    prm = {**lgbm.PARAMS, "learning_rate": lr, "num_class": n}
    stop_ok = (df.y.to_numpy() != TRAM) if stop_no_tram else None
    return lgbm.paired_cv(df[cols].to_numpy(np.float32), df.y.to_numpy(), df.fold.to_numpy(), trainable, sw, seed,
                          prm, mr, pat, log, name, stop_ok=stop_ok, **hooks)


def save_models(name, seed, res):
    MODELS.mkdir(parents=True, exist_ok=True)
    for k, m in enumerate(res["models"]):
        m.save_model(str(MODELS / f"{name}_seed{seed}_fold{k}.txt"))
    (MODELS / f"{name}_seed{seed}_folds.json").write_text(json.dumps(res["rows"]))


def weight_share(df, trainable, sw, n) -> pd.DataFrame:
    """Each source's share of the training weight per class (source weight x balanced class weight), summed over
    the training rows of the 5 outer folds. SHL User 1's other positions count as SHL User 1."""
    src = df.source.replace({OTHER: "shl_user1"}).to_numpy()
    y, fold = df.y.to_numpy(), df.fold.to_numpy()
    tot = []
    for k in range(lgbm.K):
        tr = trainable & (fold != k) & (fold >= 0)
        tot.append(pd.DataFrame({"source": src[tr], "cls": np.array(FINE_CLASSES)[y[tr] - 1],
                                 "w": lgbm.class_weights(y[tr], sw[tr], n)}))
    t = pd.concat(tot).pivot_table(index="cls", columns="source", values="w", aggfunc="sum", fill_value=0.0)
    return t.div(t.sum(1), axis=0).reindex([c for c in FINE_CLASSES if c in t.index])


# ---------------------------------------------------------------- HMM descriptives

def transitions(lab: pd.DataFrame) -> pd.DataFrame:
    """Per source: labelled 1-min steps (consecutive labelled minutes of one session, the HMM's own counting) and how
    many of them change class."""
    out = {}
    for s, d in lab.groupby("source"):
        d = d.sort_values(["user", "session", "ts"])
        same = (d.user.to_numpy()[1:] == d.user.to_numpy()[:-1]) & \
            (d.session.astype(str).to_numpy()[1:] == d.session.astype(str).to_numpy()[:-1])
        step = same & (np.abs(np.diff(d.ts.to_numpy(float)) - minutes.STEP_S) < 30.5)
        y = d.y8.to_numpy()
        out[s] = {"steps": int(step.sum()), "transitions": int((step & (y[1:] != y[:-1])).sum())}
    return pd.DataFrame(out).T


# ---------------------------------------------------------------- evaluation

def preds(p: np.ndarray) -> dict[str, np.ndarray]:
    """Predictions per level from 9-column probabilities: fine = argmax, six = argmax merged (09's rescoring),
    family = argmax of the summed probabilities (06's family)."""
    return {"fine": p.argmax(1) + 1, "six": six_pred(p), "family": family_proba(p).argmax(1) + 1}


def truth(y: np.ndarray) -> dict[str, np.ndarray]:
    return {"fine": y, "six": to_score6(y), "family": to_family(y)}


def _cm(y, pr, lvl):
    k = len(NAMES[lvl])
    c = np.zeros((k, k), np.int64)
    np.add.at(c, (y - 1, pr - 1), 1)
    return c


def block(m, tab, figs, pre, y, pr, lvl, users=None, base_prs=None, present=None, ci=True, fig=True):
    """Point metrics of one evaluation set and level (macro-F1 over `present` classes, default those in y; per-class
    recall and precision), user-bootstrap CIs and paired differences against base_prs ({name: predictions})."""
    names = NAMES[lvl]
    present = np.isin(np.arange(1, len(names) + 1), np.unique(y)) if present is None else present
    cm = _cm(y, pr, lvl)
    pt = cm_metrics(cm, present)
    key = f"{pre}_{lvl}"
    m[f"{key}_f1"] = pt["macro_f1"]
    tab[f"{key}_classes"] = ",".join(n for n, ok in zip(names, present) if ok)
    tab[f"{key}_cm"] = cm
    for i, c in enumerate(names):
        if cm[i].sum():
            m[f"{key}_recall_{c}"] = pt["recall"][i]
        if cm[:, i].sum():
            m[f"{key}_precision_{c}"] = pt["precision"][i]
    if fig:
        figs[f"cm_{key}.png"] = plot_cm(cm, lvl, key)
    if not ci:
        for b, bp in (base_prs or {}).items():
            m[f"{key}_f1_diff_{b}"] = pt["macro_f1"] - cm_metrics(_cm(y, bp, lvl), present)["macro_f1"]
        return
    bases = list((base_prs or {}).items())
    st = {}
    boot = bootstrap_cms(y, [pr] + [bp for _, bp in bases], users, lvl, N_BOOT, 0, st)
    m[f"{key}_ci_redraw_rate"] = st["redraw_rate"]

    def ci(vals):
        v = np.asarray(vals, float)
        v = v[~np.isnan(v)]
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)
    m[f"{key}_f1_lo"], m[f"{key}_f1_hi"] = ci([r[0]["macro_f1"] for r in boot])
    for i, c in enumerate(names):
        for stat in ("recall", "precision"):
            if f"{key}_{stat}_{c}" in m:
                m[f"{key}_{stat}_{c}_lo"], m[f"{key}_{stat}_{c}_hi"] = ci([r[0][stat][i] for r in boot])
    for j, (b, bp) in enumerate(bases, start=1):
        bcm = cm_metrics(_cm(y, bp, lvl), present)
        m[f"{key}_f1_diff_{b}"] = pt["macro_f1"] - bcm["macro_f1"]
        m[f"{key}_f1_diff_{b}_lo"], m[f"{key}_f1_diff_{b}_hi"] = ci([r[0]["macro_f1"] - r[j]["macro_f1"] for r in boot])
        for i, c in enumerate(names):
            if f"{key}_precision_{c}" in m:
                m[f"{key}_precision_{c}_diff_{b}"] = pt["precision"][i] - bcm["precision"][i]
                m[f"{key}_precision_{c}_diff_{b}_lo"], m[f"{key}_precision_{c}_diff_{b}_hi"] = \
                    ci([r[0]["precision"][i] - r[j]["precision"][i] for r in boot])


def evaluate(df, p, bases: dict[str, np.ndarray]):
    """Every evaluation set of the README's Plan. p, bases: (N, 9) after the HMM."""
    m, tab, figs = {}, {}, {}
    src, off = df.source.to_numpy(), df.official.to_numpy()
    pr, tr = preds(p), truth(df.y.to_numpy())
    bpr = {b: preds(bp) for b, bp in bases.items()}
    users, cov = df.user.to_numpy(), df.covered.to_numpy()

    # ExtraSensory dev (primary: six-class), CIs and paired differences by user
    sel = (src == "extrasensory") & off
    for lvl in LEVELS:
        y, q = tr[lvl][sel], pr[lvl][sel]
        block(m, tab, figs, "es", y, q, lvl, users[sel], {b: v[lvl][sel] for b, v in bpr.items()})
        for part, mk in (("covered", cov[sel]), ("uncovered", ~cov[sel])):
            m[f"es_{lvl}_f1_{part}"] = cm_metrics(_cm(y[mk], q[mk], lvl), np.isin(
                np.arange(1, len(NAMES[lvl]) + 1), np.unique(y[mk])))["macro_f1"]
        for plat in ("iphone", "android"):
            pm = (df.platform.to_numpy()[sel] == plat)
            m[f"es_{lvl}_f1_{plat}"] = cm_metrics(_cm(y[pm], q[pm], lvl), np.isin(
                np.arange(1, len(NAMES[lvl]) + 1), np.unique(y[pm])))["macro_f1"]
        fo = df.fold.to_numpy()[sel]
        per = [cm_metrics(_cm(y[fo == k], q[fo == k], lvl), np.isin(np.arange(1, len(NAMES[lvl]) + 1),
                                                                     np.unique(y[fo == k])))["macro_f1"]
               for k in range(lgbm.K)]
        m[f"es_{lvl}_f1_fold_min"], m[f"es_{lvl}_f1_fold_max"] = min(per), max(per)

    # NOR-TMD dev: the guard (bus, car, rail; six-class predictions), per class with CIs by installation
    sel = (src == "nor_tmd") & off
    y6 = tr["six"][sel]
    g = np.isin(y6, [4, 5, 6])
    gsel = np.flatnonzero(sel)[g]
    block(m, tab, figs, "nor_guard", tr["six"][gsel], pr["six"][gsel], "six", users[gsel],
          {b: v["six"][gsel] for b, v in bpr.items()}, present=np.isin(np.arange(1, 7), [4, 5, 6]))
    for lvl in ("six", "fine"):
        block(m, tab, figs, "nor", tr[lvl][sel], pr[lvl][sel], lvl, users[sel],
              {b: v[lvl][sel] for b, v in bpr.items()})

    # Collecty dev, SHL User 1 (hips, day folds): no CI
    for s, pre in (("collecty", "collecty"), ("shl_user1", "shl")):
        sel = (src == s) & off
        for lvl in LEVELS:
            block(m, tab, figs, pre, tr[lvl][sel], pr[lvl][sel], lvl,
                  base_prs={b: v[lvl][sel] for b, v in bpr.items()}, ci=False, fig=lvl != "family")

    # SHL User 1 other positions (held-out days, out of fold), per position
    for pos in ("hand", "torso", "bag"):
        sel = (src == OTHER) & off & (df.position.to_numpy() == pos)
        for lvl in ("six", "fine"):
            block(m, tab, figs, f"shlpos_{pos}", tr[lvl][sel], pr[lvl][sel], lvl,
                  base_prs={b: v[lvl][sel] for b, v in bpr.items()}, ci=False, fig=False)

    # tram where there is none (ExtraSensory, SHL): count and share, all usable minutes and official ones
    for s in ("extrasensory", "shl_user1", OTHER):
        for part, mk in (("usable", src == s), ("official", (src == s) & off)):
            t = pr["fine"][mk] == TRAM
            m[f"tram_pred_{s}_{part}_n"], m[f"tram_pred_{s}_{part}_share"] = float(t.sum()), float(t.mean())

    # (c) recall per source and class (fine), official rows
    rec = {}
    for s in SOURCES + (OTHER,):
        sel = (src == s) & off
        cm = _cm(tr["fine"][sel], pr["fine"][sel], "fine")
        rec[s] = {c: cm[i, i] / cm[i].sum() for i, c in enumerate(FINE_CLASSES) if cm[i].sum()}
        for c, r in rec[s].items():
            m[f"src_recall_{s}_{c}"] = r
    tab["recall_by_source"] = pd.DataFrame(rec).reindex(FINE_CLASSES).round(3).to_dict()
    return m, tab, figs


# ---------------------------------------------------------------- (d) source identification within class

ID_CHECKS = {"bus": ["extrasensory", "nor_tmd", "collecty", "shl_user1"],      # 09's DATASET_CHECKS, same order
             "car": ["extrasensory", "nor_tmd", "collecty", "shl_user1"],
             "train": ["shl_user1", "nor_tmd", "collecty", "extrasensory"],
             "subway": ["shl_user1", "nor_tmd"],
             "tram": ["nor_tmd", "collecty"]}
ID_PER_SOURCE = 2000
ID_BASELINE = Path(__file__).resolve().parents[1] / "09_frozen-benchmark_es-shl-nor-collecty" / "results" / \
    "tables" / "checks" / "dataset_identification.csv"


def _sample(e: pd.DataFrame, n: int, rng) -> pd.DataFrame:
    """09's checks.py `_sample`: at most n windows, spread over users (round-robin over a seeded user order)."""
    if len(e) <= n:
        return e
    e = e.sample(frac=1, random_state=int(rng.integers(1 << 31)))
    return e.groupby("user").head(max(1, n // e.user.nunique())).head(n)


def id_sample() -> dict[str, pd.DataFrame]:
    """09's exact windows (same calls, same seed): per class, (source, user, ts, group)."""
    rng = np.random.default_rng(0)
    out = {}
    for cls, srcs in ID_CHECKS.items():
        parts = []
        for s in srcs:
            cols = ["dataset", "source_platform", "split", "fold", "cls_fine", "position"] + \
                (["day"] if s.startswith("shl") else [])
            e = bench.examples(s, cols, version=BENCH)
            e = e[(e.split == "dev") & (e.cls_fine == cls)]
            e["group"] = e.user if not s.startswith("shl") else "day" + e.day.astype(str)
            if e.group.nunique() < 2:
                continue
            parts.append(_sample(e, ID_PER_SOURCE, rng)[["user", "ts", "group"]].assign(source=s))
        out[cls] = pd.concat(parts, ignore_index=True)
    return out


def identification(df, inputs: dict[str, np.ndarray], samples) -> pd.DataFrame:
    """Within-class source identification AUC from the model's out-of-fold log-probabilities, on 09's windows;
    `final_uncovered`: the final output on the sampled minutes without GPS only. 09's feature-level AUC alongside."""
    hips = (df.source != OTHER).to_numpy()
    idx = pd.Series(np.flatnonzero(hips), index=pd.MultiIndex.from_frame(df.loc[hips, ["source", "user", "ts"]]))
    base = pd.read_csv(ID_BASELINE).set_index("class").macro_auc
    rows = []
    for cls, smp in samples.items():
        ix = idx.reindex(pd.MultiIndex.from_frame(smp[["source", "user", "ts"]])).to_numpy()
        assert not np.isnan(ix.astype(float)).any(), f"{cls}: sampled windows missing from the frame"
        ix = ix.astype(int)
        y = pd.factorize(smp.source)[0]
        grp = (smp.source + ":" + smp.group.astype(str)).to_numpy()
        r = {"class": cls, "sources": " vs ".join(pd.unique(smp.source)), "windows": len(smp),
             "auc_features_09": base.get(cls, np.nan)}
        for nm, p in inputs.items():
            r[f"auc_{nm}"] = identification_auc(np.log(np.clip(p[ix], 1e-6, 1)), y, grp)[0]
        unc = ~df.covered.to_numpy()[ix]
        r["windows_uncovered"] = int(unc.sum())
        r["sources_uncovered"] = " vs ".join(pd.unique(smp.source[unc]))
        yy = pd.factorize(smp.source[unc])[0]
        r["auc_final_uncovered"] = identification_auc(np.log(np.clip(inputs["final"][ix][unc], 1e-6, 1)), yy,
                                                      grp[unc])[0] if len(set(yy)) > 1 else np.nan
        rows.append(r)
    return pd.DataFrame(rows).set_index("class")


# ---------------------------------------------------------------- MLflow inputs, latency

def datasets(row: Row) -> list[DataInput]:
    kw = {"label_rules": "bench-v2 (9 fine classes)", "features": f"acc-v1_whole_sub5s+{GPS_VERSION}_w2min",
          "rate_hz": 25, "window_s": 15, "stride_s": 60, "oof_eval": True}
    names = {"extrasensory": "extrasensory_dev_v2", "shl_user1": "shl_user1_dev", "nor_tmd": "nor_tmd_dev",
             "collecty": "collecty_dev"}
    return [DataInput(names[s], "training" if s in row.sources else "evaluation", **kw) for s in SOURCES]


def model_bytes(models) -> float:
    return float(np.mean([len(m.model_to_string()) for m in models]))


def dump(tab):
    def conv(o):
        if isinstance(o, pd.DataFrame):
            return o.round(4).to_dict()
        return o.tolist() if hasattr(o, "tolist") else str(o)
    return json.loads(json.dumps(tab, default=conv))
