"""Experiment 07: every Results table and the figure, from the exported runs only (results/runs.csv).

    uv run python tables.py                       # results/tables.md, results/curve.png
    uv run python tables.py --csv <path>          # another export (e.g. a scratch one while runs go on)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ORDER = ["0", "1", "2", "3", "5", "10", "20", "all"]
CLS7 = ["still", "walk", "run", "bike", "car", "bus", "train"]
FLAT = 0.01                         # flattens: first n whose median gain per doubling to the next point < 0.01
POOL_MEAN = np.mean([38, 37, 37, 37, 39])
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"   # reference palette


def x_of(n: str) -> float:
    return float(POOL_MEAN) if n == "all" else float(n)


def load(path: Path) -> pd.DataFrame:
    r = pd.read_csv(path)
    r = r[(r["dirty"].astype(str).str.lower() == "false") & r.es_class_f1.notna()].copy()   # clean, finished
    assert not r.duplicated(["curve", "n_users", "draw"]).any(), "one clean run per (curve, n, draw)"
    r["n"] = r["n_users"].astype(str)
    r["draw"] = r["draw"].astype(int)
    return r


def summary(r: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per n: median, min, max over draws, and the number of runs."""
    g = r.groupby("n")[col].agg(["median", "min", "max", "count"])
    return g.reindex([n for n in ORDER if n in g.index])


def fmt(v, nd=3) -> str:
    return "–" if v != v else f"{v:.{nd}f}"


def cell(s: pd.Series, nd=3) -> str:
    if s["count"] == 1:
        return fmt(s["median"], nd)
    return f"{fmt(s['median'], nd)} [{fmt(s['min'], nd)}–{fmt(s['max'], nd)}]"


def gains(r: pd.DataFrame, col: str) -> pd.DataFrame:
    """Gain per doubling between consecutive grid points n >= 1, per draw (nested draws: paired), median over
    draws. n = all is one run, compared with each draw's n = 20."""
    w = r[r.n != "0"].pivot_table(index="draw", columns="n", values=col)
    alln = r.loc[r.n == "all", col]
    ns = [n for n in ORDER[1:] if n in set(r.n)]
    rows = []
    for a, b in zip(ns, ns[1:]):
        va = w[a].dropna()
        vb = pd.Series(float(alln.iloc[0]), index=va.index) if b == "all" else w[b].reindex(va.index)
        g = (vb - va) / np.log2(x_of(b) / x_of(a))
        rows.append({"step": f"{a} → {b}", "from": a, "median": g.median(), "min": g.min(), "max": g.max()})
    return pd.DataFrame(rows)


def slope(r: pd.DataFrame, col: str) -> tuple[float, float, float]:
    """Least-squares slope of col against log2 n over n >= 1, per draw (n = all shared), median / min / max."""
    alln = float(r.loc[r.n == "all", col].iloc[0])
    out = []
    for d, g in r[(r.n != "0") & (r.n != "all")].groupby("draw"):
        x = np.log2([x_of(n) for n in g.n] + [POOL_MEAN])
        y = np.r_[g[col].to_numpy(), alln]
        out.append(np.polyfit(x, y, 1)[0])
    return float(np.median(out)), float(min(out)), float(max(out))


def flattens(gt: pd.DataFrame) -> str:
    below = gt[gt["median"] < FLAT]
    return "not within the dev pool" if below.empty else f"n = {below.iloc[0]['from']}"


def tables(r: pd.DataFrame) -> str:
    main, noshl, esc = (r[r.curve == c] for c in ("main", "noshl", "escheck"))
    L = []
    # F1 per n
    L += ["**8-class and family macro-F1 per n** (pooled ExtraSensory dev out-of-fold; median [range] over 5 draws; "
          "n = 0 and n = all: one run, with the user-bootstrap 95% CI)", "",
          "| n | 8-class, with SHL User 1 | 8-class, without | family, with | family, without | per-fold 8-class "
          "(with), min–max | 8-class before HMM (with) |", "|---|---|---|---|---|---|---|"]
    s = {(c, k): summary(d, k) for c, d in (("main", main), ("noshl", noshl))
         for k in ("es_class_f1", "es_family_f1", "es_class_f1_fold_min", "es_class_f1_fold_max", "es_class_f1_raw")}

    def ci(d, n, k):
        row = d[d.n == n]
        if len(row) == 1 and f"{k}_lo" in row and row[f"{k}_lo"].notna().all():
            return f" ({fmt(row[f'{k}_lo'].iloc[0])}–{fmt(row[f'{k}_hi'].iloc[0])})"
        return ""
    for n in ORDER:
        cells = []
        for k in ("es_class_f1", "es_family_f1"):
            for cur, d in (("main", main), ("noshl", noshl)):
                t = s[(cur, k)]
                cells.append(cell(t.loc[n]) + ci(d, n, k) if n in t.index else "–")
        fm = s[("main", "es_class_f1_fold_min")]
        fx = s[("main", "es_class_f1_fold_max")]
        cells.append(f"{fmt(fm.loc[n, 'median'])}–{fmt(fx.loc[n, 'median'])}" if n in fm.index else "–")
        rw = s[("main", "es_class_f1_raw")]
        cells.append(cell(rw.loc[n]) if n in rw.index else "–")
        L.append(f"| {n} | " + " | ".join(cells) + " |")
    L += ["", "Per-fold: median over draws of the lowest and highest fold score (spread only)."]
    # gains per doubling
    for cur, d in (("main", main), ("noshl", noshl)):
        if not (d.n == "all").any():
            continue
        L += ["", f"**Gain per doubling, 8-class and family F1 — {'with' if cur == 'main' else 'without'} SHL User 1** "
              f"(median [range] over draws; n = all counted as {POOL_MEAN:.1f} users)", "",
              "| step | 8-class | family |", "|---|---|---|"]
        g8, gf = gains(d, "es_class_f1"), gains(d, "es_family_f1")
        for (_, a), (_, b) in zip(g8.iterrows(), gf.iterrows()):
            L.append(f"| {a['step']} | {cell({**a, 'count': 5})} | {cell({**b, 'count': 5})} |")
        sl8, slf = slope(d, "es_class_f1"), slope(d, "es_family_f1")
        L += ["", f"- Least-squares slope over n ≥ 1 per doubling: 8-class {fmt(sl8[0])} [{fmt(sl8[1])}–{fmt(sl8[2])}], "
              f"family {fmt(slf[0])} [{fmt(slf[1])}–{fmt(slf[2])}].",
              f"- **Flattens (8-class, rule fixed before running): {flattens(g8)}**; family: {flattens(gf)}."]
    # per-class recall
    for cur, d in (("main", main), ("noshl", noshl)):
        L += ["", f"**Recall per class — {'with' if cur == 'main' else 'without'} SHL User 1** (median [range] over "
              "draws)", "", "| n | " + " | ".join(CLS7) + " |", "|---" * (len(CLS7) + 1) + "|"]
        for n in ORDER:
            if n not in set(d.n):
                continue
            L.append(f"| {n} | " + " | ".join(cell(summary(d, f'es_class_recall_{c}').loc[n], 2) for c in CLS7) + " |")
    # still gaining at the end
    L += ["", "**Recall gain per doubling over the last step (20 → all), main curve** (a class is still gaining if the "
          "median ≥ 0.01; F1 per class alongside)", "", "| class | recall | class F1 |", "|---|---|---|"]
    for c in CLS7:
        gr, gfc = gains(main, f"es_class_recall_{c}").iloc[-1], gains(main, f"es_class_f1_{c}").iloc[-1]
        flag = " **still gaining**" if gr["median"] >= FLAT else ""
        L.append(f"| {c} | {cell({**gr, 'count': 5})}{flag} | {cell({**gfc, 'count': 5})} |")
    # training composition
    L += ["", "**Training data per n** (ExtraSensory official minutes per class, mean over folds; median over draws; "
          "absent = training sets without the class, of 5 folds × draws)", "",
          "| n | iPhone share | " + " | ".join(CLS7) + " | absent: run / bike / bus / train | folds without a GPS "
          "model (no-SHL) |", "|---" * (len(CLS7) + 4) + "|"]
    for n in ORDER[1:]:
        d = main[main.n == n]
        if d.empty:
            continue
        mins = " | ".join(f"{d[f'train_es_minutes_{c}'].median():,.0f}" for c in CLS7)
        ab = " / ".join(f"{int(d[f'train_absent_folds_{c}'].sum())}" for c in ("run", "bike", "bus", "train"))
        nf = noshl.loc[noshl.n == n, "folds_without_gps_model"].sum() if "folds_without_gps_model" in noshl else 0
        L.append(f"| {n} | {d.iphone_share_mean.median():.2f} | {mins} | {ab} | {int(nf)} |")
    # early-stopping check
    if not esc.empty:
        L += ["", "**Early-stopping check** (users n+1 … n+5 as validation; paired difference against the fixed-rounds "
              "run of the same draw; median [range] over 5 draws)", "",
              "| n | Δ 8-class | Δ family | 8-class, early stopping | 8-class, fixed | acc rounds (fold mean) | "
              "GPS rounds (fold mean) |", "|---|---|---|---|---|---|---|"]
        for n in ("2", "10"):
            d = esc[esc.n == n]
            if d.empty:
                continue
            cs = [cell(summary(d, k).loc[n]) for k in ("es_class_f1_diff_vs_fixed", "es_family_f1_diff_vs_fixed",
                                                       "es_class_f1", "es_class_f1_fixed")]
            cs += [cell(summary(d, k).loc[n], 0) for k in ("acc_rounds_mean", "gps_rounds_mean")]
            L.append(f"| {n} | " + " | ".join(cs) + " |")
    # cost
    L += ["", "**Cost** (main curve, draw 0, fold-0 models; single thread, after warm-up; model sizes mean over folds "
          "and draws)", "", "| n | single window p50 / p95 ms | pipeline per window p50 / p95 ms | acc model MB | "
          "GPS model MB |", "|---|---|---|---|---|"]
    for n in ORDER:
        d = main[main.n == n]
        if d.empty:
            continue
        l0 = d[d.draw == 0].iloc[0]
        L.append(f"| {n} | {l0.get('lat_inference_ms_p50', np.nan):.1f} / {l0.get('lat_inference_ms_p95', np.nan):.1f} | "
                 f"{l0.get('lat_pipeline_ms_p50', np.nan):.2f} / {l0.get('lat_pipeline_ms_p95', np.nan):.2f} | "
                 f"{d.acc_bytes_mean.mean() / 1e6:.2f} | {d.gps_bytes_mean.mean() / 1e6:.2f} |")
    return "\n".join(L) + "\n"


XPOS = {n: (-1.0 if n == "0" else np.log2(x_of(n))) for n in ORDER}


def _line(a, s, col, ms, label=None):
    """Median line with the draw range as a band; n = 0 (left of the axis break) as a separate point."""
    s1 = s.drop(index="0", errors="ignore")
    x = [XPOS[n] for n in s1.index]
    a.fill_between(x, s1["min"], s1["max"], color=col, alpha=0.15, lw=0)
    a.plot(x, s1["median"], color=col, lw=2, marker="o", ms=ms, mec="white", mew=1.5, label=label)
    if "0" in s.index:
        a.plot([XPOS["0"]], [s.loc["0", "median"]], color=col, marker="o", ms=ms, mec="white", mew=1.5, ls="")


def figure(r: pd.DataFrame, out: Path):
    xpos = XPOS
    ticks = [xpos[n] for n in ORDER]
    labels = ["0\n(SHL\nonly)", "1", "2", "3", "5", "10", "20", f"all\n(~{POOL_MEAN:.0f})"]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlesize": 10, "axes.titlecolor": INK})
    fig = plt.figure(figsize=(15, 5.2), facecolor="white")
    gs = fig.add_gridspec(2, 7, width_ratios=[2.2, 2.2, 0.35, 1, 1, 1, 1], wspace=0.35, hspace=0.55)
    ax = fig.add_subplot(gs[:, :2])

    def style(a, small=False):
        a.set_xticks(ticks if not small else [xpos[n] for n in ("0", "1", "10", "all")])
        a.set_xticklabels(labels if not small else ["0", "1", "10", "all"])
        a.grid(axis="y", color=GRID, lw=0.8)
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)

    for cur, col, name in (("main", BLUE, "with SHL User 1"), ("noshl", ORANGE, "without SHL User 1")):
        s = summary(r[r.curve == cur], "es_class_f1")
        x = [xpos[n] for n in s.index]
        _line(ax, s, col, 5, name)
        ax.annotate(f"{name}  {s['median'].iloc[-1]:.3f}", (x[-1], s["median"].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=8)
    ax.axvline(-0.5, color=GRID, lw=1, ls=":")
    style(ax)
    ax.set_xlim(-1.4, xpos["all"] + 2.6)
    ax.set_xlabel("labelled ExtraSensory users n (log scale)")
    ax.set_ylabel("8-class macro-F1, pooled dev out-of-fold")
    ax.set_title("8-class F1 against labelled users (median, range over 5 draws)", loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    # per-class recall, small multiples (main curve)
    main = r[r.curve == "main"]
    for i, c in enumerate(CLS7):
        a = fig.add_subplot(gs[i // 4, 3 + i % 4])
        s = summary(main, f"es_class_recall_{c}")
        x = [xpos[n] for n in s.index]
        _line(a, s, BLUE, 3.5)
        a.set_ylim(0, 1)
        style(a, small=True)
        a.set_title(c if c != "train" else "train (2–3 ES users)", loc="left", fontsize=9)
    fig.text(0.55, 0.97, "Recall per class, with SHL User 1 (median, range over draws)", color=INK, fontsize=10,
             va="top")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=HERE / "results" / "runs.csv")
    a = ap.parse_args()
    r = load(a.csv)
    (HERE / "results").mkdir(exist_ok=True)
    md = tables(r)
    (HERE / "results" / "tables.md").write_text(md)
    figure(r, HERE / "results" / "curve.png")
    print(md)


if __name__ == "__main__":
    main()
