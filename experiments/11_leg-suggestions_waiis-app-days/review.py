"""Stop-point 3: review of leg suggestions against the diaries (no labels; review days, not a test set).

Diary legs (`<processed>/11_.../waiis/diaries/<date>.csv`, local Europe/Madrid times, tool top level, rail sub-mode)
against each model's legs (model legs + GPS-only legs) and the package's own predictions:
- matched: a predicted leg of the same top-level mode with IoU >= 0.5 with the diary leg (11's Plan);
- also per diary leg: the share of its time under legs of the same mode, under any model leg (data coverage), and the
  dominant predicted mode;
- missed: diary legs not matched;
- extra: model legs of a moving mode (not still / unknown) with < 50% of their time inside diary legs of the same
  mode. On 2026-10-31 walking was not tracked, so walking / running legs there are "not in diary", not extra.
Tram check (added at the OK of stop-point 2): per diary tram leg and model, the minute shares of the argmax fine
class (tram / train / subway / bus / car / other), the GPS coverage of those minutes and the sub-mode hint.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tmd import legs as L
from tmd.paths import PROCESSED

TZ = "Europe/Madrid"
OUT = PROCESSED / "11_leg-suggestions_waiis-app-days" / "waiis"
DIARIES = OUT / "diaries"
WALK_UNTRACKED = {"2026-10-31"}
MOVING = {"walking", "running", "bike", "car", "bus", "rail"}
PKG_TOP = {"still": "still", "walking": "walking", "running": "running", "bike": "bike", "car": "car", "bus": "bus",
           "subway": "rail", "train": "rail", "tram": "rail"}
PKG_HINT = {"subway": "metro", "train": "train", "tram": "tram"}
PKG_MODEL = "fusion_late__hmm_viterbi"


def diary(date: str) -> pd.DataFrame:
    d = pd.read_csv(DIARIES / f"{date}.csv", dtype=str, keep_default_na=False)
    for c in ("start", "end"):
        t = pd.to_datetime(d.date + " " + d[f"{c}_local"]).dt.tz_localize(TZ)
        d[c] = (t - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)   # resolution-independent
    d["diary_id"] = [f"D{i + 1}" for i in range(len(d))]
    return d


def package_legs(day_path, model: str = PKG_MODEL) -> pd.DataFrame:
    """The package's 18 s rows mapped to the top level, joined into legs (same mode, rows <= 60 s apart)."""
    p = pd.read_parquet(day_path / "predictions" / f"{model}.parquet")
    p = p[p.track == "transport_mode"].sort_values("t_start_ms").reset_index(drop=True)
    if p.empty:
        return pd.DataFrame(columns=["start", "end", "mode", "hint", "source"])
    top = p["class"].map(PKG_TOP).fillna("unknown")
    s, e = p.t_start_ms.to_numpy() / 1e3, p.t_end_ms.to_numpy() / 1e3
    new = np.ones(len(p), bool)
    new[1:] = (top.to_numpy()[1:] != top.to_numpy()[:-1]) | (s[1:] - e[:-1] > 60)
    g = np.cumsum(new)
    out = pd.DataFrame({"g": g, "start": s, "end": e, "mode": top, "cls": p["class"]}).groupby("g").agg(
        start=("start", "min"), end=("end", "max"), mode=("mode", "first"),
        hint=("cls", lambda c: PKG_HINT.get(c.value_counts().index[0], "")))
    return out.assign(source="package").reset_index(drop=True)


def _ov(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def match(d: pd.DataFrame, legs: pd.DataFrame, date: str, model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per diary leg, per predicted leg) tables. legs: start, end, mode (name), source."""
    drows = []
    for r in d.itertuples():
        dur = r.end - r.start
        ov = np.array([_ov(r.start, r.end, a, b) for a, b in zip(legs.start, legs.end)]) if len(legs) else np.zeros(0)
        same = (legs["mode"].to_numpy() == r.mode) if len(legs) else np.zeros(0, bool)
        iou = ov / np.maximum((legs.end - legs.start).to_numpy() + dur - ov, 1e-9) if len(legs) else ov
        best = int(np.argmax(np.where(same, iou, -1))) if same.any() else -1
        by_mode = pd.Series(ov).groupby(legs["mode"].to_numpy()).sum() if len(legs) else pd.Series(dtype=float)
        model_cov = ov[(legs.source == "model").to_numpy()].sum() if len(legs) else 0.0
        drows.append({"model": model, "diary_id": r.diary_id, "start_local": r.start_local, "end_local": r.end_local,
                      "diary_mode": r.mode, "diary_sub": r.sub_mode, "gt": r.gt,
                      "matched": bool(best >= 0 and iou[best] >= 0.5),
                      "best_iou": float(iou[best]) if best >= 0 else 0.0,
                      "same_mode_share": float(ov[same].sum() / dur) if len(legs) else 0.0,
                      "model_data_share": float(model_cov / dur),
                      "dominant": by_mode.idxmax() if len(by_mode) and by_mode.max() > 0 else "no data",
                      "hint": legs.hint.iat[best] if best >= 0 and iou[best] > 0 and "hint" in legs else ""})
    lrows = []
    for i, r in legs.reset_index(drop=True).iterrows():
        dur = max(r.end - r.start, 1e-9)
        sm = sum(_ov(r.start, r.end, a, b) for a, b, m in zip(d.start, d.end, d["mode"]) if m == r["mode"])
        anyd = sum(_ov(r.start, r.end, a, b) for a, b in zip(d.start, d.end))
        if r["mode"] in ("still", "unknown") or r.source != "model" and r.source != "package":
            status = "not scored"
        elif sm / dur >= 0.5:
            status = "in diary"
        elif date in WALK_UNTRACKED and r["mode"] in ("walking", "running") and sm == 0:
            status = "not in diary (walking not tracked)"
        else:
            status = "extra"
        lrows.append({"model": model, "leg": i, "same_mode_diary_share": sm / dur, "any_diary_share": anyd / dur,
                      "status": status})
    return pd.DataFrame(drows), pd.DataFrame(lrows)


def tram_check(d: pd.DataFrame, minutes: pd.DataFrame, legs: pd.DataFrame, model: str) -> pd.DataFrame:
    """Per diary tram leg: argmax fine class shares, GPS coverage, top-level mode and hint of the dominant leg."""
    rows = []
    fine = [c for c in L.FINE_CLASSES]
    for r in d[(d["mode"] == "rail") & (d.sub_mode == "tram")].itertuples():
        m = minutes[(minutes.ts + 60 > r.start) & (minutes.ts < r.end)]
        arg = np.array(fine)[m[[f"pf_{c}" for c in fine]].to_numpy().argmax(1)] if len(m) else np.array([])
        sh = {f"share_{c}": float((arg == c).mean()) if len(arg) else np.nan
              for c in ("tram", "train", "subway", "bus", "car")}
        sh["share_other"] = 1 - np.nansum(list(sh.values())) if len(arg) else np.nan
        ov = np.array([_ov(r.start, r.end, a, b) for a, b in zip(legs.start, legs.end)])
        k = int(ov.argmax()) if len(ov) and ov.max() > 0 else -1
        rows.append({"model": model, "diary_id": r.diary_id, "start_local": r.start_local, "end_local": r.end_local,
                     "minutes_with_data": len(m), "gps_covered_share": float(m.covered.mean()) if len(m) else np.nan,
                     "leg_mode": legs["mode"].iat[k] if k >= 0 else "no leg",
                     "leg_hint": legs.hint.iat[k] if k >= 0 else "",
                     "leg_hint_share": legs.hint_share.iat[k] if k >= 0 and "hint_share" in legs else np.nan,
                     "top_level_right": bool(k >= 0 and legs["mode"].iat[k] == "rail"), **sh})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- sheet

COLORS = {"still": "#2a78d6", "walking": "#eb6834", "running": "#1baf7a", "bike": "#eda100", "car": "#e87ba4",
          "bus": "#008300", "rail": "#6250d6", "unknown": "#b5b4ad"}
ABBR = {"still": "still", "walking": "walk", "running": "run", "bike": "bike", "car": "car", "bus": "bus",
        "rail": "rail", "unknown": "?"}


def sheet(path, date: str, lanes: dict, cover: pd.DataFrame, fixes: pd.DataFrame, title: str) -> None:
    """lanes: name -> legs (start, end, mode, hint, conf, source). cover: ts, kind (acc+gps / acc).
    fixes: t, v_next (derived), speed_mps (reported). Writes a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    def x(t):
        return pd.to_datetime(np.asarray(t, float), unit="s", utc=True).tz_convert(TZ).tz_localize(None)

    allt = np.concatenate([l[["start", "end"]].to_numpy().ravel() for l in lanes.values() if len(l)])
    lo, hi = np.nanmin(allt) - 900, np.nanmax(allt) + 900
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(18, 6.2), sharex=True, gridspec_kw={"height_ratios": [3.2, 1.4]})
    names = list(lanes) + ["data"]
    for yi, (name, lg) in enumerate(lanes.items()):
        y = len(names) - 1 - yi
        for r in lg.itertuples():
            if r.end < lo or r.start > hi:
                continue
            c = COLORS.get(r.mode, "#b5b4ad")
            x0, x1 = x([r.start])[0], x([r.end])[0]
            ax.barh(y, x1 - x0, left=x0, height=0.62, color=c, edgecolor="#fcfcfb", linewidth=1.5,
                    hatch="//" if r.mode == "unknown" or getattr(r, "source", "") .startswith("rule") else None)
            w = (r.end - r.start) / 60
            lab = ABBR.get(r.mode, r.mode) + (f" {r.hint}" if getattr(r, "hint", "") and w >= 12 else "")
            if getattr(r, "conf", np.nan) == getattr(r, "conf", np.nan) and w >= 14:
                lab += f" {r.conf:.2f}"
            if w >= 6:
                ax.text(x0 + (x1 - x0) / 2, y, lab, ha="center", va="center", fontsize=7, color="#0b0b0b")
    # data coverage strip
    for kind, c in (("acc+gps", "#52514e"), ("acc", "#b5b4ad")):
        m = cover[cover.kind == kind]
        for t in m.ts:
            ax.barh(0, pd.Timedelta(minutes=1), left=x([t])[0], height=0.4, color=c, linewidth=0)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(list(reversed(names)), fontsize=9)
    ax.set_xlim(x([lo])[0], x([hi])[0])
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles = [Patch(color=COLORS[m], label=m) for m in COLORS] + \
        [Patch(facecolor="white", edgecolor="#52514e", hatch="//", label="GPS-only / unknown"),
         Patch(color="#52514e", label="data: acc + GPS"), Patch(color="#b5b4ad", label="data: acc only")]
    ax.legend(handles=handles, ncol=11, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False)
    ax.set_title(title, fontsize=10, loc="left", pad=26)
    f = fixes[(fixes.t >= lo) & (fixes.t <= hi)]
    ax2.plot(x(f.t), f.v_next * 3.6, color="#2a78d6", linewidth=1.2, label="derived speed (consecutive fixes)")
    r = f.dropna(subset=["speed_mps"])
    ax2.scatter(x(r.t), r.speed_mps * 3.6, s=9, color="#eb6834", label="reported speed", zorder=3)
    ax2.set_ylim(0, min(80, max(10, np.nanpercentile(f.v_next * 3.6, 99) * 1.1)) if len(f) else 50)
    ax2.set_ylabel("GPS speed [km/h]", fontsize=8)
    ax2.grid(axis="y", color="#e6e5e0", linewidth=0.6)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2.set_xlabel(f"local time ({TZ}), {date}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
