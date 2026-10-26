"""Evaluation tables shared by the step-2/4 scripts: SHL (per position) and ExtraSensory (labels v2, per position,
platform, carried state), each at 8-class and family level.

ExtraSensory reporting rules (approved 2026-10-26): main macro-F1 on labels v2 (resting still capped at 20% per user);
still is reported twice, "still (v2)" and "still, carried phone only" (pocket / hand / bag reported, not resting);
still predicted as rail is broken down by carried / not carried / unknown position.
"""
import numpy as np
import pandas as pd

from tmd import extrasensory as es
from tmd.evaluate import bootstrap_users, family_proba, plot_cm, scores
from tmd.labels import CLASSES, FAMILIES, to_family

CARRIED = ["PHONE_IN_POCKET", "PHONE_IN_HAND", "PHONE_IN_BAG"]


def es_label_v2(win: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Adds to ExtraSensory windows: v2 reason, carry state ('carried' | 'not_carried' | 'unknown') and
    `v2_keep` (labelled ok under v2 and not removed by the resting-still cap)."""
    mins = es.minutes_table(win.user.unique())
    mins["carry"] = np.where(mins.resting, "not_carried",
                             np.where(mins[CARRIED].any(axis=1), "carried", "unknown"))
    ok = mins.reason == "ok"
    keep = np.zeros(len(mins), bool)
    keep[np.flatnonzero(ok)] = es.cap_resting_still(mins[ok], seed=seed)
    mins["v2_keep"] = keep
    cols = mins[["user", "ts", "reason", "carry", "v2_keep"]].rename(columns={"reason": "reason_v2"})
    out = win.merge(cols, on=["user", "ts"], how="left", validate="many_to_one")
    out["v2_keep"] = out.v2_keep.fillna(False).astype(bool)
    return out


def _flat(prefix, s, lvl):
    names = CLASSES if lvl == "class" else FAMILIES
    out = {f"{prefix}_macro_f1": s["macro_f1"], f"{prefix}_acc": s["acc"], f"{prefix}_n": s["n"]}
    out.update({f"{prefix}_recall_{n}": s["recall"][n] for n in names if s["recall"][n] == s["recall"][n]})
    return out


def evaluate(shl: pd.DataFrame | None, shl_p: np.ndarray | None, es_w: pd.DataFrame | None, es_p: np.ndarray | None,
             title: str, ci: bool = True) -> tuple[dict, dict, dict]:
    """Returns (metrics, tables, figures). shl/es_w rows align with shl_p/es_p (N, 8) probabilities; es_w comes
    from `es_label_v2`."""
    metrics, tables, figs = {}, {}, {}

    def block(name, y, p8, pf, users=None, fig=False):
        for lvl, yy, pp in [("class", y, p8), ("family", to_family(y), pf)]:
            s = scores(yy, pp, lvl)
            metrics.update(_flat(f"{name}_{lvl}", s, lvl))
            tables[f"{name}_{lvl}_cm"] = s["cm"].tolist()
            if fig:
                figs[f"cm_{name}_{lvl}.png"] = plot_cm(s["cm"], lvl, f"{title}: {name}, {lvl}")
            if users is not None and ci:
                lo, hi = bootstrap_users(yy, pp, users, lvl)
                metrics[f"{name}_{lvl}_macro_f1_ci_lo"], metrics[f"{name}_{lvl}_macro_f1_ci_hi"] = lo, hi

    if shl is not None:
        y = shl.y.to_numpy()
        p8, pf = shl_p.argmax(1) + 1, family_proba(shl_p).argmax(1) + 1
        groups = [("shl", np.ones(len(y), bool)), ("shl_pure", shl.pure.to_numpy())] + \
                 [(f"shl_{p.lower()}", (shl.position == p).to_numpy()) for p in ["Hips", "Hand", "Torso", "Bag"]]
        for name, sel in groups:
            block(name, y[sel], p8[sel], pf[sel], fig=(name == "shl"))

    if es_w is not None:
        y = es_w.y.to_numpy()
        p8, pf = es_p.argmax(1) + 1, family_proba(es_p).argmax(1) + 1
        main = es_w.v2_keep.to_numpy() & (es_w.reason == "ok").to_numpy()
        carry = es_w.carry.to_numpy()
        users = es_w.user.to_numpy()
        still = y == 1
        groups = [("es", main, True), ("es_stillcarried", main & (~still | (carry == "carried")), True)] + \
                 [(f"es_carry_{c}", main & (carry == c), False) for c in ["carried", "not_carried", "unknown"]] + \
                 [(f"es_{p}", main & (es_w.position == p).to_numpy(), False)
                  for p in ["in_pocket", "in_hand", "in_bag", "on_table", "none"]] + \
                 [(f"es_{p}", main & (es_w.platform == p).to_numpy(), False) for p in ["iphone", "android"]]
        for name, sel, with_ci in groups:
            block(name, y[sel], p8[sel], pf[sel], users[sel] if with_ci else None, fig=(name == "es"))
        # still twice: v2 and carried-phone-only subset
        for name, sel in [("still_v2", main & still), ("still_carried", main & still & (carry == "carried"))]:
            metrics[f"es_{name}_n"] = int(sel.sum())
            metrics[f"es_{name}_recall_class"] = float((p8[sel] == 1).mean())
            metrics[f"es_{name}_recall_family"] = float((pf[sel] == 1).mean())
        # still predicted as rail, by carry state
        for c in ["carried", "not_carried", "unknown"]:
            sel = main & still & (carry == c)
            metrics[f"es_still_{c}_n"] = int(sel.sum())
            metrics[f"es_still_{c}_to_rail"] = float((pf[sel] == 5).mean()) if sel.any() else float("nan")
            metrics[f"es_still_{c}_to_still"] = float((pf[sel] == 1).mean()) if sel.any() else float("nan")
        fam = to_family(y)
        for src, fid in [("road", 4), ("still", 1)]:
            sel = main & (fam == fid)
            metrics[f"es_rail_fp_rate_on_{src}"] = float((pf[sel] == 5).mean())
        metrics["es_train_windows"] = int((main & (y == 7)).sum())
        moto = (es_w.reason == "check:MOTORBIKE").to_numpy()
        tables["es_motorbike_pred_family"] = pd.Series(pf[moto]).map(dict(enumerate(FAMILIES, 1))).value_counts().to_dict()
        tables["es_motorbike_pred_class"] = pd.Series(p8[moto]).map(dict(enumerate(CLASSES, 1))).value_counts().to_dict()
    return metrics, tables, figs
