"""Paired user-bootstrap differences between saved rows, plus quiet-split still recall and car / bus / train recall
CIs of one row (descriptive). Development data only.

    uv run python compare.py --rows F_fb:F_viterbi F_fb:X_Dm_fb --detail F_fb
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
import pipeline as pl

from tmd.evaluate import (
    bootstrap_users_diff,
    bootstrap_users_recall,
    family_proba,
    scores,
)
from tmd.labels import to_family


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="*", default=[], help="a:b pairs, difference a - b")
    ap.add_argument("--detail")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    df = pl.load(pl.Cfg("keys"))
    es = ((df.source == "extrasensory") & df.official).to_numpy()
    shl = ((df.source == "shl_user1") & df.official).to_numpy()
    y8, users = df.y.to_numpy(), df.user.to_numpy()
    out = {}
    for pair in a.rows:
        ra, rb = pair.split(":")
        pa, pb = pl.load_oof(ra, a.seed, df), pl.load_oof(rb, a.seed, df)
        for lvl in ("family", "class"):
            f = (lambda p: family_proba(p).argmax(1) + 1) if lvl == "family" else (lambda p: p.argmax(1) + 1)
            yy = to_family(y8) if lvl == "family" else y8
            d = bootstrap_users_diff(yy[es], f(pa[es]), f(pb[es]), users[es], lvl)
            ds = scores(yy[shl], f(pa[shl]), lvl)["macro_f1"] - scores(yy[shl], f(pb[shl]), lvl)["macro_f1"]
            out[f"{pair} {lvl}"] = {"es_diff": round(d["diff"], 4), "es_lo": round(d["lo"], 4),
                                    "es_hi": round(d["hi"], 4), "shl_diff": round(ds, 4)}
    if a.detail:
        p = pl.load_oof(a.detail, a.seed, df)
        pred = p.argmax(1) + 1
        for src, sel in (("es", es), ("shl", shl)):
            still = sel & (y8 == 1)
            q = df.quiet.to_numpy().astype(bool)
            out[f"{a.detail} {src} still recall"] = {
                "quiet": round(float((pred[still & q] == 1).mean()), 4), "n_quiet": int((still & q).sum()),
                "not_quiet": round(float((pred[still & ~q] == 1).mean()), 4), "n_not_quiet": int((still & ~q).sum())}
        ci = bootstrap_users_recall(y8[es], pred[es], users[es], "class")
        s = scores(y8[es], pred[es], "class")["recall"]
        out[f"{a.detail} es recall"] = {c: [round(s[c], 4), round(ci[c][0], 4), round(ci[c][1], 4)]
                                        for c in ("car", "bus", "train")}
        cm = pd.crosstab(pd.Series(y8[es], name="true"), pd.Series(pred[es], name="pred"))
        out[f"{a.detail} es car/bus/train rows"] = {int(i): cm.loc[i].to_dict() for i in (5, 6, 7) if i in cm.index}
    for k, v in out.items():
        print(k, json.dumps(v, default=int))


if __name__ == "__main__":
    main()
