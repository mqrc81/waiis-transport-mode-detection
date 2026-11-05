"""The labelling tool's format for leg suggestions (provisional, experiment 11): the only module that knows it. When
the tool's repo is available, its real format changes this file only.

One JSON file per model and day. Times in UTC epoch ms. Modes on the tool's top level (`tmd.legs.TOOL_TOP`).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tmd import legs as L

FORMAT = "tmd-legs/0.1"


def _mode(i) -> str:
    return "unknown" if i is None or i < 0 else L.MODES[int(i)]


def _num(v, nd=4):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), nd)


def document(legs: pd.DataFrame, user: str, date: str, model_id: str, model_version: str, rules: str,
             inputs: dict, code_commit: str) -> dict:
    out_legs = []
    for r in legs.itertuples():
        top2 = [] if r.source == "rule:gps_only_moving" else \
            [{"mode": _mode(r.top1), "p": _num(r.p1)}] + ([{"mode": _mode(r.top2), "p": _num(r.p2)}]
                                                        if r.source == "model" else [])
        mid = model_id if r.source == "model" else r.source
        out_legs.append({
            "start_ms": int(round(r.start * 1000)), "end_ms": int(round(r.end * 1000)), "mode": _mode(r.mode),
            "sub_mode_hint": {"mode": r.hint, "share": _num(r.hint_share, 3)} if r.hint else None,
            "top2": top2, "show_second": False, "confidence": _num(r.p1),
            "low_confidence": bool(r.mode == L.UNKNOWN) if r.source == "model" else r.source != "model",
            "flag_reasons": list(r.flag_reasons), "coverage": r.coverage,
            "gps_covered_share": _num(r.gps_covered_share, 3), "n_minutes": int(r.n_minutes),
            "model_id": mid, "model_version": model_version if r.source == "model" else "11"})
    m = legs.attrs.get("minutes", pd.DataFrame())
    out_min = [{"t_ms": int(row.ts) * 1000, "coverage": "acc+gps" if row.covered else "acc",
                "p": {k: _num(getattr(row, f"p_{k}")) for k in L.MODES},
                "p_fine": {c: _num(getattr(row, f"pf_{c}")) for c in L.FINE_CLASSES}}
               for row in m.itertuples()]
    return {"format": FORMAT, "user_key": user, "date": date, "time": "UTC epoch ms",
            "model": {"id": model_id, "version": model_version, "rules": rules, "code_commit": code_commit},
            "input": inputs, "legs": out_legs, "minutes": out_min}


def write(doc: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1))


def read_legs(path: Path) -> pd.DataFrame:
    """The legs of a written file as a table (start / end in s)."""
    d = json.loads(Path(path).read_text())
    t = pd.DataFrame(d["legs"])
    if len(t):
        t["start"], t["end"] = t.start_ms / 1000, t.end_ms / 1000
    return t
