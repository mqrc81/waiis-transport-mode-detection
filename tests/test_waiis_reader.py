"""tmd.readers.waiis: device table lookup, axis sign, sessions (no data needed)."""
import numpy as np
import pandas as pd
import pytest

from tmd.readers import waiis

COLS = ["user_key", "valid_from", "valid_to", "platform", "manufacturer", "phone_model", "axis_convention",
        "axis_evidence", "source", "confirmed_on", "notes"]


def _table(rows):
    return pd.DataFrame([dict(zip(COLS, r)) for r in rows], columns=COLS).fillna("")


def test_committed_table_covers_the_recorded_user():
    t = waiis.devices()
    assert list(t.columns) == COLS
    d = waiis.device("sub_0c0b7a3d92467061eee0", "2026-11-03", t)
    assert (d.platform, d.phone_model, d.axis_convention) == ("iphone", "iPhone 16", "ios")


def test_phone_change_and_gaps():
    t = _table([("u", "2026-01-01", "2026-06-30", "android", "", "", "android", "", "", "", ""),
                ("u", "2026-07-01", "", "iphone", "", "", "ios", "", "", "", "")])
    assert waiis.device("u", "2026-06-30", t).platform == "android"
    assert waiis.device("u", "2026-07-01", t).platform == "iphone"
    assert waiis.device("u", "2030-01-01", t).platform == "iphone"
    with pytest.raises(LookupError):
        waiis.device("u", "2025-12-31", t)
    with pytest.raises(LookupError):
        waiis.device("other", "2026-07-01", t)


def test_overlapping_rows_fail():
    t = _table([("u", "2026-01-01", "", "android", "", "", "android", "", "", "", ""),
                ("u", "2026-07-01", "", "iphone", "", "", "ios", "", "", "", "")])
    with pytest.raises(LookupError):
        waiis.device("u", "2026-08-01", t)


def test_axis_sign():
    assert waiis.axis_sign(pd.Series({"axis_convention": "ios", "user_key": "u"})) == -1.0
    assert waiis.axis_sign(pd.Series({"axis_convention": "android", "user_key": "u"})) == 1.0
    with pytest.raises(ValueError):
        waiis.axis_sign(pd.Series({"axis_convention": "unknown", "user_key": "u"}))


def test_sessions_join_short_pauses():
    t = np.concatenate([np.arange(0, 300, 0.04), np.arange(304, 600, 0.04), np.arange(800, 900, 0.04)])
    s = waiis.session_ids(t)
    assert s[0] == s[np.searchsorted(t, 304)] == 0 and s[-1] == 1
