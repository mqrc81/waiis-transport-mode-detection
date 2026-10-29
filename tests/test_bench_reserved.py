"""The reserved final-test users (tmd.splits.reserved_test(), the current set) must never reach a pre-training accessor of tmd.bench.

Two layers: a synthetic check of the filter itself (always runs), and a check of every accessor on the built
waiis_bench_v1 (skipped when the data are not on this machine).
"""
import pandas as pd
import pytest

from tmd import bench

ACCESSORS = {  # accessor -> sources it applies to
    "pretraining_minutes": bench.SOURCES,
    "unlabelled_minutes": bench.SOURCES,
    "snippets": ("extrasensory",),
    "shl_stream": ("shl_user1", "shl_user23"),
    "shl_other_positions": ("shl_user1",),
    "shl_gps_1hz": ("shl_user1", "shl_user23"),
    "shl_gps_unblanked": ("shl_user1", "shl_user23"),
}


def test_every_pretraining_accessor_is_checked():
    assert set(ACCESSORS) == set(bench.PRETRAINING_ACCESSORS)


@pytest.mark.parametrize("source", bench.SOURCES)
def test_drop_reserved_removes_every_reserved_user(source):
    reserved = sorted(bench.reserved_users(source))
    assert reserved, "the reserved set lists no users for this source"
    df = pd.DataFrame({"user": reserved + ["someone-else"], "ts": range(len(reserved) + 1)})
    assert set(bench.drop_reserved(df, source).user) == {"someone-else"}


def _call(name, source):
    f = getattr(bench, name)
    if name in ("snippets", "shl_other_positions"):
        return f(include_reserved=False)
    return f(source, include_reserved=False)


@pytest.mark.parametrize("name,source", [(n, s) for n, ss in ACCESSORS.items() for s in ss])
def test_accessor_never_returns_reserved_users(name, source):
    try:
        bench.path(source, "labels")
    except FileNotFoundError:
        pytest.skip(f"{bench.VERSION} is not built on this machine")
    if name == "snippets":                       # 2 GB table: check the user column only
        users = set(pd.read_parquet(bench.path("extrasensory", "snippets"), columns=["user"]).user)
        reserved_in_table = users & bench.reserved_users("extrasensory")
        assert reserved_in_table, "test is vacuous: the table has no reserved user to filter"
        got = set(bench.snippets(filters=[("user", "in", sorted(reserved_in_table))]).user)
    elif name == "shl_stream" and source == "shl_user1":
        got = set(bench.shl_stream(source, filters=[("rec", "==", "010317")]).user)
    else:
        got = set(_call(name, source).user)
    assert not got & bench.reserved_users(source)


def test_excluded_users_never_official_or_pretraining():
    from tmd.extrasensory import EXCLUDED_USERS
    try:
        bench.path("extrasensory", "labels")
    except FileNotFoundError:
        pytest.skip(f"{bench.VERSION} is not built on this machine")
    assert not set(bench.examples("extrasensory").user) & set(EXCLUDED_USERS)
    assert not set(bench.pretraining_minutes("extrasensory").user) & set(EXCLUDED_USERS)
    assert not set(bench.unlabelled_minutes("extrasensory").user) & set(EXCLUDED_USERS)
