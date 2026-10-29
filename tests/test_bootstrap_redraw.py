"""User bootstrap (tmd.evaluate): resamples lacking a class of the full set are redrawn (experiment 05)."""
import numpy as np

from tmd import evaluate


def _data(n_users=20, rare_users=2, seed=0):
    """Every user has classes 1 and 2; only the first `rare_users` users have class 3."""
    rng = np.random.default_rng(seed)
    y, users = [], []
    for u in range(n_users):
        cls = [1, 2] + ([3] if u < rare_users else [])
        k = rng.integers(20, 50)
        y.append(rng.choice(cls, k))
        users.append(np.full(k, u))
    return np.concatenate(y), np.concatenate(users)


def test_every_resample_has_every_class_and_rate_matches():
    y, users = _data()
    stats = {}
    sels = list(evaluate.user_resamples(y, users, n_boot=2000, seed=0, stats=stats))
    assert len(sels) == 2000
    assert all(set(np.unique(y[s])) == {1, 2, 3} for s in sels)
    p_missing = (18 / 20) ** 20                   # no rare user drawn
    assert abs(stats["redraw_rate"] - p_missing) < 0.03


def test_no_redraw_when_no_class_can_be_missing_matches_plain_bootstrap():
    """With every class in every user, the draws equal the plain bootstrap (same rng stream as before the change)."""
    y, users = _data(rare_users=20)
    pred = np.where(np.random.default_rng(1).random(len(y)) < 0.8, y, 1)
    stats = {}
    got = evaluate.bootstrap_users(y, pred, users, "class", n_boot=200, seed=0, stats=stats)
    rng = np.random.default_rng(0)
    uniq = np.unique(users)
    idx = {u: np.flatnonzero(users == u) for u in uniq}
    vals = []
    for _ in range(200):
        sel = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq))])
        vals.append(evaluate.scores(y[sel], pred[sel], "class")["macro_f1"])
    assert stats["redraw_rate"] == 0
    assert got == (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def test_recall_and_diff_report_rate():
    y, users = _data()
    pred = y.copy()
    s1, s2 = {}, {}
    ci = evaluate.bootstrap_users_recall(y, pred, users, "class", n_boot=300, stats=s1)
    assert set(ci) == {"macro_f1", "still", "walk", "run"}
    d = evaluate.bootstrap_users_diff(y, pred, np.ones_like(y), users, "class", n_boot=300, stats=s2)
    assert d["lo"] > 0 and s1["redraw_rate"] > 0 and s2["redraw_rate"] == s1["redraw_rate"]
