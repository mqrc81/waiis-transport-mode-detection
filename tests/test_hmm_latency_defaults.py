"""The defaults of tmd.hmm and tmd.evaluate.inference_latency reproduce their outputs from before experiment 05 added
step_s / pi / shape (golden values saved at commit 5852704: tests/data/golden_hmm_latency_v0.npz)."""
import hashlib
from pathlib import Path

import numpy as np

from tmd import evaluate, hmm

G = np.load(Path(__file__).parent / "data" / "golden_hmm_latency_v0.npz")


def test_hmm_defaults_unchanged():
    p, t, seq, y = G["p"], G["t"], G["seq"], G["y"]
    a = hmm.transition_matrix(y, t, seq)
    np.testing.assert_array_equal(a, G["a"])
    np.testing.assert_array_equal(hmm.forward(p, t, seq, a), G["fwd"])
    np.testing.assert_array_equal(hmm.viterbi(p, t, seq, a), G["vit"])
    np.testing.assert_array_equal(hmm.fixed_lag(p, t, seq, a, lag_s=30.0), G["fl"])


def test_forward_backward_equals_unbounded_fixed_lag():
    p, t, seq = G["p"], G["t"], G["seq"]
    a, pi = G["a"], np.arange(1, 9) / 36
    fb = hmm.forward_backward(p, t, seq, a, pi=pi)
    np.testing.assert_allclose(fb, hmm.fixed_lag(p, t, seq, a, lag_s=1e6, pi=pi), atol=1e-10)


def test_minute_step_matches_rescaled_time():
    """step_s = 60 on minute timestamps is the default 2.5 s step on time rescaled by 2.5 / 60."""
    p, t, seq = G["p"], G["t"], G["seq"]
    a = G["a"]
    tm = t * 60 / 2.5
    np.testing.assert_allclose(hmm.forward(p, tm, seq, a, max_gap_s=60.0 * 24, step_s=60.0),
                               hmm.forward(p, t, seq, a), atol=1e-12)


def test_inference_latency_default_window_unchanged():
    seen = []
    out = evaluate.inference_latency(lambda x: x.sum(), lambda w: seen.append(w.copy()) or w.reshape(1, -1),
                                     n=1, warmup=0)
    assert tuple(seen[0].shape) == tuple(G["lat_window_shape"])
    assert hashlib.sha256(seen[0].tobytes()).hexdigest() == str(G["lat_window_sha"])
    assert set(out) == {f"lat_{p}_ms_{q}" for p in ("inference", "features", "model") for q in ("p50", "p95")}
