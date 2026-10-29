"""tracking.run() must refuse the legacy split names (the superseded 2026-10-25 test set, experiment 02 only)."""
import pytest

from tmd import tracking


@pytest.mark.parametrize("name", sorted(tracking.LEGACY))
def test_run_refuses_legacy_split_names(name, monkeypatch):
    started = []
    monkeypatch.setattr(tracking.mlflow, "start_run", lambda *a, **k: started.append(1))
    monkeypatch.setattr(tracking, "_use_experiment", lambda *a: started.append(1))
    d = tracking.DataInput(name, "evaluation", label_rules="v2", features="feat-v1")
    with pytest.raises(ValueError, match="legacy"):
        with tracking.run("00_legacy-check", {}, datasets=[d]):
            pass
    assert not started, "a run was started before the legacy name was refused"


def test_legacy_names_resolve_and_current_names_exist():
    for name in tracking.LEGACY:
        assert (tracking.PROJECT / tracking.SPLITS[name]).exists()
    for name in ("extrasensory_test_v2", "shl_user23_test_v2", "extrasensory_dev_v2"):
        assert name in tracking.SPLITS and name not in tracking.LEGACY
