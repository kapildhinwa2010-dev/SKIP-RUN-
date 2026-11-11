import numpy as np
import pytest

from skip_run.depth import real_depth, run_stream, skipped_indices, skip_sweep


def _zero(v):
    return np.zeros_like(v)


def _const(v):
    return np.array([5.0, 0.0, 0.0, 0.0])


def _blocks():
    return [_zero, _zero, _const, _zero]


def test_run_stream():
    x = np.array([1.0, 0.0, 0.0, 0.0])
    out = run_stream(_blocks(), x)
    assert out[0] == pytest.approx(6.0)
    assert np.allclose(out, [6.0, 0.0, 0.0, 0.0])


def test_skipped_indices():
    assert skipped_indices(4, 1) == [0, 1, 2, 3]
    assert skipped_indices(4, 2) == [1, 3]
    assert skipped_indices(4, 3) == [2]
    assert skipped_indices(4, 4) == [3]
    assert skipped_indices(4, 5) == []


def test_skip_sweep():
    x = np.array([1.0, 0.0, 0.0, 0.0])
    report = skip_sweep(_blocks(), x, score_fn=lambda v: float(v[0]))
    assert report["baseline_score"] == pytest.approx(6.0)
    rows = {row["k"]: row for row in report["sweep"]}
    assert rows[1]["removed"] == [0, 1, 2, 3]
    assert rows[1]["delta"] == pytest.approx(5.0)
    assert rows[2]["removed"] == [1, 3]
    assert rows[2]["delta"] == pytest.approx(0.0)
    assert rows[3]["removed"] == [2]
    assert rows[3]["delta"] == pytest.approx(5.0)
    assert rows[4]["removed"] == [3]
    assert rows[4]["delta"] == pytest.approx(0.0)
    assert rows[2]["retained_fraction"] == pytest.approx(0.5)


def test_real_depth():
    x = np.array([1.0, 0.0, 0.0, 0.0])
    report = skip_sweep(_blocks(), x, score_fn=lambda v: float(v[0]))
    assert real_depth(report, tolerance=0.1) == 4
    assert real_depth(report, tolerance=0.001) is not None


def test_k_beyond_n():
    x = np.array([1.0, 0.0, 0.0, 0.0])
    report = skip_sweep(_blocks(), x, score_fn=lambda v: float(v[0]), k_values=(5,))
    assert report["sweep"][0]["removed"] == []
    assert report["sweep"][0]["delta"] == pytest.approx(0.0)
    assert report["sweep"][0]["retained_fraction"] == 1.0


def test_errors():
    with pytest.raises(ValueError):
        skip_sweep([], np.zeros(4), score_fn=lambda v: 0.0)
    with pytest.raises(ValueError):
        skipped_indices(4, 0)
    with pytest.raises(ValueError):
        real_depth({})