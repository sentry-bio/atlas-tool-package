"""Tests for Mondrian conformal calibrator."""
import pytest
import numpy as np
from biosphere_atlas.hplg.calibrator import MondrianConformalCalibrator, MIN_CAL_SAMPLES
from biosphere_atlas.hplg.taxonomy import Rank


def test_default_thresholds():
    cal = MondrianConformalCalibrator()
    t = cal.get_thresholds(Rank.SPECIES)
    assert t.q_accept > 0
    assert t.q_fallback >= t.q_accept


def test_three_zone_decision():
    cal = MondrianConformalCalibrator()
    t = cal.get_thresholds(Rank.DOMAIN)

    assert cal.decide(0.0, Rank.DOMAIN) == "accept"
    assert cal.decide(t.q_accept - 0.01, Rank.DOMAIN) == "accept"
    assert cal.decide(t.q_fallback + 0.1, Rank.DOMAIN) == "fallback"

    # Escalation zone
    mid = (t.q_accept + t.q_fallback) / 2
    assert cal.decide(mid, Rank.DOMAIN) == "escalate"


def test_ordering_constraint():
    """q_fallback must always be >= q_accept."""
    cal = MondrianConformalCalibrator()
    for rank in Rank:
        t = cal.get_thresholds(rank)
        assert t.q_fallback >= t.q_accept, f"Ordering violated at {rank.name}"


def test_calibration_updates_thresholds():
    cal = MondrianConformalCalibrator(epsilon_accept=0.10, epsilon_fallback=0.01)

    # Feed uniform scores
    np.random.seed(42)
    for _ in range(100):
        score = np.random.uniform(0, 1)
        cal.add_score(score, Rank.GENUS)

    t = cal.get_thresholds(Rank.GENUS)
    # With uniform [0,1] scores:
    # q_accept should be near 0.9 (90th percentile)
    # q_fallback should be near 0.99 (99th percentile)
    assert 0.7 < t.q_accept < 1.0
    assert t.q_fallback >= t.q_accept


def test_calibration_needs_minimum_samples():
    cal = MondrianConformalCalibrator()
    t_before = cal.get_thresholds(Rank.FAMILY)

    # Add fewer than MIN_CAL_SAMPLES
    for i in range(MIN_CAL_SAMPLES - 5):
        cal.add_score(0.3, Rank.FAMILY)

    t_after = cal.get_thresholds(Rank.FAMILY)
    # Should still be using defaults
    assert t_after.q_accept == t_before.q_accept


def test_calibration_summary():
    cal = MondrianConformalCalibrator()
    for _ in range(50):
        cal.add_score(np.random.uniform(0, 1), Rank.SPECIES)

    summary = cal.calibration_summary()
    assert summary[Rank.SPECIES]["calibrated"] is True
    assert summary[Rank.SPECIES]["n_scores"] == 50
    assert summary[Rank.DOMAIN]["calibrated"] is False


def test_state_dict_roundtrip():
    cal = MondrianConformalCalibrator()
    np.random.seed(42)
    for _ in range(50):
        cal.add_score(np.random.uniform(0, 1), Rank.GENUS)

    state = cal.state_dict()
    loaded = MondrianConformalCalibrator.from_state_dict(state)

    t_orig = cal.get_thresholds(Rank.GENUS)
    t_loaded = loaded.get_thresholds(Rank.GENUS)
    assert abs(t_orig.q_accept - t_loaded.q_accept) < 1e-6
    assert abs(t_orig.q_fallback - t_loaded.q_fallback) < 1e-6
