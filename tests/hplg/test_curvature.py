"""Tests for curvature adapter."""
import pytest
from biosphere_atlas.hplg.curvature import CurvatureAdapter


def test_initial_state():
    adapter = CurvatureAdapter()
    assert adapter.state.phase == "functional"
    assert adapter.state.kappa == 1.0


def test_functional_phase():
    adapter = CurvatureAdapter()
    state = adapter.update(1.01)
    assert state.phase == "functional"
    assert state.transition_progress < 0.05


def test_phylogenetic_phase():
    adapter = CurvatureAdapter()
    state = adapter.update(1.245)
    assert state.phase == "phylogenetic"
    assert state.transition_progress > 0.95


def test_transition_detection():
    adapter = CurvatureAdapter()
    state = adapter.update(1.12)
    assert state.phase == "transition"
    assert 0.05 < state.transition_progress < 0.95


def test_threshold_scale_during_transition():
    adapter = CurvatureAdapter()
    # Force instability
    for k in [1.0, 1.05, 1.1, 1.15, 1.1, 1.12, 1.08, 1.13]:
        adapter.update(k)

    # In unstable transition, threshold scale should be > 1.0
    scale = adapter.threshold_scale()
    assert scale >= 1.0


def test_threshold_scale_stable():
    adapter = CurvatureAdapter()
    # Stable at functional
    for _ in range(60):
        adapter.update(1.0)

    scale = adapter.threshold_scale()
    assert scale == 1.0


def test_momentum_scale_transition():
    adapter = CurvatureAdapter()
    # Force unstable transition
    for k in [1.0, 1.05, 1.1, 1.15, 1.1, 1.12]:
        adapter.update(k)

    scale = adapter.momentum_scale()
    # During transition, should dampen (< 1.0)
    assert scale <= 1.0


def test_reanchor_scale_transition():
    adapter = CurvatureAdapter()
    for k in [1.0, 1.05, 1.1, 1.15, 1.1, 1.12]:
        adapter.update(k)

    scale = adapter.reanchor_scale()
    # During transition, reanchoring should increase
    assert scale >= 1.0


def test_phase_transition_sequence():
    """Simulate the full kappa transition from 1.0 to 1.2475."""
    adapter = CurvatureAdapter()
    phases_seen = set()

    # Gradual ramp
    for i in range(100):
        kappa = 1.0 + (1.2475 - 1.0) * (i / 99)
        state = adapter.update(kappa)
        phases_seen.add(state.phase)

    assert "functional" in phases_seen
    assert "transition" in phases_seen
    assert "phylogenetic" in phases_seen
