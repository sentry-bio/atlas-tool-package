"""
Tests for conformal calibration.
"""

import math

import pytest
import torch

from biosphere_atlas.place.calibrator import (
    DEFAULT_WEIGHTS,
    PlacementCalibrator,
    ThresholdPair,
    compute_nonconformity,
)
from biosphere_atlas.place.hyperbolic import KAPPA_DEFAULT, _clamp_to_ball
from biosphere_atlas.place.placer import PlacementCandidate, PlacementResult
from biosphere_atlas.place.reference import Rank


KAPPA = KAPPA_DEFAULT
DIM = 8


class TestNonconformity:
    def test_low_distance_low_score(self):
        """Close placement with large margin should have low nonconformity."""
        score = compute_nonconformity(
            best_distance=0.1, margin=2.0, atlas_r=1.0, rank=Rank.SPECIES
        )
        assert score < 0  # Negative because margin dominates

    def test_high_distance_high_score(self):
        """Far placement with small margin should have high nonconformity."""
        score = compute_nonconformity(
            best_distance=5.0, margin=0.01, atlas_r=3.0, rank=Rank.SPECIES
        )
        assert score > 4.0

    def test_margin_reduces_score(self):
        """Larger margin should reduce nonconformity."""
        s1 = compute_nonconformity(0.5, margin=0.1, atlas_r=1.0)
        s2 = compute_nonconformity(0.5, margin=1.0, atlas_r=1.0)
        assert s2 < s1

    def test_radial_increases_score(self):
        """Higher evolutionary depth should increase nonconformity."""
        s1 = compute_nonconformity(0.5, margin=0.5, atlas_r=0.5)
        s2 = compute_nonconformity(0.5, margin=0.5, atlas_r=5.0)
        assert s2 > s1


class TestThresholdPair:
    def test_accept_zone(self):
        tp = ThresholdPair(q_accept=1.0, q_fallback=3.0)
        assert tp.decide(0.5) == "accept"

    def test_escalate_zone(self):
        tp = ThresholdPair(q_accept=1.0, q_fallback=3.0)
        assert tp.decide(2.0) == "escalate"

    def test_fallback_zone(self):
        tp = ThresholdPair(q_accept=1.0, q_fallback=3.0)
        assert tp.decide(4.0) == "fallback"

    def test_auto_swap(self):
        """q_accept and q_fallback should auto-swap if inverted."""
        tp = ThresholdPair(q_accept=3.0, q_fallback=1.0)
        assert tp.q_accept <= tp.q_fallback


class TestPlacementCalibrator:
    def test_uncalibrated_defaults(self):
        cal = PlacementCalibrator(epsilon=0.10)
        assert not cal.is_calibrated
        # Should still make decisions using defaults
        zone = cal.decide(0.1, Rank.SPECIES)
        assert zone in ("accept", "escalate", "fallback")

    def test_calibration_with_scores(self):
        cal = PlacementCalibrator(epsilon=0.10)
        # Add many scores for species rank
        torch.manual_seed(42)
        for _ in range(100):
            score = torch.randn(1).item() * 0.5 + 1.0
            cal.add_score(score, Rank.SPECIES)

        assert cal.is_calibrated
        assert cal.total_calibration_scores == 100

        # Low scores should be accepted
        zone_low = cal.decide(-1.0, Rank.SPECIES)
        assert zone_low == "accept"

        # Very high scores should be fallback
        zone_high = cal.decide(10.0, Rank.SPECIES)
        assert zone_high == "fallback"

    def test_confidence_range(self):
        cal = PlacementCalibrator()
        for _ in range(50):
            cal.add_score(torch.randn(1).item(), Rank.GENUS)

        conf = cal.confidence(0.0, Rank.GENUS)
        assert 0.0 <= conf <= 1.0

    def test_prediction_set_sizes(self):
        cal = PlacementCalibrator()
        for _ in range(100):
            cal.add_score(torch.randn(1).item() * 0.5, Rank.FAMILY)

        # Accept zone → set size 1
        assert cal.prediction_set_size(-5.0, Rank.FAMILY) == 1
        # Fallback zone → set size -1
        assert cal.prediction_set_size(100.0, Rank.FAMILY) == -1

    def test_calibrate_placement(self):
        cal = PlacementCalibrator()
        # Feed some calibration data
        for _ in range(50):
            cal.add_score(torch.randn(1).item() * 0.5 + 0.5, Rank.SPECIES)

        # Create a mock placement
        result = PlacementResult(
            sequence_id="test",
            candidates=[
                PlacementCandidate(
                    taxon_id="s__Ecoli",
                    rank=Rank.SPECIES,
                    distance=0.3,
                    lineage=("d__Bacteria", "s__Ecoli"),
                ),
            ],
            best_distance=0.3,
            margin=1.5,
            atlas_r=1.0,
            atlas_theta=0.5,
        )

        cal.calibrate_placement(result)
        assert result.zone is not None
        assert result.zone in ("accept", "escalate", "fallback")
        assert result.confidence is not None
        assert result.prediction_set_size is not None

    def test_batch_calibration(self):
        cal = PlacementCalibrator()
        for _ in range(50):
            cal.add_score(torch.randn(1).item(), Rank.GENUS)

        results = [
            PlacementResult(
                sequence_id=f"seq_{i}",
                candidates=[
                    PlacementCandidate("g__Test", Rank.GENUS, 0.5, ("g__Test",))
                ],
                best_distance=0.5,
                margin=0.8,
                atlas_r=1.0,
                atlas_theta=0.0,
            )
            for i in range(5)
        ]

        cal.calibrate_batch(results, ranks=[Rank.GENUS] * 5)
        for r in results:
            assert r.zone is not None

    def test_serialization(self):
        cal = PlacementCalibrator(epsilon=0.05, fallback_epsilon=0.005)
        for _ in range(30):
            cal.add_score(torch.randn(1).item(), Rank.ORDER)

        state = cal.state_dict()
        cal2 = PlacementCalibrator.from_state_dict(state)

        assert cal2.epsilon == 0.05
        assert cal2.fallback_epsilon == 0.005
        assert cal2.total_calibration_scores == 30
        # Decisions should be consistent
        for score in [-1.0, 0.0, 1.0, 5.0]:
            assert cal.decide(score, Rank.ORDER) == cal2.decide(score, Rank.ORDER)
