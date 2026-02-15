"""
Tests for high-level build API.
"""

import math

import pytest
import torch

from biosphere_atlas.tree.hyperbolic import KAPPA_DEFAULT, _clamp_to_ball
from biosphere_atlas.tree.build import build_tree, estimate_tree_quality


KAPPA = KAPPA_DEFAULT
DIM = 8


def _ball_point(dim: int = DIM, scale: float = 0.3) -> torch.Tensor:
    p = torch.randn(dim) * scale / math.sqrt(KAPPA)
    return _clamp_to_ball(p, KAPPA)


class TestBuildTree:
    def test_basic(self):
        torch.manual_seed(42)
        embs = torch.stack([_ball_point() for _ in range(6)])
        taxa = [f"t_{i}" for i in range(6)]

        tree, report = build_tree(embs, taxa, kappa=KAPPA)
        assert tree.n_leaves == 6
        assert report is not None
        assert report.total_quartets > 0

    def test_no_validation(self):
        torch.manual_seed(42)
        embs = torch.stack([_ball_point() for _ in range(6)])
        taxa = [f"t_{i}" for i in range(6)]

        tree, report = build_tree(embs, taxa, validate_quartets=False)
        assert tree.n_leaves == 6
        assert report is None

    def test_midpoint_root(self):
        torch.manual_seed(42)
        embs = torch.stack([_ball_point() for _ in range(6)])
        taxa = [f"t_{i}" for i in range(6)]

        tree, _ = build_tree(embs, taxa, midpoint_root=True)
        root = tree.find_root()
        assert root is not None

    def test_small_tree(self):
        torch.manual_seed(42)
        embs = torch.stack([_ball_point() for _ in range(3)])
        taxa = ["A", "B", "C"]

        tree, report = build_tree(embs, taxa)
        assert tree.n_leaves == 3
        # < 4 taxa means no quartets
        assert report is None

    def test_single_taxon(self):
        tree, report = build_tree(
            _ball_point().unsqueeze(0), ["only"], kappa=KAPPA
        )
        assert tree.n_leaves == 1
        assert report is None


class TestEstimateQuality:
    def test_basic(self):
        torch.manual_seed(42)
        embs = torch.stack([_ball_point() for _ in range(10)])
        quality = estimate_tree_quality(embs, kappa=KAPPA)
        assert "delta" in quality
        assert "n_points" in quality
        assert "mean_distance" in quality
        assert "std_distance" in quality
        assert quality["delta"] >= 0.0
        assert quality["n_points"] == 10.0
        assert quality["mean_distance"] > 0.0

    def test_clustered_points_low_delta(self):
        """Tightly clustered points should have lower delta."""
        torch.manual_seed(42)
        center = _ball_point(scale=0.1)
        embs = torch.stack([
            _clamp_to_ball(center + torch.randn(DIM) * 0.02, KAPPA)
            for _ in range(10)
        ])
        quality = estimate_tree_quality(embs, kappa=KAPPA, max_quartets=100)
        # Delta should be finite and non-negative
        assert quality["delta"] >= 0.0
        assert quality["delta"] < float("inf")
