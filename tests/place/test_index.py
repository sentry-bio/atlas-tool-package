"""
Tests for spatial index (brute-force + VP-tree).
"""

import math

import pytest
import torch

from biosphere_atlas.place.hyperbolic import KAPPA_DEFAULT, _clamp_to_ball, poincare_distance
from biosphere_atlas.place.index import BruteForceIndex, PlacementIndex, VPTree


KAPPA = KAPPA_DEFAULT
DIM = 8


def _ball_point(dim: int = DIM, scale: float = 0.3) -> torch.Tensor:
    p = torch.randn(dim) * scale / math.sqrt(KAPPA)
    return _clamp_to_ball(p, KAPPA)


def _make_db(n: int = 50) -> torch.Tensor:
    return torch.stack([_ball_point() for _ in range(n)])


class TestBruteForce:
    def test_nearest_is_self(self):
        db = _make_db(20)
        index = BruteForceIndex(db, kappa=KAPPA)
        # Query the 5th point — nearest should be itself
        result = index.query(db[5], k=1)
        assert result.nearest_index == 5
        assert result.nearest_distance < 0.01  # asinh epsilon floor ~1e-4

    def test_k_results(self):
        db = _make_db(20)
        index = BruteForceIndex(db, kappa=KAPPA)
        result = index.query(_ball_point(), k=5)
        assert len(result.indices) == 5
        assert len(result.distances) == 5
        # Distances should be sorted ascending
        for i in range(len(result.distances) - 1):
            assert result.distances[i] <= result.distances[i + 1] + 1e-6

    def test_margin(self):
        db = _make_db(20)
        index = BruteForceIndex(db, kappa=KAPPA)
        result = index.query(_ball_point(), k=3)
        expected_margin = result.distances[1] - result.distances[0]
        assert abs(result.margin - expected_margin) < 1e-6

    def test_batch_query(self):
        db = _make_db(20)
        index = BruteForceIndex(db, kappa=KAPPA)
        queries = torch.stack([_ball_point() for _ in range(4)])
        results = index.query_batch(queries, k=3)
        assert len(results) == 4
        for r in results:
            assert len(r.indices) == 3

    def test_k_larger_than_db(self):
        db = _make_db(3)
        index = BruteForceIndex(db, kappa=KAPPA)
        result = index.query(_ball_point(), k=10)
        assert len(result.indices) == 3


class TestVPTree:
    def test_nearest_is_self(self):
        db = _make_db(30)
        tree = VPTree(db, kappa=KAPPA)
        result = tree.query(db[10], k=1)
        assert result.nearest_index == 10
        assert result.nearest_distance < 0.01  # asinh epsilon floor

    def test_matches_brute_force(self):
        """VP-tree should return the same nearest neighbour as brute force."""
        torch.manual_seed(123)
        db = _make_db(50)
        bf = BruteForceIndex(db, kappa=KAPPA)
        vp = VPTree(db, kappa=KAPPA)

        for _ in range(10):
            q = _ball_point()
            bf_result = bf.query(q, k=1)
            vp_result = vp.query(q, k=1)
            # The nearest neighbour should be the same
            assert bf_result.nearest_index == vp_result.nearest_index, (
                f"BF found idx={bf_result.nearest_index} d={bf_result.nearest_distance:.6f}, "
                f"VP found idx={vp_result.nearest_index} d={vp_result.nearest_distance:.6f}"
            )

    def test_top_k_sorted(self):
        db = _make_db(30)
        tree = VPTree(db, kappa=KAPPA)
        result = tree.query(_ball_point(), k=5)
        for i in range(len(result.distances) - 1):
            assert result.distances[i] <= result.distances[i + 1] + 1e-5

    def test_batch(self):
        db = _make_db(30)
        tree = VPTree(db, kappa=KAPPA)
        queries = torch.stack([_ball_point() for _ in range(5)])
        results = tree.query_batch(queries, k=3)
        assert len(results) == 5


class TestPlacementIndex:
    def test_auto_selects_brute_force(self):
        db = _make_db(20)
        index = PlacementIndex(db, kappa=KAPPA)
        assert index.strategy == "brute_force"

    def test_force_tree(self):
        db = _make_db(20)
        index = PlacementIndex(db, kappa=KAPPA, force_tree=True)
        assert index.strategy == "vp_tree"

    def test_query_works(self):
        db = _make_db(20)
        index = PlacementIndex(db, kappa=KAPPA)
        result = index.query(_ball_point(), k=3)
        assert len(result.indices) == 3
