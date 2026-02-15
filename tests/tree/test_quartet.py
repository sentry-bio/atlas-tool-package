"""
Tests for quartet consistency checking.
"""

import math

import pytest
import torch

from biosphere_atlas.tree.hyperbolic import KAPPA_DEFAULT, _clamp_to_ball, poincare_distance
from biosphere_atlas.tree.nj import neighbor_joining
from biosphere_atlas.tree.quartet import (
    ConsistencyReport,
    Topology,
    check_quartet_consistency,
    coordinate_quartet_topology,
    four_point_delta,
    quartet_topology_from_distances,
    tree_quartet_topology,
)
from biosphere_atlas.tree.tree_struct import PhyloTree


KAPPA = KAPPA_DEFAULT
DIM = 8


def _ball_point(dim: int = DIM, scale: float = 0.3) -> torch.Tensor:
    p = torch.randn(dim) * scale / math.sqrt(KAPPA)
    return _clamp_to_ball(p, KAPPA)


class TestQuartetTopology:
    def test_ab_cd_topology(self):
        """When d(a,b) + d(c,d) is smallest, topology should be AB_CD."""
        # d(a,b) = 1, d(c,d) = 1, d(a,c) = 5, d(b,d) = 5, d(a,d) = 5, d(b,c) = 5
        # S1 = 1+1 = 2, S2 = 5+5 = 10, S3 = 5+5 = 10
        topo, delta = quartet_topology_from_distances(1, 5, 5, 5, 5, 1)
        assert topo == Topology.AB_CD

    def test_ac_bd_topology(self):
        """When d(a,c) + d(b,d) is smallest, topology should be AC_BD."""
        topo, delta = quartet_topology_from_distances(5, 1, 5, 5, 1, 5)
        assert topo == Topology.AC_BD

    def test_ad_bc_topology(self):
        """When d(a,d) + d(b,c) is smallest, topology should be AD_BC."""
        topo, delta = quartet_topology_from_distances(5, 5, 1, 1, 5, 5)
        assert topo == Topology.AD_BC

    def test_perfect_tree_delta_zero(self):
        """For a perfect tree metric, the two largest sums are equal -> delta=0."""
        # Perfect tree metric: d(a,b)=2, d(c,d)=2, d(a,c)=d(a,d)=d(b,c)=d(b,d)=3
        # S1 = 2+2 = 4, S2 = 3+3 = 6, S3 = 3+3 = 6
        _, delta = quartet_topology_from_distances(2, 3, 3, 3, 3, 2)
        assert delta < 1e-6

    def test_delta_non_negative(self):
        for _ in range(10):
            dists = [abs(torch.randn(1).item()) + 0.1 for _ in range(6)]
            _, delta = quartet_topology_from_distances(*dists)
            assert delta >= -1e-10


class TestCoordinateTopology:
    def test_well_separated_clusters(self):
        """Two pairs of close points should give clear topology."""
        torch.manual_seed(42)
        # Create two clusters
        center_1 = _ball_point(scale=0.1)
        center_2 = _ball_point(scale=0.1)
        a = _clamp_to_ball(center_1 + torch.randn(DIM) * 0.01, KAPPA)
        b = _clamp_to_ball(center_1 + torch.randn(DIM) * 0.01, KAPPA)
        c = _clamp_to_ball(center_2 + torch.randn(DIM) * 0.01, KAPPA)
        d = _clamp_to_ball(center_2 + torch.randn(DIM) * 0.01, KAPPA)

        topo, delta = coordinate_quartet_topology(a, b, c, d, KAPPA)
        # a,b are close together and c,d are close together -> AB_CD
        assert topo == Topology.AB_CD

    def test_returns_valid_topology(self):
        torch.manual_seed(99)
        for _ in range(10):
            a, b, c, d = [_ball_point() for _ in range(4)]
            topo, delta = coordinate_quartet_topology(a, b, c, d, KAPPA)
            assert topo in (Topology.AB_CD, Topology.AC_BD, Topology.AD_BC)
            assert delta >= -1e-10


class TestQuartetConsistency:
    def test_nj_tree_consistency(self):
        """NJ tree should have reasonable quartet consistency with its source coordinates."""
        torch.manual_seed(42)
        n = 8
        embs = torch.stack([_ball_point() for _ in range(n)])
        taxa = [f"t_{i}" for i in range(n)]
        tree = neighbor_joining(embs, taxa, kappa=KAPPA)

        emb_map = {}
        for leaf in tree.leaves():
            if leaf.embedding is not None:
                emb_map[leaf.node_id] = leaf.embedding

        report = check_quartet_consistency(tree, emb_map, kappa=KAPPA)
        assert isinstance(report, ConsistencyReport)
        assert report.total_quartets > 0
        # NJ should maintain reasonable consistency
        assert report.consistency_fraction >= 0.5

    def test_too_few_leaves(self):
        """With < 4 leaves, no quartets can be checked."""
        emb_map = {0: _ball_point(), 1: _ball_point(), 2: _ball_point()}
        tree = PhyloTree(kappa=KAPPA)
        report = check_quartet_consistency(tree, emb_map, kappa=KAPPA)
        assert report.total_quartets == 0
        assert report.consistency_fraction == 1.0

    def test_report_summary(self):
        torch.manual_seed(42)
        embs = torch.stack([_ball_point() for _ in range(6)])
        taxa = [f"t_{i}" for i in range(6)]
        tree = neighbor_joining(embs, taxa, kappa=KAPPA)
        emb_map = {leaf.node_id: leaf.embedding for leaf in tree.leaves() if leaf.embedding is not None}
        report = check_quartet_consistency(tree, emb_map, kappa=KAPPA)
        s = report.summary()
        assert "QuartetConsistency" in s
        assert "consistent=" in s

    def test_max_quartets_sampling(self):
        torch.manual_seed(42)
        n = 10
        embs = torch.stack([_ball_point() for _ in range(n)])
        taxa = [f"t_{i}" for i in range(n)]
        tree = neighbor_joining(embs, taxa, kappa=KAPPA)
        emb_map = {leaf.node_id: leaf.embedding for leaf in tree.leaves() if leaf.embedding is not None}

        report = check_quartet_consistency(
            tree, emb_map, kappa=KAPPA, max_quartets=10, seed=42
        )
        assert report.total_quartets <= 10


class TestFourPointDelta:
    def test_basic(self):
        torch.manual_seed(42)
        embs = torch.stack([_ball_point() for _ in range(8)])
        delta = four_point_delta(embs, kappa=KAPPA, max_quartets=50)
        assert delta >= 0.0

    def test_few_points(self):
        embs = torch.stack([_ball_point() for _ in range(3)])
        delta = four_point_delta(embs, kappa=KAPPA)
        assert delta == 0.0  # Can't form quartets
