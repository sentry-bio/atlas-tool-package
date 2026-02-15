"""
Tests for PhyloTree data structure.
"""

import math

import pytest
import torch

from biosphere_atlas.tree.hyperbolic import KAPPA_DEFAULT, _clamp_to_ball, poincare_distance
from biosphere_atlas.tree.tree_struct import PhyloTree, TreeNode, TreeEdge


KAPPA = KAPPA_DEFAULT
DIM = 8


def _ball_point(dim: int = DIM, scale: float = 0.3) -> torch.Tensor:
    p = torch.randn(dim) * scale / math.sqrt(KAPPA)
    return _clamp_to_ball(p, KAPPA)


class TestPhyloTree:
    def test_add_leaf(self):
        tree = PhyloTree(kappa=KAPPA)
        emb = _ball_point()
        nid = tree.add_leaf("E_coli", emb)
        assert tree.n_leaves == 1
        assert tree.n_internal == 0
        assert tree.get_node(nid).taxon_id == "E_coli"
        assert tree.get_node(nid).is_leaf

    def test_add_internal(self):
        tree = PhyloTree(kappa=KAPPA)
        l1 = tree.add_leaf("A", _ball_point())
        l2 = tree.add_leaf("B", _ball_point())
        iid = tree.add_internal(children=[l1, l2], embedding=_ball_point())
        assert tree.n_internal == 1
        assert tree.n_leaves == 2
        assert not tree.get_node(iid).is_leaf
        assert tree.get_node(l1).parent == iid
        assert tree.get_node(l2).parent == iid

    def test_add_edge(self):
        tree = PhyloTree(kappa=KAPPA)
        l1 = tree.add_leaf("A", _ball_point())
        l2 = tree.add_leaf("B", _ball_point())
        tree.add_edge(l1, l2, 0.5)
        assert tree.n_edges == 1
        edge = tree.get_edge(l1, l2)
        assert edge is not None
        assert abs(edge.length - 0.5) < 1e-6

    def test_connect_computes_distance(self):
        tree = PhyloTree(kappa=KAPPA)
        emb_a = _ball_point()
        emb_b = _ball_point()
        l1 = tree.add_leaf("A", emb_a)
        l2 = tree.add_leaf("B", emb_b)
        tree.connect(l1, l2)
        edge = tree.get_edge(l1, l2)
        expected = poincare_distance(emb_a, emb_b, KAPPA).item()
        assert edge is not None
        assert abs(edge.length - expected) < 1e-4

    def test_path_distance(self):
        tree = PhyloTree(kappa=KAPPA)
        a = tree.add_leaf("A", _ball_point())
        b = tree.add_leaf("B", _ball_point())
        c = tree.add_leaf("C", _ball_point())
        i = tree.add_internal(children=[a, b], embedding=_ball_point())
        tree.add_edge(i, a, 1.0)
        tree.add_edge(i, b, 2.0)
        tree.add_edge(i, c, 3.0)
        # Path from A to B goes through I: 1.0 + 2.0 = 3.0
        assert abs(tree.path_distance(a, b) - 3.0) < 1e-6
        # Path from A to C: 1.0 + 3.0 = 4.0
        assert abs(tree.path_distance(a, c) - 4.0) < 1e-6
        # Self-distance
        assert tree.path_distance(a, a) == 0.0

    def test_total_branch_length(self):
        tree = PhyloTree(kappa=KAPPA)
        a = tree.add_leaf("A", _ball_point())
        b = tree.add_leaf("B", _ball_point())
        tree.add_edge(a, b, 1.5)
        assert abs(tree.total_branch_length() - 1.5) < 1e-6

    def test_leaves_and_internal(self):
        tree = PhyloTree(kappa=KAPPA)
        a = tree.add_leaf("A", _ball_point())
        b = tree.add_leaf("B", _ball_point())
        i = tree.add_internal(children=[a, b])
        leaves = tree.leaves()
        internals = tree.internal_nodes()
        assert len(leaves) == 2
        assert len(internals) == 1
        assert all(l.is_leaf for l in leaves)
        assert not internals[0].is_leaf

    def test_to_dict(self):
        tree = PhyloTree(kappa=KAPPA)
        a = tree.add_leaf("A", _ball_point())
        b = tree.add_leaf("B", _ball_point())
        tree.add_edge(a, b, 0.7)
        d = tree.to_dict()
        assert d["n_leaves"] == 2
        assert d["kappa"] == KAPPA
        assert len(d["nodes"]) == 2
        assert len(d["edges"]) == 1

    def test_summary(self):
        tree = PhyloTree(kappa=KAPPA)
        tree.add_leaf("A", _ball_point())
        s = tree.summary()
        assert "PhyloTree" in s
        assert "leaves=1" in s

    def test_find_root(self):
        tree = PhyloTree(kappa=KAPPA)
        a = tree.add_leaf("A", _ball_point())
        b = tree.add_leaf("B", _ball_point())
        i = tree.add_internal(children=[a, b])
        assert tree.find_root() == i  # Only node with no parent

    def test_root_at(self):
        tree = PhyloTree(kappa=KAPPA)
        a = tree.add_leaf("A", _ball_point())
        b = tree.add_leaf("B", _ball_point())
        i = tree.add_internal(children=[a, b])
        tree.add_edge(i, a, 1.0)
        tree.add_edge(i, b, 2.0)
        # Re-root at leaf A
        tree.root_at(a)
        assert tree.get_node(a).parent is None
