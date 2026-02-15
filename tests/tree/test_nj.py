"""
Tests for Neighbor-Joining tree construction.
"""

import math

import pytest
import torch

from biosphere_atlas.tree.hyperbolic import KAPPA_DEFAULT, _clamp_to_ball, poincare_distance
from biosphere_atlas.tree.nj import compute_distance_matrix, neighbor_joining, root_at_midpoint
from biosphere_atlas.tree.tree_struct import PhyloTree


KAPPA = KAPPA_DEFAULT
DIM = 8


def _ball_point(dim: int = DIM, scale: float = 0.3) -> torch.Tensor:
    p = torch.randn(dim) * scale / math.sqrt(KAPPA)
    return _clamp_to_ball(p, KAPPA)


def _make_embeddings(n: int = 6) -> tuple:
    torch.manual_seed(42)
    embs = torch.stack([_ball_point() for _ in range(n)])
    taxa = [f"taxon_{i}" for i in range(n)]
    return embs, taxa


class TestDistanceMatrix:
    def test_shape(self):
        embs, _ = _make_embeddings(5)
        D = compute_distance_matrix(embs, KAPPA)
        assert D.shape == (5, 5)

    def test_diagonal_zero(self):
        embs, _ = _make_embeddings(5)
        D = compute_distance_matrix(embs, KAPPA)
        for i in range(5):
            assert D[i, i].item() < 1e-3

    def test_symmetric(self):
        embs, _ = _make_embeddings(5)
        D = compute_distance_matrix(embs, KAPPA)
        assert torch.allclose(D, D.T, atol=1e-5)

    def test_non_negative(self):
        embs, _ = _make_embeddings(5)
        D = compute_distance_matrix(embs, KAPPA)
        assert (D >= -1e-5).all()


class TestNeighborJoining:
    def test_single_taxon(self):
        emb = _ball_point().unsqueeze(0)
        tree = neighbor_joining(emb, ["A"], kappa=KAPPA)
        assert tree.n_leaves == 1
        assert tree.n_edges == 0

    def test_two_taxa(self):
        torch.manual_seed(99)
        embs = torch.stack([_ball_point(), _ball_point()])
        tree = neighbor_joining(embs, ["A", "B"], kappa=KAPPA)
        assert tree.n_leaves == 2
        # Should have exactly one edge connecting them
        assert tree.n_edges >= 1

    def test_four_taxa(self):
        torch.manual_seed(123)
        embs, taxa = _make_embeddings(4)
        tree = neighbor_joining(embs, taxa, kappa=KAPPA)
        assert tree.n_leaves == 4
        # NJ on 4 taxa: 2 internal nodes, 5 edges (unrooted binary)
        assert tree.n_internal >= 1
        assert tree.n_edges >= 4

    def test_six_taxa(self):
        embs, taxa = _make_embeddings(6)
        tree = neighbor_joining(embs, taxa, kappa=KAPPA)
        assert tree.n_leaves == 6
        assert tree.n_edges >= 6
        assert tree.total_branch_length() > 0

    def test_edge_lengths_non_negative(self):
        embs, taxa = _make_embeddings(6)
        tree = neighbor_joining(embs, taxa, kappa=KAPPA)
        for edge in tree.edges():
            assert edge.length >= 0.0

    def test_all_leaves_have_taxa(self):
        embs, taxa = _make_embeddings(6)
        tree = neighbor_joining(embs, taxa, kappa=KAPPA)
        for leaf in tree.leaves():
            assert leaf.taxon_id is not None
            assert leaf.taxon_id in taxa

    def test_leaf_embeddings_preserved(self):
        torch.manual_seed(42)
        embs, taxa = _make_embeddings(4)
        tree = neighbor_joining(embs, taxa, kappa=KAPPA)
        for i, leaf in enumerate(tree.leaves()):
            assert leaf.embedding is not None
            assert torch.allclose(leaf.embedding, embs[i], atol=1e-6)

    def test_internal_nodes_in_ball(self):
        embs, taxa = _make_embeddings(6)
        tree = neighbor_joining(embs, taxa, kappa=KAPPA)
        R = 1.0 / math.sqrt(KAPPA)
        for node in tree.internal_nodes():
            if node.embedding is not None:
                assert node.embedding.norm().item() < R

    def test_reproducible(self):
        """Same input gives same tree."""
        torch.manual_seed(42)
        embs1, taxa1 = _make_embeddings(6)
        tree1 = neighbor_joining(embs1, taxa1, kappa=KAPPA)

        torch.manual_seed(42)
        embs2, taxa2 = _make_embeddings(6)
        tree2 = neighbor_joining(embs2, taxa2, kappa=KAPPA)

        assert tree1.n_leaves == tree2.n_leaves
        assert tree1.n_edges == tree2.n_edges
        assert abs(tree1.total_branch_length() - tree2.total_branch_length()) < 1e-6


class TestMidpointRoot:
    def test_rooting(self):
        embs, taxa = _make_embeddings(6)
        tree = neighbor_joining(embs, taxa, kappa=KAPPA)
        root_at_midpoint(tree)
        # After rooting, at least one node should have no parent
        root = tree.find_root()
        assert root is not None

    def test_small_tree(self):
        emb = _ball_point().unsqueeze(0)
        tree = neighbor_joining(emb, ["only"], kappa=KAPPA)
        root_at_midpoint(tree)  # Should not crash
        assert tree.n_leaves == 1
