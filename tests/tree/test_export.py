"""
Tests for Newick, JSON, and SVG export.
"""

import json
import math
import os
import tempfile

import pytest
import torch

from biosphere_atlas.tree.hyperbolic import KAPPA_DEFAULT, _clamp_to_ball
from biosphere_atlas.tree.nj import neighbor_joining
from biosphere_atlas.tree.export import (
    to_json,
    to_newick,
    to_svg,
    write_json,
    write_newick,
    write_svg,
)
from biosphere_atlas.tree.tree_struct import PhyloTree


KAPPA = KAPPA_DEFAULT
DIM = 8


def _ball_point(dim: int = DIM, scale: float = 0.3) -> torch.Tensor:
    p = torch.randn(dim) * scale / math.sqrt(KAPPA)
    return _clamp_to_ball(p, KAPPA)


def _make_tree(n: int = 6) -> PhyloTree:
    torch.manual_seed(42)
    embs = torch.stack([_ball_point() for _ in range(n)])
    taxa = [f"taxon_{i}" for i in range(n)]
    return neighbor_joining(embs, taxa, kappa=KAPPA)


class TestNewick:
    def test_basic_newick(self):
        tree = _make_tree(4)
        nwk = to_newick(tree)
        assert nwk.endswith(";")
        # Should contain taxon names
        for i in range(4):
            assert f"taxon_{i}" in nwk

    def test_newick_has_lengths(self):
        tree = _make_tree(4)
        nwk = to_newick(tree)
        assert ":" in nwk  # Branch lengths present

    def test_newick_single_leaf(self):
        tree = PhyloTree(kappa=KAPPA)
        tree.add_leaf("only_one", _ball_point())
        nwk = to_newick(tree)
        assert "only_one" in nwk
        assert nwk.endswith(";")

    def test_write_newick(self):
        tree = _make_tree(4)
        with tempfile.NamedTemporaryFile(suffix=".nwk", delete=False) as f:
            path = f.name
        try:
            write_newick(tree, path)
            with open(path) as f:
                content = f.read()
            assert content.strip().endswith(";")
            assert "taxon_0" in content
        finally:
            os.unlink(path)


class TestJSON:
    def test_basic_json(self):
        tree = _make_tree(4)
        j = to_json(tree)
        data = json.loads(j)
        assert data["n_leaves"] == 4
        assert data["kappa"] == KAPPA
        assert len(data["nodes"]) > 0
        assert len(data["edges"]) > 0

    def test_write_json(self):
        tree = _make_tree(4)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            write_json(tree, path)
            with open(path) as f:
                data = json.load(f)
            assert data["n_leaves"] == 4
        finally:
            os.unlink(path)


class TestSVG:
    def test_basic_svg(self):
        tree = _make_tree(6)
        svg = to_svg(tree)
        assert svg.startswith("<svg")
        assert "</svg>" in svg
        # Should contain taxon labels
        for i in range(6):
            assert f"taxon_{i}" in svg

    def test_svg_dimensions(self):
        tree = _make_tree(4)
        svg = to_svg(tree, width=1000, height=600)
        assert 'width="1000"' in svg
        assert 'height="600"' in svg

    def test_svg_empty_tree(self):
        tree = PhyloTree(kappa=KAPPA)
        svg = to_svg(tree)
        assert "Empty tree" in svg

    def test_write_svg(self):
        tree = _make_tree(4)
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            path = f.name
        try:
            write_svg(tree, path)
            with open(path) as f:
                content = f.read()
            assert "<svg" in content
        finally:
            os.unlink(path)

    def test_svg_with_support(self):
        tree = _make_tree(4)
        # Add some edges with support < 1
        for edge in tree.edges():
            edge.support = 0.75
        svg = to_svg(tree, show_support=True)
        assert "0.75" in svg
