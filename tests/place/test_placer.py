"""
Tests for placement engine.
"""

import math

import pytest
import torch

from biosphere_atlas.place.hyperbolic import KAPPA_DEFAULT, _clamp_to_ball
from biosphere_atlas.place.placer import PlacementEngine, PlacementResult
from biosphere_atlas.place.reference import Rank, ReferenceDB


KAPPA = KAPPA_DEFAULT
DIM = 8


def _ball_point(dim: int = DIM, scale: float = 0.3) -> torch.Tensor:
    p = torch.randn(dim) * scale / math.sqrt(KAPPA)
    return _clamp_to_ball(p, KAPPA)


def _make_reference() -> ReferenceDB:
    """Build a small reference DB with known structure."""
    lineages = [
        ("d__Bacteria", "p__Proteobacteria", "c__Gammaproteobacteria",
         "o__Enterobacterales", "f__Enterobacteriaceae", "g__Escherichia",
         "s__Escherichia_coli"),
        ("d__Bacteria", "p__Proteobacteria", "c__Gammaproteobacteria",
         "o__Enterobacterales", "f__Enterobacteriaceae", "g__Salmonella",
         "s__Salmonella_enterica"),
        ("d__Bacteria", "p__Firmicutes", "c__Bacilli",
         "o__Lactobacillales", "f__Lactobacillaceae", "g__Lactobacillus",
         "s__Lactobacillus_acidophilus"),
        ("d__Archaea", "p__Euryarchaeota", "c__Methanobacteria",
         "o__Methanobacteriales", "f__Methanobacteriaceae",
         "g__Methanobacterium", "s__Methanobacterium_thermoautotrophicum"),
    ]
    embeddings = torch.stack([_ball_point() for _ in range(len(lineages))])
    return ReferenceDB.from_lineages(lineages, embeddings, kappa=KAPPA)


class TestPlacementEngine:
    def test_flat_placement(self):
        ref = _make_reference()
        engine = PlacementEngine(ref, kappa=KAPPA, top_k=3)

        query = _ball_point()
        result = engine.place(query, sequence_id="test_seq", mode="flat")

        assert isinstance(result, PlacementResult)
        assert result.sequence_id == "test_seq"
        assert len(result.candidates) <= 3
        assert result.best_distance >= 0
        assert result.atlas_r >= 0
        assert 0 <= result.atlas_theta < 2 * math.pi

    def test_nearest_to_itself(self):
        """A prototype's own embedding should match itself as nearest."""
        ref = _make_reference()
        engine = PlacementEngine(ref, kappa=KAPPA, top_k=3)

        # Get the embedding for E. coli
        proto = ref.get_prototype("s__Escherichia_coli")
        assert proto is not None

        result = engine.place(proto.embedding, sequence_id="ecoli", mode="flat")
        # Best candidate should be E. coli itself
        assert result.best_placement is not None
        assert result.best_placement.taxon_id == "s__Escherichia_coli"
        assert result.best_distance < 0.01  # asinh epsilon floor ~1e-4

    def test_hierarchical_placement(self):
        ref = _make_reference()
        engine = PlacementEngine(ref, kappa=KAPPA, top_k=3)

        query = _ball_point()
        result = engine.place(query, sequence_id="test_hier", mode="hierarchical")

        assert isinstance(result, PlacementResult)
        assert len(result.candidates) >= 1
        assert result.best_distance >= 0

    def test_batch_placement(self):
        ref = _make_reference()
        engine = PlacementEngine(ref, kappa=KAPPA, top_k=3)

        queries = torch.stack([_ball_point() for _ in range(5)])
        ids = [f"seq_{i}" for i in range(5)]
        results = engine.place_batch(queries, sequence_ids=ids, mode="flat")

        assert len(results) == 5
        for r, sid in zip(results, ids):
            assert r.sequence_id == sid
            assert len(r.candidates) <= 3

    def test_to_dict(self):
        ref = _make_reference()
        engine = PlacementEngine(ref, kappa=KAPPA, top_k=3)

        result = engine.place(_ball_point(), sequence_id="dict_test")
        d = result.to_dict()

        assert "sequence_id" in d
        assert "classification" in d
        assert "distance" in d
        assert "atlas_r" in d
        assert "atlas_theta" in d
        assert d["sequence_id"] == "dict_test"

    def test_lineage_string(self):
        ref = _make_reference()
        engine = PlacementEngine(ref, kappa=KAPPA, top_k=1)

        proto = ref.get_prototype("s__Escherichia_coli")
        result = engine.place(proto.embedding, sequence_id="ecoli")
        lineage = result.lineage_string
        assert "d__Bacteria" in lineage

    def test_empty_reference(self):
        ref = ReferenceDB(kappa=KAPPA, embedding_dim=DIM)
        engine = PlacementEngine(ref, kappa=KAPPA)
        result = engine.place(_ball_point(), sequence_id="empty_test")
        assert len(result.candidates) == 0
        assert result.best_distance == float("inf")
