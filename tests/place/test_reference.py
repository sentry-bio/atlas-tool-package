"""
Tests for reference database.
"""

import math
import tempfile
from pathlib import Path

import pytest
import torch

from biosphere_atlas.place.hyperbolic import KAPPA_DEFAULT, _clamp_to_ball, ball_radius
from biosphere_atlas.place.reference import Rank, ReferenceDB


KAPPA = KAPPA_DEFAULT
DIM = 8


def _ball_point(dim: int = DIM, scale: float = 0.5) -> torch.Tensor:
    p = torch.randn(dim) * scale / math.sqrt(KAPPA)
    return _clamp_to_ball(p, KAPPA)


def _make_lineage(domain="d__Bac", phylum="p__Pro", cls="c__Gam",
                  order="o__Ent", family="f__Ent", genus="g__Esc",
                  species="s__Ecoli"):
    return (domain, phylum, cls, order, family, genus, species)


class TestReferenceDB:
    def test_add_and_retrieve(self):
        db = ReferenceDB(kappa=KAPPA, embedding_dim=DIM)
        emb = _ball_point()
        lineage = _make_lineage()
        db.add_prototype("s__Ecoli", Rank.SPECIES, emb, lineage)

        assert db.size == 1
        assert "s__Ecoli" in db
        proto = db.get_prototype("s__Ecoli")
        assert proto is not None
        assert proto.rank == Rank.SPECIES
        assert torch.allclose(proto.embedding, emb, atol=1e-4)

    def test_aggregation_on_duplicate(self):
        db = ReferenceDB(kappa=KAPPA, embedding_dim=DIM)
        lineage = _make_lineage()
        e1 = _ball_point()
        e2 = _ball_point()
        db.add_prototype("s__Ecoli", Rank.SPECIES, e1, lineage)
        db.add_prototype("s__Ecoli", Rank.SPECIES, e2, lineage)

        assert db.size == 1
        proto = db.get_prototype("s__Ecoli")
        assert proto.observation_count == 2
        # Aggregated embedding should be inside ball
        assert proto.embedding.norm().item() < ball_radius(KAPPA)

    def test_rank_lookup(self):
        db = ReferenceDB(kappa=KAPPA, embedding_dim=DIM)
        lineage = _make_lineage()
        db.add_prototype("s__Ecoli", Rank.SPECIES, _ball_point(), lineage)
        db.add_prototype("g__Esc", Rank.GENUS, _ball_point(), lineage[:6])

        ids, embs = db.get_prototypes_at_rank(Rank.SPECIES)
        assert len(ids) == 1
        assert ids[0] == "s__Ecoli"
        assert embs.shape == (1, DIM)

        ids, embs = db.get_prototypes_at_rank(Rank.GENUS)
        assert len(ids) == 1

    def test_from_lineages(self):
        lineages = [
            _make_lineage(species="s__Ecoli"),
            _make_lineage(species="s__Salmonella"),
            _make_lineage(phylum="p__Firm", cls="c__Bac",
                          order="o__Lac", family="f__Lac",
                          genus="g__Lac", species="s__Lactobacillus"),
        ]
        embs = torch.stack([_ball_point() for _ in range(3)])
        db = ReferenceDB.from_lineages(lineages, embs, kappa=KAPPA)

        # Should have species-level prototypes
        ids, _ = db.get_prototypes_at_rank(Rank.SPECIES)
        assert len(ids) == 3

        # Should have aggregated higher ranks
        ids, _ = db.get_prototypes_at_rank(Rank.DOMAIN)
        assert len(ids) >= 1  # d__Bac

    def test_serialization_json(self, tmp_path):
        db = ReferenceDB(kappa=KAPPA, embedding_dim=DIM)
        db.add_prototype("s__Ecoli", Rank.SPECIES, _ball_point(),
                         _make_lineage())

        path = tmp_path / "ref.json"
        db.save(path)
        db2 = ReferenceDB.load(path)

        assert db2.size == 1
        assert "s__Ecoli" in db2
        assert db2.kappa == KAPPA

    def test_serialization_pickle(self, tmp_path):
        db = ReferenceDB(kappa=KAPPA, embedding_dim=DIM)
        db.add_prototype("s__Ecoli", Rank.SPECIES, _ball_point(),
                         _make_lineage())

        path = tmp_path / "ref.pkl"
        db.save(path)
        db2 = ReferenceDB.load(path)

        assert db2.size == 1
        proto = db2.get_prototype("s__Ecoli")
        assert proto is not None

    def test_summary(self):
        lineages = [
            _make_lineage(species="s__Ecoli"),
            _make_lineage(species="s__Salmonella"),
        ]
        embs = torch.stack([_ball_point() for _ in range(2)])
        db = ReferenceDB.from_lineages(lineages, embs, kappa=KAPPA)

        summary = db.summary()
        assert "species" in summary
        assert summary["species"] == 2

    def test_get_children(self):
        lineages = [
            _make_lineage(genus="g__Esc", species="s__Ecoli"),
            _make_lineage(genus="g__Esc", species="s__Efergusonii"),
            _make_lineage(genus="g__Sal", species="s__Salmonella"),
        ]
        embs = torch.stack([_ball_point() for _ in range(3)])
        db = ReferenceDB.from_lineages(lineages, embs, kappa=KAPPA)

        children = db.get_children("g__Esc", Rank.SPECIES)
        assert len(children) == 2
        assert "s__Ecoli" in children
        assert "s__Efergusonii" in children
