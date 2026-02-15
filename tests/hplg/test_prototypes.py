"""Tests for dual-bank prototype system."""
import pytest
import torch
from biosphere_atlas.hplg.prototypes import DualBankPrototypes
from biosphere_atlas.hplg.taxonomy import Rank
from biosphere_atlas.hplg.hyperbolic import (
    KAPPA_DEFAULT,
    ball_radius,
    poincare_distance,
)


def _make_ball_point(dim=16, scale=0.3, kappa=KAPPA_DEFAULT):
    """Create a random point inside the Poincare ball."""
    R = ball_radius(kappa)
    x = torch.randn(dim)
    x = x / x.norm() * scale * R
    return x


def _make_bank(dim=16):
    """Create a prototype bank with a few registered prototypes."""
    bank = DualBankPrototypes(embedding_dim=dim)
    bank.register_prototype("d__Bacteria", Rank.DOMAIN, _make_ball_point(dim))
    bank.register_prototype("d__Archaea", Rank.DOMAIN, _make_ball_point(dim))
    bank.register_prototype("p__Proteo", Rank.PHYLUM, _make_ball_point(dim))
    bank.register_prototype("g__Ecoli", Rank.GENUS, _make_ball_point(dim))
    bank.register_prototype("s__Ecoli_K12", Rank.SPECIES, _make_ball_point(dim))
    return bank


def test_register_prototype():
    bank = DualBankPrototypes(embedding_dim=16)
    emb = _make_ball_point(16)
    bank.register_prototype("test_taxon", Rank.GENUS, emb)
    assert bank.num_prototypes == 1
    retrieved = bank.get_embedding("test_taxon")
    assert retrieved is not None
    assert retrieved.shape == (16,)


def test_register_from_exemplars():
    """Registration from a batch computes Karcher mean."""
    bank = DualBankPrototypes(embedding_dim=16)
    exemplars = torch.stack([_make_ball_point(16) for _ in range(5)])
    bank.register_prototype("test_mean", Rank.FAMILY, exemplars)
    retrieved = bank.get_embedding("test_mean")
    assert retrieved is not None
    # Mean should be inside the ball
    R = ball_radius(KAPPA_DEFAULT)
    assert retrieved.norm().item() < R


def test_dual_banks_independent():
    """Teacher and student banks are initially identical but independent."""
    bank = _make_bank()
    teacher = bank.get_embedding("d__Bacteria", bank="teacher")
    student = bank.get_embedding("d__Bacteria", bank="student")
    assert torch.allclose(teacher, student)


def test_update_moves_student():
    bank = _make_bank()
    original = bank.get_embedding("g__Ecoli").clone()
    new_point = _make_ball_point(16)

    bank.update("g__Ecoli", new_point, force=True)
    updated = bank.get_embedding("g__Ecoli")

    # Student should have moved (unless momentum is exactly 1)
    # With genus momentum 0.97, it moves 3% toward new point
    assert not torch.allclose(original, updated, atol=1e-6)


def test_update_preserves_teacher():
    bank = _make_bank()
    teacher_before = bank.get_embedding("g__Ecoli", bank="teacher").clone()
    new_point = _make_ball_point(16)

    bank.update("g__Ecoli", new_point, force=True)
    teacher_after = bank.get_embedding("g__Ecoli", bank="teacher")

    # Teacher should be unchanged
    assert torch.allclose(teacher_before, teacher_after)


def test_reanchor_pulls_toward_teacher():
    bank = _make_bank()

    # Move student away from teacher
    for _ in range(20):
        bank.update("g__Ecoli", _make_ball_point(16), force=True)

    student_before = bank.get_embedding("g__Ecoli").clone()
    teacher = bank.get_embedding("g__Ecoli", bank="teacher")
    dist_before = poincare_distance(
        student_before.unsqueeze(0), teacher.unsqueeze(0), KAPPA_DEFAULT
    ).item()

    # Reanchor
    bank.reanchor()

    student_after = bank.get_embedding("g__Ecoli")
    dist_after = poincare_distance(
        student_after.unsqueeze(0), teacher.unsqueeze(0), KAPPA_DEFAULT
    ).item()

    # Should be closer to teacher after reanchoring
    assert dist_after <= dist_before + 1e-6


def test_get_all_embeddings():
    bank = _make_bank()
    ids = ["d__Bacteria", "d__Archaea", "p__Proteo"]
    embeddings = bank.get_all_embeddings(ids)
    assert embeddings.shape == (3, 16)


def test_reliability():
    bank = _make_bank()
    # No observations yet
    assert not bank.is_reliable("s__Ecoli_K12")
    # Species needs 20 observations
    for _ in range(20):
        bank.update("s__Ecoli_K12", _make_ball_point(16), force=True)
    assert bank.is_reliable("s__Ecoli_K12")


def test_state_dict_roundtrip():
    bank = _make_bank()
    state = bank.state_dict()
    loaded = DualBankPrototypes.from_state_dict(state)
    assert loaded.num_prototypes == bank.num_prototypes
    for tid in bank.taxon_ids():
        orig = bank.get_embedding(tid)
        loaded_emb = loaded.get_embedding(tid)
        assert torch.allclose(orig, loaded_emb)
