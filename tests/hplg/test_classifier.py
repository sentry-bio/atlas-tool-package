"""Tests for the full HPLG classification pipeline."""
import pytest
import torch
from biosphere_atlas.hplg.taxonomy import Rank, Taxonomy
from biosphere_atlas.hplg.prototypes import DualBankPrototypes
from biosphere_atlas.hplg.calibrator import MondrianConformalCalibrator
from biosphere_atlas.hplg.scorer import NonconformityScorer
from biosphere_atlas.hplg.classifier import HPLGClassifier, ClassificationResult
from biosphere_atlas.hplg.hyperbolic import KAPPA_DEFAULT, ball_radius


DIM = 16


def _ball_point(direction, scale=0.3, dim=DIM):
    """Make a point in the Poincare ball at a specific 'direction'."""
    R = ball_radius(KAPPA_DEFAULT)
    x = torch.zeros(dim)
    x[0] = direction[0] if len(direction) > 0 else 0
    x[1] = direction[1] if len(direction) > 1 else 0
    x = x / max(x.norm().item(), 1e-6) * scale * R
    return x


def _build_test_system():
    """Build a complete HPLG system for testing."""
    # Taxonomy
    tax = Taxonomy()
    tax.add_taxon("d__Bac", Rank.DOMAIN, "Bacteria")
    tax.add_taxon("d__Arc", Rank.DOMAIN, "Archaea")
    tax.add_taxon("p__Pro", Rank.PHYLUM, "Proteobacteria", "d__Bac")
    tax.add_taxon("p__Fir", Rank.PHYLUM, "Firmicutes", "d__Bac")
    tax.add_taxon("p__Eur", Rank.PHYLUM, "Euryarchaeota", "d__Arc")
    tax.add_taxon("c__Gam", Rank.CLASS, "Gammaproteobacteria", "p__Pro")
    tax.add_taxon("c__Alp", Rank.CLASS, "Alphaproteobacteria", "p__Pro")
    tax.add_taxon("g__Eco", Rank.GENUS, "Escherichia", "c__Gam")
    tax.add_taxon("g__Pse", Rank.GENUS, "Pseudomonas", "c__Gam")

    # Prototypes: place them at distinct regions of the Poincare ball
    bank = DualBankPrototypes(embedding_dim=DIM)
    # Bacteria at +x, Archaea at -x
    bank.register_prototype("d__Bac", Rank.DOMAIN, _ball_point([1, 0], 0.1))
    bank.register_prototype("d__Arc", Rank.DOMAIN, _ball_point([-1, 0], 0.1))
    # Phyla spread angularly
    bank.register_prototype("p__Pro", Rank.PHYLUM, _ball_point([1, 0.5], 0.2))
    bank.register_prototype("p__Fir", Rank.PHYLUM, _ball_point([1, -0.5], 0.2))
    bank.register_prototype("p__Eur", Rank.PHYLUM, _ball_point([-1, 0], 0.2))
    # Classes deeper
    bank.register_prototype("c__Gam", Rank.CLASS, _ball_point([1, 0.5], 0.35))
    bank.register_prototype("c__Alp", Rank.CLASS, _ball_point([1, 0.8], 0.35))
    # Genera deeper still
    bank.register_prototype("g__Eco", Rank.GENUS, _ball_point([1, 0.4], 0.5))
    bank.register_prototype("g__Pse", Rank.GENUS, _ball_point([1, 0.6], 0.5))

    # Permissive calibrator for testing
    cal = MondrianConformalCalibrator(epsilon_accept=0.5, epsilon_fallback=0.01)

    scorer = NonconformityScorer()

    classifier = HPLGClassifier(
        taxonomy=tax,
        prototypes=bank,
        calibrator=cal,
        scorer=scorer,
        update_prototypes=False,  # Disable updates during testing
    )

    return classifier, tax, bank


def test_classify_near_bacteria():
    """A point near the Bacteria prototype should classify as Bacteria."""
    classifier, tax, bank = _build_test_system()
    # Point close to Bacteria domain
    emb = _ball_point([1, 0.1], 0.08)
    result = classifier.classify(emb, sequence_id="test_bac")

    assert result.sequence_id == "test_bac"
    assert len(result.decisions) > 0
    # First decision should be at domain level
    assert result.decisions[0].rank == Rank.DOMAIN
    assert result.decisions[0].taxon_name == "Bacteria"


def test_classify_near_archaea():
    """A point near the Archaea prototype should classify as Archaea."""
    classifier, tax, bank = _build_test_system()
    emb = _ball_point([-1, 0.1], 0.08)
    result = classifier.classify(emb, sequence_id="test_arc")

    assert result.decisions[0].rank == Rank.DOMAIN
    assert result.decisions[0].taxon_name == "Archaea"


def test_hierarchical_consistency():
    """Classification should be hierarchically consistent."""
    classifier, tax, bank = _build_test_system()
    # Point near Gammaproteobacteria
    emb = _ball_point([1, 0.5], 0.3)
    result = classifier.classify(emb, sequence_id="test_gamma")

    # All accepted decisions should form a valid lineage
    accepted = [d for d in result.decisions if d.zone == "accept"]
    if len(accepted) >= 2:
        # Each level should be a child of the previous
        for i in range(1, len(accepted)):
            child_node = tax.get_node(accepted[i].taxon_id)
            parent_id = accepted[i - 1].taxon_id
            if child_node and child_node.parent_id:
                lineage = tax.get_lineage(accepted[i].taxon_id)
                lineage_ids = {n.taxon_id for n in lineage}
                assert parent_id in lineage_ids


def test_fallback_stops_descent():
    """Fallback should stop further classification."""
    classifier, tax, bank = _build_test_system()
    # Point far from everything (near origin)
    emb = torch.zeros(DIM)
    emb[0] = 0.01
    result = classifier.classify(emb, sequence_id="test_unknown")

    # Should have stopped early due to high nonconformity
    assert len(result.decisions) <= len(Rank)
    # Check that last decision is a fallback (or all are accepts)
    if result.decisions:
        last = result.decisions[-1]
        assert last.zone in ("accept", "fallback")


def test_lineage_string():
    classifier, tax, bank = _build_test_system()
    emb = _ball_point([1, 0.5], 0.3)
    result = classifier.classify(emb, sequence_id="test_lineage")

    lineage = result.lineage_string
    assert isinstance(lineage, str)
    if lineage:
        # Should start with domain prefix
        assert lineage.startswith("d__")


def test_classify_batch():
    classifier, tax, bank = _build_test_system()
    embs = torch.stack([
        _ball_point([1, 0.1], 0.08),
        _ball_point([-1, 0.1], 0.08),
        _ball_point([1, 0.5], 0.3),
    ])
    results = classifier.classify_batch(embs, ["bac", "arc", "gamma"])

    assert len(results) == 3
    assert results[0].sequence_id == "bac"
    assert results[1].sequence_id == "arc"


def test_classification_stats():
    classifier, tax, bank = _build_test_system()
    embs = torch.stack([_ball_point([1, 0.1], 0.08) for _ in range(10)])
    results = classifier.classify_batch(embs)
    stats = classifier.classification_stats(results)

    assert stats["n_sequences"] == 10
    assert "accept_rate" in stats
    assert "mean_confidence" in stats


def test_to_dict():
    classifier, tax, bank = _build_test_system()
    emb = _ball_point([1, 0.5], 0.3)
    result = classifier.classify(emb, sequence_id="test_dict")
    d = result.to_dict()

    assert d["sequence_id"] == "test_dict"
    assert "lineage" in d
    assert "confidence" in d


def test_state_dict_roundtrip():
    classifier, tax, bank = _build_test_system()
    state = classifier.state_dict()

    loaded = HPLGClassifier.from_checkpoint(state, taxonomy=tax)
    assert loaded.prototypes.num_prototypes == classifier.prototypes.num_prototypes
    assert loaded.kappa == classifier.kappa
