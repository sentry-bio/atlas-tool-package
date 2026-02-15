"""Tests for taxonomy hierarchy."""
import pytest
from biosphere_atlas.hplg.taxonomy import Rank, Taxonomy


def _make_simple_taxonomy():
    """Build a small taxonomy for testing."""
    tax = Taxonomy()
    tax.add_taxon("d__Bacteria", Rank.DOMAIN, "Bacteria")
    tax.add_taxon("d__Archaea", Rank.DOMAIN, "Archaea")

    tax.add_taxon("p__Proteo", Rank.PHYLUM, "Proteobacteria", "d__Bacteria")
    tax.add_taxon("p__Firmi", Rank.PHYLUM, "Firmicutes", "d__Bacteria")
    tax.add_taxon("p__Euryarch", Rank.PHYLUM, "Euryarchaeota", "d__Archaea")

    tax.add_taxon("c__Gamma", Rank.CLASS, "Gammaproteobacteria", "p__Proteo")
    tax.add_taxon("c__Alpha", Rank.CLASS, "Alphaproteobacteria", "p__Proteo")
    tax.add_taxon("c__Bacilli", Rank.CLASS, "Bacilli", "p__Firmi")

    tax.add_taxon("g__Ecoli", Rank.GENUS, "Escherichia", "c__Gamma")
    tax.add_taxon("g__Pseudo", Rank.GENUS, "Pseudomonas", "c__Gamma")

    return tax


def test_taxonomy_size():
    tax = _make_simple_taxonomy()
    assert tax.size == 10


def test_rank_sizes():
    tax = _make_simple_taxonomy()
    sizes = tax.rank_sizes()
    assert sizes[Rank.DOMAIN] == 2
    assert sizes[Rank.PHYLUM] == 3
    assert sizes[Rank.CLASS] == 3
    assert sizes[Rank.GENUS] == 2


def test_get_children():
    tax = _make_simple_taxonomy()
    children = tax.get_children("d__Bacteria")
    child_ids = {c.taxon_id for c in children}
    assert child_ids == {"p__Proteo", "p__Firmi"}


def test_candidates_at_rank():
    tax = _make_simple_taxonomy()
    # Classes under Proteobacteria
    candidates = tax.get_candidates_at_rank(Rank.CLASS, "p__Proteo")
    cand_ids = {c.taxon_id for c in candidates}
    assert cand_ids == {"c__Gamma", "c__Alpha"}


def test_candidates_without_parent():
    tax = _make_simple_taxonomy()
    # All domains
    candidates = tax.get_candidates_at_rank(Rank.DOMAIN)
    assert len(candidates) == 2


def test_lineage():
    tax = _make_simple_taxonomy()
    lineage = tax.get_lineage("g__Ecoli")
    ranks = [n.rank for n in lineage]
    assert ranks == [Rank.DOMAIN, Rank.PHYLUM, Rank.CLASS, Rank.GENUS]
    assert lineage[0].name == "Bacteria"
    assert lineage[-1].name == "Escherichia"


def test_from_lineages():
    lineages = [
        [("domain", "Bacteria"), ("phylum", "Proteobacteria"), ("genus", "Escherichia")],
        [("domain", "Bacteria"), ("phylum", "Firmicutes"), ("genus", "Bacillus")],
        [("domain", "Archaea"), ("phylum", "Euryarchaeota")],
    ]
    tax = Taxonomy.from_lineages(lineages)
    assert tax.rank_sizes()[Rank.DOMAIN] == 2
    assert tax.rank_sizes()[Rank.PHYLUM] == 3
    assert tax.rank_sizes()[Rank.GENUS] == 2
