"""
Taxonomic hierarchy definitions for HPLG classification.

The HPLG classifier navigates the taxonomic tree from domain to species,
making a three-zone decision (Accept / Escalate / Fallback) at each rank.
This module defines the rank structure and provides utilities for
hierarchical traversal.

Standard ranks: Domain > Phylum > Class > Order > Family > Genus > Species
Each rank has distinct confidence requirements and prototype update dynamics.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple


class Rank(IntEnum):
    """Taxonomic ranks in descending order (Domain=0 is broadest)."""
    DOMAIN = 0
    PHYLUM = 1
    CLASS = 2
    ORDER = 3
    FAMILY = 4
    GENUS = 5
    SPECIES = 6


RANKS = list(Rank)
RANK_NAMES = {r: r.name.lower() for r in Rank}

# Default EMA momentum per rank (species needs most stability)
DEFAULT_MOMENTUM = {
    Rank.DOMAIN: 0.90,
    Rank.PHYLUM: 0.92,
    Rank.CLASS: 0.93,
    Rank.ORDER: 0.95,
    Rank.FAMILY: 0.95,
    Rank.GENUS: 0.97,
    Rank.SPECIES: 0.99,
}

# Minimum observations before prototype is considered reliable
DEFAULT_MIN_OBS = {
    Rank.DOMAIN: 2,
    Rank.PHYLUM: 3,
    Rank.CLASS: 3,
    Rank.ORDER: 5,
    Rank.FAMILY: 5,
    Rank.GENUS: 10,
    Rank.SPECIES: 20,
}


@dataclass
class TaxonNode:
    """A node in the taxonomy tree."""
    taxon_id: str
    rank: Rank
    name: str
    parent_id: Optional[str] = None
    children: Set[str] = field(default_factory=set)

    def __hash__(self):
        return hash(self.taxon_id)


class Taxonomy:
    """
    Hierarchical taxonomy structure for HPLG classification.

    Provides:
    - Parent-child traversal (get children at rank r for parent at rank r-1)
    - Lineage lookup (domain -> phylum -> ... -> species for any taxon)
    - Rank-specific candidate filtering during classification
    """

    def __init__(self):
        self._nodes: Dict[str, TaxonNode] = {}
        self._by_rank: Dict[Rank, Dict[str, TaxonNode]] = {r: {} for r in Rank}

    def add_taxon(
        self,
        taxon_id: str,
        rank: Rank,
        name: str,
        parent_id: Optional[str] = None,
    ) -> TaxonNode:
        """Add a taxon to the taxonomy tree."""
        node = TaxonNode(
            taxon_id=taxon_id,
            rank=rank,
            name=name,
            parent_id=parent_id,
        )
        self._nodes[taxon_id] = node
        self._by_rank[rank][taxon_id] = node

        if parent_id and parent_id in self._nodes:
            self._nodes[parent_id].children.add(taxon_id)

        return node

    def get_node(self, taxon_id: str) -> Optional[TaxonNode]:
        """Retrieve a taxon node by ID."""
        return self._nodes.get(taxon_id)

    def get_children(self, taxon_id: str) -> List[TaxonNode]:
        """Get all direct children of a taxon."""
        node = self._nodes.get(taxon_id)
        if node is None:
            return []
        return [self._nodes[cid] for cid in node.children if cid in self._nodes]

    def get_candidates_at_rank(
        self,
        rank: Rank,
        parent_id: Optional[str] = None,
    ) -> List[TaxonNode]:
        """
        Get candidate taxa at a given rank, optionally filtered by parent.

        This is the core function for hierarchical HPLG classification:
        at each rank, the classifier considers only children of the
        accepted parent at the previous rank.
        """
        if parent_id is None:
            return list(self._by_rank[rank].values())

        parent = self._nodes.get(parent_id)
        if parent is None:
            return list(self._by_rank[rank].values())

        candidates = []
        for child_id in parent.children:
            child = self._nodes.get(child_id)
            if child and child.rank == rank:
                candidates.append(child)
            elif child and child.rank < rank:
                # Recurse through intermediate ranks
                candidates.extend(self._get_descendants_at_rank(child_id, rank))

        return candidates

    def _get_descendants_at_rank(self, taxon_id: str, target_rank: Rank) -> List[TaxonNode]:
        """Recursively find descendants at a target rank."""
        node = self._nodes.get(taxon_id)
        if node is None:
            return []

        if node.rank == target_rank:
            return [node]

        results = []
        for child_id in node.children:
            results.extend(self._get_descendants_at_rank(child_id, target_rank))
        return results

    def get_lineage(self, taxon_id: str) -> List[TaxonNode]:
        """Get full lineage from domain to the given taxon (inclusive)."""
        lineage = []
        current = self._nodes.get(taxon_id)
        while current is not None:
            lineage.append(current)
            current = self._nodes.get(current.parent_id) if current.parent_id else None
        lineage.reverse()
        return lineage

    def taxa_at_rank(self, rank: Rank) -> List[TaxonNode]:
        """Get all taxa at a given rank."""
        return list(self._by_rank[rank].values())

    @property
    def size(self) -> int:
        """Total number of taxa in the taxonomy."""
        return len(self._nodes)

    def rank_sizes(self) -> Dict[Rank, int]:
        """Number of taxa at each rank."""
        return {r: len(nodes) for r, nodes in self._by_rank.items()}

    @classmethod
    def from_lineages(cls, lineages: List[List[Tuple[str, str]]]) -> "Taxonomy":
        """
        Build taxonomy from a list of lineage paths.

        Each lineage is a list of (rank_name, taxon_name) tuples from domain to species:
            [("domain", "Bacteria"), ("phylum", "Proteobacteria"), ...]

        This matches GTDB-style lineage strings.
        """
        tax = cls()
        rank_map = {r.name.lower(): r for r in Rank}

        for lineage in lineages:
            parent_id = None
            for rank_name, taxon_name in lineage:
                rank = rank_map.get(rank_name.lower())
                if rank is None:
                    continue

                taxon_id = f"{rank_name[0]}__{taxon_name}"
                if taxon_id not in tax._nodes:
                    tax.add_taxon(taxon_id, rank, taxon_name, parent_id)
                parent_id = taxon_id

        return tax
