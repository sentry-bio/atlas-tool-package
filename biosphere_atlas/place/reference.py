"""
Reference database for phylogenetic placement.
===============================================

Manages a bank of prototype embeddings (taxon → Poincaré ball embedding)
with associated taxonomy.  Supports serialization, incremental updates,
and construction from GTDB-style lineage files.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from biosphere_atlas.core.hyperbolic import KAPPA_DEFAULT, karcher_mean, poincare_distance
from biosphere_atlas.core.hyperbolic import _clamp_to_ball


# ── Taxonomy ranks (GTDB convention) ────────────────────────────────────────

class Rank(IntEnum):
    DOMAIN = 0
    PHYLUM = 1
    CLASS = 2
    ORDER = 3
    FAMILY = 4
    GENUS = 5
    SPECIES = 6


RANK_PREFIXES = {
    Rank.DOMAIN: "d__",
    Rank.PHYLUM: "p__",
    Rank.CLASS: "c__",
    Rank.ORDER: "o__",
    Rank.FAMILY: "f__",
    Rank.GENUS: "g__",
    Rank.SPECIES: "s__",
}

RANKS = list(Rank)


# ── Prototype entry ──────────────────────────────────────────────────────────

@dataclass
class Prototype:
    """A single reference prototype in the atlas."""

    taxon_id: str
    """Unique identifier (e.g. 's__Escherichia coli')."""

    rank: Rank
    """Taxonomic rank of this prototype."""

    embedding: Tensor
    """Poincaré ball embedding, shape (D,)."""

    lineage: Tuple[str, ...]
    """Full lineage from domain to this rank, e.g. ('d__Bacteria', 'p__Proteobacteria', ...)."""

    observation_count: int = 1
    """Number of sequences aggregated into this prototype."""


# ── Reference database ───────────────────────────────────────────────────────

class ReferenceDB:
    """
    Collection of prototype embeddings with taxonomy.

    Stores prototypes organized by rank and supports:
    - Fast tensor-batched lookup (all prototypes at a given rank)
    - Incremental updates via Karcher mean aggregation
    - Serialization / deserialization
    - Construction from lineage + embedding pairs
    """

    def __init__(self, kappa: float = KAPPA_DEFAULT, embedding_dim: int = 0):
        self.kappa = kappa
        self.embedding_dim = embedding_dim
        self._prototypes: Dict[str, Prototype] = {}
        # Rank-indexed caches (invalidated on mutation)
        self._rank_cache: Dict[Rank, Optional[Tuple[List[str], Tensor]]] = {
            r: None for r in RANKS
        }

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._prototypes)

    @property
    def ranks_populated(self) -> List[Rank]:
        return sorted({p.rank for p in self._prototypes.values()})

    def __len__(self) -> int:
        return self.size

    def __contains__(self, taxon_id: str) -> bool:
        return taxon_id in self._prototypes

    # ── Core operations ──────────────────────────────────────────────────

    def add_prototype(
        self,
        taxon_id: str,
        rank: Rank,
        embedding: Tensor,
        lineage: Sequence[str],
        observation_count: int = 1,
    ) -> None:
        """Add or update a prototype.  If it already exists, aggregate via Karcher mean."""
        embedding = _clamp_to_ball(embedding.detach(), self.kappa)

        if self.embedding_dim == 0:
            self.embedding_dim = embedding.shape[-1]

        if taxon_id in self._prototypes:
            existing = self._prototypes[taxon_id]
            # Weighted Karcher mean of old + new
            total = existing.observation_count + observation_count
            w_old = existing.observation_count / total
            w_new = observation_count / total
            points = torch.stack([existing.embedding, embedding])
            weights = torch.tensor([w_old, w_new], device=embedding.device)
            mean, _ = karcher_mean(points, kappa=self.kappa, weights=weights)
            existing.embedding = mean
            existing.observation_count = total
        else:
            self._prototypes[taxon_id] = Prototype(
                taxon_id=taxon_id,
                rank=rank,
                embedding=embedding,
                lineage=tuple(lineage),
                observation_count=observation_count,
            )

        # Invalidate cache for this rank
        self._rank_cache[rank] = None

    def get_prototype(self, taxon_id: str) -> Optional[Prototype]:
        return self._prototypes.get(taxon_id)

    def get_prototypes_at_rank(self, rank: Rank) -> Tuple[List[str], Tensor]:
        """
        Return all prototypes at a given rank as (taxon_ids, embeddings_tensor).

        Uses caching for repeated queries.  embeddings_tensor has shape (N, D).
        """
        if self._rank_cache[rank] is not None:
            return self._rank_cache[rank]

        protos = [p for p in self._prototypes.values() if p.rank == rank]
        if not protos:
            self._rank_cache[rank] = ([], torch.zeros(0, max(self.embedding_dim, 1)))
            return self._rank_cache[rank]

        ids = [p.taxon_id for p in protos]
        embeddings = torch.stack([p.embedding for p in protos])
        self._rank_cache[rank] = (ids, embeddings)
        return ids, embeddings

    def get_all_prototypes(self) -> Tuple[List[str], Tensor, List[Rank]]:
        """Return all prototypes across all ranks as (ids, embeddings, ranks)."""
        if not self._prototypes:
            return [], torch.zeros(0, max(self.embedding_dim, 1)), []

        protos = list(self._prototypes.values())
        ids = [p.taxon_id for p in protos]
        embeddings = torch.stack([p.embedding for p in protos])
        ranks = [p.rank for p in protos]
        return ids, embeddings, ranks

    def get_lineage(self, taxon_id: str) -> Optional[Tuple[str, ...]]:
        proto = self._prototypes.get(taxon_id)
        return proto.lineage if proto else None

    def get_children(self, parent_id: str, child_rank: Rank) -> List[str]:
        """Get all prototypes at child_rank whose lineage passes through parent_id."""
        children = []
        for proto in self._prototypes.values():
            if proto.rank == child_rank and parent_id in proto.lineage:
                children.append(proto.taxon_id)
        return children

    # ── Bulk construction ────────────────────────────────────────────────

    @classmethod
    def from_lineages(
        cls,
        lineages: List[Tuple[str, ...]],
        embeddings: Tensor,
        kappa: float = KAPPA_DEFAULT,
        leaf_rank: Rank = Rank.SPECIES,
    ) -> "ReferenceDB":
        """
        Build a reference database from lineage strings and embeddings.

        Args:
            lineages: List of tuples, each (d__X, p__Y, ..., s__Z).
            embeddings: (N, D) tensor of Poincaré ball embeddings.
            kappa: curvature.
            leaf_rank: rank of the provided embeddings.

        Returns:
            Populated ReferenceDB with prototypes at all ranks
            (higher ranks aggregated via Karcher mean).
        """
        db = cls(kappa=kappa, embedding_dim=embeddings.shape[-1])

        # Register leaf-level prototypes
        for lineage, emb in zip(lineages, embeddings):
            taxon_id = lineage[leaf_rank.value] if len(lineage) > leaf_rank.value else lineage[-1]
            db.add_prototype(
                taxon_id=taxon_id,
                rank=leaf_rank,
                embedding=emb,
                lineage=lineage,
            )

        # Aggregate higher ranks
        for rank in reversed(RANKS):
            if rank >= leaf_rank:
                continue
            # Group leaf embeddings by this rank's taxon
            groups: Dict[str, List[Tuple[Tensor, Tuple[str, ...]]]] = {}
            for lineage, emb in zip(lineages, embeddings):
                if len(lineage) <= rank.value:
                    continue
                tid = lineage[rank.value]
                if tid not in groups:
                    groups[tid] = []
                groups[tid].append((emb, lineage))

            for tid, items in groups.items():
                points = torch.stack([e for e, _ in items])
                mean, _ = karcher_mean(points, kappa=kappa)
                representative_lineage = items[0][1][: rank.value + 1]
                db.add_prototype(
                    taxon_id=tid,
                    rank=rank,
                    embedding=mean,
                    lineage=representative_lineage,
                    observation_count=len(items),
                )

        return db

    # ── Serialization ────────────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> None:
        """Save reference database to disk."""
        path = Path(path)
        data = {
            "kappa": self.kappa,
            "embedding_dim": self.embedding_dim,
            "prototypes": {
                tid: {
                    "rank": p.rank.value,
                    "embedding": p.embedding.cpu().tolist(),
                    "lineage": list(p.lineage),
                    "observation_count": p.observation_count,
                }
                for tid, p in self._prototypes.items()
            },
        }
        if path.suffix == ".json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            with open(path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ReferenceDB":
        """Load reference database from disk."""
        path = Path(path)
        if path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
        else:
            with open(path, "rb") as f:
                data = pickle.load(f)

        db = cls(kappa=data["kappa"], embedding_dim=data["embedding_dim"])
        for tid, pdata in data["prototypes"].items():
            db.add_prototype(
                taxon_id=tid,
                rank=Rank(pdata["rank"]),
                embedding=torch.tensor(pdata["embedding"]),
                lineage=pdata["lineage"],
                observation_count=pdata["observation_count"],
            )
        return db

    # ── Summary ──────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, int]:
        """Count of prototypes per rank."""
        counts: Dict[str, int] = {}
        for rank in RANKS:
            ids, _ = self.get_prototypes_at_rank(rank)
            if ids:
                counts[rank.name.lower()] = len(ids)
        return counts
