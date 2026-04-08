"""
Core phylogenetic placement logic.
===================================

Given a query embedding and a reference database, finds the k nearest
prototypes, computes margins, and produces ranked placement candidates.

This module handles geometry-only placement.  Conformal calibration
(three-zone decisions, coverage guarantees) is in calibrator.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from biosphere_atlas.core.hyperbolic import KAPPA_DEFAULT, dist_from_origin, poincare_distance
from biosphere_atlas.place.index import NNResult, PlacementIndex
from biosphere_atlas.place.reference import Prototype, Rank, ReferenceDB


# ── Placement result ─────────────────────────────────────────────────────────

@dataclass
class PlacementCandidate:
    """A single candidate placement."""

    taxon_id: str
    rank: Rank
    distance: float
    """Geodesic distance to this prototype."""
    lineage: Tuple[str, ...]

    @property
    def rank_name(self) -> str:
        return self.rank.name.lower()


@dataclass
class PlacementResult:
    """Full placement result for a single query sequence."""

    sequence_id: str
    candidates: List[PlacementCandidate]
    """Top-k candidates ordered by ascending distance."""

    best_distance: float
    """Distance to nearest prototype."""

    margin: float
    """Gap between 1st and 2nd nearest prototype."""

    atlas_r: float
    """Hyperbolic radial coordinate (distance from LUCA)."""

    atlas_theta: float
    """Angular coordinate in BiosphereAtlas (r, θ) system."""

    embedding: Optional[Tensor] = field(default=None, repr=False)
    """Query embedding (optional, for downstream use)."""

    # Populated by calibrator
    zone: Optional[str] = None
    """Conformal zone: 'accept', 'escalate', or 'fallback'."""

    confidence: Optional[float] = None
    """Calibrated placement confidence ∈ [0, 1]."""

    prediction_set_size: Optional[int] = None
    """Size of the conformal prediction set."""

    @property
    def best_placement(self) -> Optional[PlacementCandidate]:
        return self.candidates[0] if self.candidates else None

    @property
    def lineage_string(self) -> str:
        if not self.candidates:
            return ""
        return ";".join(self.candidates[0].lineage)

    def to_dict(self) -> Dict:
        """Convert to flat dictionary for output."""
        best = self.best_placement
        return {
            "sequence_id": self.sequence_id,
            "classification": best.taxon_id if best else "",
            "rank": best.rank_name if best else "",
            "lineage": self.lineage_string,
            "distance": round(self.best_distance, 6),
            "margin": round(self.margin, 6),
            "zone": self.zone or "",
            "confidence": round(self.confidence, 6) if self.confidence is not None else "",
            "prediction_set_size": self.prediction_set_size or "",
            "atlas_r": round(self.atlas_r, 6),
            "atlas_theta": round(self.atlas_theta, 6),
            "n_candidates": len(self.candidates),
        }


# ── Placement engine ─────────────────────────────────────────────────────────

class PlacementEngine:
    """
    Nearest-prototype phylogenetic placement.

    Supports two modes:
    - Flat:         query against all prototypes regardless of rank
    - Hierarchical: descend through ranks, narrowing candidates at each level
    """

    def __init__(
        self,
        reference: ReferenceDB,
        kappa: float = KAPPA_DEFAULT,
        top_k: int = 5,
    ):
        self.reference = reference
        self.kappa = kappa
        self.top_k = top_k

        # Build flat index over all prototypes
        ids, embeddings, ranks = reference.get_all_prototypes()
        self._all_ids = ids
        self._all_ranks = ranks
        self._all_embeddings = embeddings

        if embeddings.size(0) > 0:
            self._flat_index = PlacementIndex(embeddings, kappa=kappa)
        else:
            self._flat_index = None

        # Build per-rank indices for hierarchical placement
        self._rank_indices: Dict[Rank, Tuple[List[str], PlacementIndex]] = {}
        for rank in reference.ranks_populated:
            rids, rembs = reference.get_prototypes_at_rank(rank)
            if rembs.size(0) > 0:
                self._rank_indices[rank] = (rids, PlacementIndex(rembs, kappa=kappa))

    def place(
        self,
        embedding: Tensor,
        sequence_id: str = "",
        mode: str = "flat",
    ) -> PlacementResult:
        """
        Place a single query embedding.

        Args:
            embedding: (D,) Poincaré ball embedding.
            sequence_id: identifier for the query.
            mode: 'flat' or 'hierarchical'.

        Returns:
            PlacementResult with ranked candidates.
        """
        # Keep query on same device as reference index tensors.
        if self._all_embeddings.numel() > 0 and embedding.device != self._all_embeddings.device:
            embedding = embedding.to(self._all_embeddings.device)

        if mode == "hierarchical":
            return self._place_hierarchical(embedding, sequence_id)
        return self._place_flat(embedding, sequence_id)

    def place_batch(
        self,
        embeddings: Tensor,
        sequence_ids: Optional[List[str]] = None,
        mode: str = "flat",
    ) -> List[PlacementResult]:
        """
        Place a batch of query embeddings.

        Args:
            embeddings: (B, D) query embeddings.
            sequence_ids: optional list of identifiers.
            mode: 'flat' or 'hierarchical'.

        Returns:
            List of PlacementResult.
        """
        B = embeddings.size(0)
        if sequence_ids is None:
            sequence_ids = [f"query_{i}" for i in range(B)]

        results = []
        for i in range(B):
            results.append(self.place(embeddings[i], sequence_ids[i], mode=mode))
        return results

    # ── Flat placement ───────────────────────────────────────────────────

    def _place_flat(self, embedding: Tensor, sequence_id: str) -> PlacementResult:
        """Place against all prototypes regardless of rank."""
        if self._flat_index is None or self._flat_index.size == 0:
            return self._empty_result(embedding, sequence_id)

        nn = self._flat_index.query(embedding, k=self.top_k)
        candidates = self._nn_to_candidates(nn, self._all_ids, self._all_ranks)

        r, theta = self._extract_coordinates(embedding)

        return PlacementResult(
            sequence_id=sequence_id,
            candidates=candidates,
            best_distance=nn.nearest_distance,
            margin=nn.margin,
            atlas_r=r,
            atlas_theta=theta,
            embedding=embedding,
        )

    # ── Hierarchical placement ───────────────────────────────────────────

    def _place_hierarchical(self, embedding: Tensor, sequence_id: str) -> PlacementResult:
        """
        Hierarchical top-down placement.

        Start at the coarsest rank, find the best candidate, then
        narrow to its children at the next rank, and repeat.
        """
        sorted_ranks = sorted(self._rank_indices.keys())
        if not sorted_ranks:
            return self._empty_result(embedding, sequence_id)

        best_candidates: List[PlacementCandidate] = []
        parent_id: Optional[str] = None
        last_nn: Optional[NNResult] = None

        for rank in sorted_ranks:
            rids, rank_index = self._rank_indices[rank]

            if parent_id is not None:
                # Filter to children of the best parent
                children = self.reference.get_children(parent_id, rank)
                if not children:
                    break

                # Get embeddings for these children
                child_embs = []
                child_ids = []
                for cid in children:
                    proto = self.reference.get_prototype(cid)
                    if proto is not None:
                        child_ids.append(cid)
                        child_embs.append(proto.embedding)

                if not child_embs:
                    break

                child_tensor = torch.stack(child_embs)
                child_index = PlacementIndex(child_tensor, kappa=self.kappa)
                nn = child_index.query(embedding, k=min(self.top_k, len(child_ids)))

                for i, (idx, dist) in enumerate(zip(nn.indices, nn.distances)):
                    tid = child_ids[idx]
                    proto = self.reference.get_prototype(tid)
                    if proto is not None:
                        best_candidates.append(PlacementCandidate(
                            taxon_id=tid,
                            rank=rank,
                            distance=dist,
                            lineage=proto.lineage,
                        ))
                        if i == 0:
                            parent_id = tid
                last_nn = nn
            else:
                # First rank: query all prototypes at this rank
                nn = rank_index.query(embedding, k=min(self.top_k, rank_index.size))
                for i, (idx, dist) in enumerate(zip(nn.indices, nn.distances)):
                    tid = rids[idx]
                    proto = self.reference.get_prototype(tid)
                    if proto is not None:
                        best_candidates.append(PlacementCandidate(
                            taxon_id=tid,
                            rank=rank,
                            distance=dist,
                            lineage=proto.lineage,
                        ))
                        if i == 0:
                            parent_id = tid
                last_nn = nn

        # Sort by distance across all ranks
        best_candidates.sort(key=lambda c: c.distance)
        best_candidates = best_candidates[: self.top_k]

        r, theta = self._extract_coordinates(embedding)

        best_dist = best_candidates[0].distance if best_candidates else float("inf")
        margin = (
            best_candidates[1].distance - best_candidates[0].distance
            if len(best_candidates) >= 2
            else float("inf")
        )

        return PlacementResult(
            sequence_id=sequence_id,
            candidates=best_candidates,
            best_distance=best_dist,
            margin=margin,
            atlas_r=r,
            atlas_theta=theta,
            embedding=embedding,
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    def _nn_to_candidates(
        self, nn: NNResult, ids: List[str], ranks: List[Rank]
    ) -> List[PlacementCandidate]:
        candidates = []
        for idx, dist in zip(nn.indices, nn.distances):
            tid = ids[idx]
            rank = ranks[idx]
            proto = self.reference.get_prototype(tid)
            lineage = proto.lineage if proto else (tid,)
            candidates.append(PlacementCandidate(
                taxon_id=tid, rank=rank, distance=dist, lineage=lineage,
            ))
        return candidates

    def _extract_coordinates(self, embedding: Tensor) -> Tuple[float, float]:
        """Extract (r, θ) BiosphereAtlas coordinates."""
        import math

        r = dist_from_origin(embedding.unsqueeze(0), kappa=self.kappa).item()

        # Angular coordinate from first two dimensions
        if embedding.shape[-1] >= 2:
            theta = math.atan2(embedding[1].item(), embedding[0].item()) % (2 * math.pi)
        else:
            theta = 0.0

        return r, theta

    def _empty_result(self, embedding: Tensor, sequence_id: str) -> PlacementResult:
        r, theta = self._extract_coordinates(embedding)
        return PlacementResult(
            sequence_id=sequence_id,
            candidates=[],
            best_distance=float("inf"),
            margin=float("inf"),
            atlas_r=r,
            atlas_theta=theta,
            embedding=embedding,
        )
