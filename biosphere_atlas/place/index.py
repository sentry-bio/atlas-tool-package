"""
Spatial index for O(log n) nearest-neighbour queries in hyperbolic space.
=========================================================================

Provides:
- BruteForceIndex : exact NN via pairwise geodesic distance (small DBs)
- VPTree          : vantage-point tree using geodesic distance (large DBs)
- PlacementIndex  : auto-selects strategy based on database size

The VP-tree is the natural data structure for metric spaces —
the only requirement is a distance function, which we supply via
Poincaré geodesic distance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from biosphere_atlas.core.hyperbolic import KAPPA_DEFAULT, poincare_distance


# ── Query result ─────────────────────────────────────────────────────────────

@dataclass
class NNResult:
    """Result of a nearest-neighbour query."""

    indices: List[int]
    """Indices into the reference set, ordered by ascending distance."""

    distances: List[float]
    """Corresponding geodesic distances."""

    @property
    def nearest_index(self) -> int:
        return self.indices[0]

    @property
    def nearest_distance(self) -> float:
        return self.distances[0]

    @property
    def margin(self) -> float:
        """Distance gap between 1st and 2nd nearest (∞ if only one result)."""
        if len(self.distances) < 2:
            return float("inf")
        return self.distances[1] - self.distances[0]


# ── Brute-force index ────────────────────────────────────────────────────────

class BruteForceIndex:
    """
    Exact nearest-neighbour search via pairwise geodesic distance.

    Suitable for reference sets up to ~100k prototypes.
    """

    def __init__(self, embeddings: Tensor, kappa: float = KAPPA_DEFAULT):
        self.embeddings = embeddings  # (N, D)
        self.kappa = kappa
        self._size = embeddings.size(0)

    @property
    def size(self) -> int:
        return self._size

    def query(self, point: Tensor, k: int = 5) -> NNResult:
        """
        Find k nearest neighbours to a single query point.

        Args:
            point: (D,) query embedding.
            k: number of neighbours.

        Returns:
            NNResult with top-k indices and distances.
        """
        k = min(k, self._size)
        # Broadcast: (N, D) vs (1, D) → (N,)
        dists = poincare_distance(
            self.embeddings, point.unsqueeze(0).expand_as(self.embeddings),
            kappa=self.kappa,
        )
        topk_dists, topk_idx = torch.topk(dists, k, largest=False)
        return NNResult(
            indices=topk_idx.tolist(),
            distances=topk_dists.tolist(),
        )

    def query_batch(self, points: Tensor, k: int = 5) -> List[NNResult]:
        """
        Batch nearest-neighbour query.

        Args:
            points: (B, D) query embeddings.
            k: number of neighbours per query.

        Returns:
            List of NNResult, one per query.
        """
        k = min(k, self._size)
        B = points.size(0)
        N = self._size

        # (B, N) distance matrix
        q_exp = points.unsqueeze(1).expand(B, N, -1)
        r_exp = self.embeddings.unsqueeze(0).expand(B, N, -1)
        dist_matrix = poincare_distance(q_exp, r_exp, kappa=self.kappa)

        topk_dists, topk_idx = torch.topk(dist_matrix, k, dim=1, largest=False)

        results = []
        for i in range(B):
            results.append(NNResult(
                indices=topk_idx[i].tolist(),
                distances=topk_dists[i].tolist(),
            ))
        return results


# ── Vantage-point tree ───────────────────────────────────────────────────────

@dataclass
class _VPNode:
    """Internal node of a vantage-point tree."""

    vantage_idx: int
    """Index of the vantage point in the reference set."""

    threshold: float
    """Median distance — left subtree ≤ threshold, right subtree > threshold."""

    left: Optional["_VPNode"] = None
    right: Optional["_VPNode"] = None


class VPTree:
    """
    Vantage-point tree for O(log n) nearest-neighbour queries in metric spaces.

    Build cost: O(n log n).  Query cost: O(log n) average case.

    Uses Poincaré geodesic distance as the metric.
    """

    VP_TREE_THRESHOLD = 256
    """Below this many points, fall back to brute-force within the tree."""

    def __init__(self, embeddings: Tensor, kappa: float = KAPPA_DEFAULT):
        self.embeddings = embeddings
        self.kappa = kappa
        self._size = embeddings.size(0)
        indices = list(range(self._size))
        self._root = self._build(indices)

    @property
    def size(self) -> int:
        return self._size

    def _dist(self, i: int, j: int) -> float:
        return poincare_distance(
            self.embeddings[i].unsqueeze(0),
            self.embeddings[j].unsqueeze(0),
            kappa=self.kappa,
        ).item()

    def _dist_to_point(self, idx: int, point: Tensor) -> float:
        return poincare_distance(
            self.embeddings[idx].unsqueeze(0),
            point.unsqueeze(0),
            kappa=self.kappa,
        ).item()

    def _build(self, indices: List[int]) -> Optional[_VPNode]:
        if not indices:
            return None

        if len(indices) == 1:
            return _VPNode(vantage_idx=indices[0], threshold=0.0)

        # Choose vantage point: pick the one with highest distance variance
        # (approximation: use first point for speed)
        vp_idx = indices[0]
        rest = indices[1:]

        # Compute distances from vantage to all others
        dists = [(idx, self._dist(vp_idx, idx)) for idx in rest]
        dists.sort(key=lambda x: x[1])

        median_pos = len(dists) // 2
        threshold = dists[median_pos][1] if dists else 0.0

        left_indices = [idx for idx, d in dists[:median_pos + 1]]
        right_indices = [idx for idx, d in dists[median_pos + 1:]]

        node = _VPNode(
            vantage_idx=vp_idx,
            threshold=threshold,
            left=self._build(left_indices) if left_indices else None,
            right=self._build(right_indices) if right_indices else None,
        )
        return node

    def query(self, point: Tensor, k: int = 5) -> NNResult:
        """
        Find k nearest neighbours using the VP-tree.

        Args:
            point: (D,) query embedding.
            k: number of neighbours.

        Returns:
            NNResult with top-k indices and distances.
        """
        k = min(k, self._size)
        # Max-heap of (neg_distance, index) — we keep the k closest
        best: List[Tuple[float, int]] = []
        self._search(self._root, point, k, best)

        # Sort by ascending distance
        best.sort(key=lambda x: -x[0])  # neg_dist → ascending
        return NNResult(
            indices=[idx for _, idx in best],
            distances=[-d for d, _ in best],
        )

    def _search(
        self,
        node: Optional[_VPNode],
        point: Tensor,
        k: int,
        best: List[Tuple[float, int]],
    ) -> None:
        if node is None:
            return

        d = self._dist_to_point(node.vantage_idx, point)
        tau = -best[0][0] if len(best) >= k else float("inf")

        # Consider this vantage point
        if d < tau or len(best) < k:
            if len(best) < k:
                best.append((-d, node.vantage_idx))
                best.sort()
            elif d < tau:
                best[0] = (-d, node.vantage_idx)
                best.sort()

        tau = -best[0][0] if len(best) >= k else float("inf")

        # Prune: decide which subtrees to search
        if d <= node.threshold:
            # Point is inside the threshold — search left first
            if d - tau <= node.threshold:
                self._search(node.left, point, k, best)
            tau = -best[0][0] if len(best) >= k else float("inf")
            if d + tau > node.threshold:
                self._search(node.right, point, k, best)
        else:
            # Point is outside the threshold — search right first
            if d + tau > node.threshold:
                self._search(node.right, point, k, best)
            tau = -best[0][0] if len(best) >= k else float("inf")
            if d - tau <= node.threshold:
                self._search(node.left, point, k, best)

    def query_batch(self, points: Tensor, k: int = 5) -> List[NNResult]:
        """Batch query — runs each point through the tree."""
        return [self.query(points[i], k=k) for i in range(points.size(0))]


# ── Auto-scaling index ───────────────────────────────────────────────────────

class PlacementIndex:
    """
    Auto-scaling spatial index.

    Uses brute-force for small reference sets (< threshold) and
    builds a VP-tree for larger ones.
    """

    AUTO_TREE_THRESHOLD = 10_000
    """Build VP-tree when reference set exceeds this size."""

    def __init__(
        self,
        embeddings: Tensor,
        kappa: float = KAPPA_DEFAULT,
        force_tree: bool = False,
    ):
        self.kappa = kappa
        self._size = embeddings.size(0)

        if force_tree or self._size > self.AUTO_TREE_THRESHOLD:
            self._impl = VPTree(embeddings, kappa=kappa)
            self.strategy = "vp_tree"
        else:
            self._impl = BruteForceIndex(embeddings, kappa=kappa)
            self.strategy = "brute_force"

    @property
    def size(self) -> int:
        return self._size

    def query(self, point: Tensor, k: int = 5) -> NNResult:
        return self._impl.query(point, k=k)

    def query_batch(self, points: Tensor, k: int = 5) -> List[NNResult]:
        return self._impl.query_batch(points, k=k)
