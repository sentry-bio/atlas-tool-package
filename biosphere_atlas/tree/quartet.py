"""
Quartet consistency checking.
==============================

A quartet is a set of four taxa {a, b, c, d} with three possible
unrooted topologies:

    1. ab|cd  (a,b together vs c,d together)
    2. ac|bd  (a,c together vs b,d together)
    3. ad|bc  (a,d together vs b,c together)

In a metric space, the four-point condition determines the correct
topology: the two largest of the three pairwise sums
{d(a,b)+d(c,d), d(a,c)+d(b,d), d(a,d)+d(b,c)} must be equal, and
the smallest identifies the topology.

For embeddings in hyperbolic space (delta-hyperbolic with delta -> 0
for trees), the four-point condition is automatically satisfied when
the embedding faithfully represents phylogenetic relationships.  This
module validates that the NJ tree agrees with the coordinate-implied
topology for all (or sampled) quartets.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from biosphere_atlas.core.hyperbolic import KAPPA_DEFAULT, poincare_distance
from .tree_struct import PhyloTree


class Topology(IntEnum):
    """Three possible unrooted topologies for a quartet {a, b, c, d}."""
    AB_CD = 0  # (a,b)|(c,d)
    AC_BD = 1  # (a,c)|(b,d)
    AD_BC = 2  # (a,d)|(b,c)


@dataclass
class QuartetResult:
    """Result of a single quartet consistency check."""
    taxa: Tuple[str, str, str, str]
    coordinate_topology: Topology
    tree_topology: Optional[Topology]
    consistent: bool
    delta: float
    """Four-point delta: how close the four-point condition is to being
    perfectly tree-like.  delta=0 means perfect tree metric."""


@dataclass
class ConsistencyReport:
    """Summary of quartet consistency checking."""
    total_quartets: int
    consistent: int
    inconsistent: int
    unresolved: int
    mean_delta: float
    max_delta: float
    consistency_fraction: float

    def summary(self) -> str:
        return (
            f"QuartetConsistency(total={self.total_quartets}, "
            f"consistent={self.consistent}/{self.total_quartets} "
            f"({self.consistency_fraction:.1%}), "
            f"mean_delta={self.mean_delta:.6f}, max_delta={self.max_delta:.6f})"
        )


def quartet_topology_from_distances(
    d_ab: float, d_ac: float, d_ad: float,
    d_bc: float, d_bd: float, d_cd: float,
) -> Tuple[Topology, float]:
    """Determine quartet topology from the six pairwise distances.

    Uses the four-point condition: compute the three sums
        S1 = d(a,b) + d(c,d)    -> topology ab|cd
        S2 = d(a,c) + d(b,d)    -> topology ac|bd
        S3 = d(a,d) + d(b,c)    -> topology ad|bc

    The topology with the SMALLEST sum is the correct one (the other
    two sums should be equal for a perfect tree metric).

    Returns:
        (topology, delta) where delta measures deviation from tree metric.
    """
    s1 = d_ab + d_cd  # ab|cd
    s2 = d_ac + d_bd  # ac|bd
    s3 = d_ad + d_bc  # ad|bc

    sums = [(s1, Topology.AB_CD), (s2, Topology.AC_BD), (s3, Topology.AD_BC)]
    sums.sort(key=lambda x: x[0])

    # Topology is the one with smallest sum
    topology = sums[0][1]

    # Delta-hyperbolicity: for a perfect tree metric, the two largest
    # sums are equal.  delta = (largest - second largest) / 2
    delta = (sums[2][0] - sums[1][0]) / 2.0

    return topology, abs(delta)


def coordinate_quartet_topology(
    emb_a: Tensor, emb_b: Tensor, emb_c: Tensor, emb_d: Tensor,
    kappa: float = KAPPA_DEFAULT,
) -> Tuple[Topology, float]:
    """Determine quartet topology directly from Poincare ball coordinates."""
    d_ab = poincare_distance(emb_a, emb_b, kappa).item()
    d_ac = poincare_distance(emb_a, emb_c, kappa).item()
    d_ad = poincare_distance(emb_a, emb_d, kappa).item()
    d_bc = poincare_distance(emb_b, emb_c, kappa).item()
    d_bd = poincare_distance(emb_b, emb_d, kappa).item()
    d_cd = poincare_distance(emb_c, emb_d, kappa).item()
    return quartet_topology_from_distances(d_ab, d_ac, d_ad, d_bc, d_bd, d_cd)


def tree_quartet_topology(
    tree: PhyloTree,
    nid_a: int, nid_b: int, nid_c: int, nid_d: int,
) -> Optional[Tuple[Topology, float]]:
    """Determine quartet topology from patristic (tree-path) distances.

    Returns None if any path is disconnected (inf distance).
    """
    d_ab = tree.path_distance(nid_a, nid_b)
    d_ac = tree.path_distance(nid_a, nid_c)
    d_ad = tree.path_distance(nid_a, nid_d)
    d_bc = tree.path_distance(nid_b, nid_c)
    d_bd = tree.path_distance(nid_b, nid_d)
    d_cd = tree.path_distance(nid_c, nid_d)

    if any(d == float("inf") for d in [d_ab, d_ac, d_ad, d_bc, d_bd, d_cd]):
        return None

    return quartet_topology_from_distances(d_ab, d_ac, d_ad, d_bc, d_bd, d_cd)


def check_quartet_consistency(
    tree: PhyloTree,
    embeddings: Dict[int, Tensor],
    kappa: float = KAPPA_DEFAULT,
    max_quartets: Optional[int] = None,
    seed: Optional[int] = None,
) -> ConsistencyReport:
    """Check whether the tree's quartet topologies agree with coordinate-implied topologies.

    For each quartet of leaves, the four-point condition on geodesic
    distances determines a topology.  We compare this to the topology
    implied by the tree's patristic distances.

    Args:
        tree: the phylogenetic tree to check.
        embeddings: map from leaf node_id to Poincare ball embedding.
        kappa: curvature.
        max_quartets: if set, randomly sample this many quartets
            instead of checking all.
        seed: random seed for sampling.

    Returns:
        A ``ConsistencyReport`` with detailed statistics.
    """
    leaf_ids = sorted(embeddings.keys())
    n = len(leaf_ids)

    if n < 4:
        return ConsistencyReport(
            total_quartets=0, consistent=0, inconsistent=0,
            unresolved=0, mean_delta=0.0, max_delta=0.0,
            consistency_fraction=1.0,
        )

    # Generate all or sampled quartets
    all_quartets = list(itertools.combinations(leaf_ids, 4))
    if max_quartets is not None and len(all_quartets) > max_quartets:
        rng = random.Random(seed)
        quartets = rng.sample(all_quartets, max_quartets)
    else:
        quartets = all_quartets

    consistent = 0
    inconsistent = 0
    unresolved = 0
    deltas: List[float] = []

    for a, b, c, d in quartets:
        # Coordinate-implied topology
        coord_topo, coord_delta = coordinate_quartet_topology(
            embeddings[a], embeddings[b], embeddings[c], embeddings[d], kappa
        )

        # Tree-implied topology
        tree_result = tree_quartet_topology(tree, a, b, c, d)
        if tree_result is None:
            unresolved += 1
            continue

        tree_topo, tree_delta = tree_result
        deltas.append(coord_delta)

        if coord_topo == tree_topo:
            consistent += 1
        else:
            inconsistent += 1

    total = consistent + inconsistent + unresolved
    mean_d = sum(deltas) / max(len(deltas), 1)
    max_d = max(deltas) if deltas else 0.0
    frac = consistent / max(consistent + inconsistent, 1)

    return ConsistencyReport(
        total_quartets=total,
        consistent=consistent,
        inconsistent=inconsistent,
        unresolved=unresolved,
        mean_delta=mean_d,
        max_delta=max_d,
        consistency_fraction=frac,
    )


def four_point_delta(
    embeddings: Tensor, kappa: float = KAPPA_DEFAULT,
    max_quartets: int = 1000, seed: int = 42,
) -> float:
    """Estimate the Gromov delta-hyperbolicity of the embedding set.

    For a perfect tree metric, delta = 0.  Small delta indicates
    tree-like structure.  This is a global measure of how well the
    embedding space supports phylogenetic tree construction.

    Args:
        embeddings: (N, D) coordinates.
        kappa: curvature.
        max_quartets: number of quartets to sample.
        seed: random seed.

    Returns:
        Estimated delta (mean over sampled quartets).
    """
    n = embeddings.size(0)
    if n < 4:
        return 0.0

    rng = random.Random(seed)
    indices = list(range(n))
    deltas: List[float] = []

    for _ in range(min(max_quartets, n * (n - 1) * (n - 2) * (n - 3) // 24)):
        a, b, c, d = rng.sample(indices, 4)
        _, delta = coordinate_quartet_topology(
            embeddings[a], embeddings[b], embeddings[c], embeddings[d], kappa
        )
        deltas.append(delta)

    return sum(deltas) / max(len(deltas), 1)
