"""
High-level tree construction API.
==================================

Convenience functions that combine neighbor-joining, quartet validation,
and optional export in a single call.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from biosphere_atlas.core.hyperbolic import KAPPA_DEFAULT
from .nj import compute_distance_matrix, neighbor_joining, root_at_midpoint
from .quartet import (
    ConsistencyReport,
    check_quartet_consistency,
    four_point_delta,
)
from .tree_struct import PhyloTree


def build_tree(
    embeddings: Tensor,
    taxon_ids: List[str],
    kappa: float = KAPPA_DEFAULT,
    midpoint_root: bool = False,
    validate_quartets: bool = True,
    max_quartets: int = 5000,
    seed: int = 42,
) -> Tuple[PhyloTree, Optional[ConsistencyReport]]:
    """Build a phylogenetic tree from Poincare ball embeddings.

    This is the main entry point for tree construction.  It runs
    neighbor-joining on geodesic distances and optionally validates
    quartet consistency.

    Args:
        embeddings: (N, D) Poincare ball coordinates.
        taxon_ids: list of N taxon labels.
        kappa: curvature constant.
        midpoint_root: if True, root the tree at the midpoint of the
            longest patristic path.
        validate_quartets: if True, check quartet consistency.
        max_quartets: maximum quartets to sample for validation.
        seed: random seed for quartet sampling.

    Returns:
        (tree, report) where report is None if validation was skipped.
    """
    tree = neighbor_joining(embeddings, taxon_ids, kappa=kappa)

    if midpoint_root:
        root_at_midpoint(tree)

    report = None
    if validate_quartets and len(taxon_ids) >= 4:
        # Build embedding map for leaf nodes
        emb_map: Dict[int, Tensor] = {}
        for leaf in tree.leaves():
            if leaf.embedding is not None:
                emb_map[leaf.node_id] = leaf.embedding

        report = check_quartet_consistency(
            tree, emb_map, kappa=kappa,
            max_quartets=max_quartets, seed=seed,
        )

    return tree, report


def estimate_tree_quality(
    embeddings: Tensor,
    kappa: float = KAPPA_DEFAULT,
    max_quartets: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Estimate how well the embedding space supports tree construction.

    Returns metrics indicating tree-like structure in the coordinates:

    - ``delta``: Gromov delta-hyperbolicity (0 = perfect tree metric).
    - ``n_points``: number of embedding points.
    - ``mean_distance``: mean pairwise geodesic distance.
    - ``std_distance``: std of pairwise geodesic distances.

    Args:
        embeddings: (N, D) Poincare ball coordinates.
        kappa: curvature.
        max_quartets: quartets to sample for delta estimation.
        seed: random seed.

    Returns:
        Dictionary of quality metrics.
    """
    n = embeddings.size(0)
    delta = four_point_delta(embeddings, kappa, max_quartets, seed)

    # Compute distance statistics from a sample
    D = compute_distance_matrix(embeddings, kappa)
    # Upper triangle (excluding diagonal)
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    dists = D[mask]

    return {
        "delta": delta,
        "n_points": float(n),
        "mean_distance": dists.mean().item() if dists.numel() > 0 else 0.0,
        "std_distance": dists.std().item() if dists.numel() > 1 else 0.0,
    }
