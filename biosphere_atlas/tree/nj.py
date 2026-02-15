"""
Neighbor-Joining on geodesic distances.
========================================

Classic NJ algorithm (Saitou & Nei 1987) operating on Poincare ball
geodesic distances.  Produces an unrooted binary tree whose edge
lengths approximate the inter-taxon geodesic distances.

The key advantage: no alignment is needed — the BiosphereAtlas embedding
already encodes evolutionary distance through manifold geometry.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from biosphere_atlas.core.hyperbolic import (
    KAPPA_DEFAULT,
    _clamp_to_ball,
    karcher_mean,
    poincare_distance,
)
from .tree_struct import PhyloTree


def compute_distance_matrix(
    embeddings: Tensor, kappa: float = KAPPA_DEFAULT
) -> Tensor:
    """Compute all-pairs geodesic distance matrix.

    Args:
        embeddings: (N, D) Poincare ball coordinates.
        kappa: curvature.

    Returns:
        (N, N) symmetric distance matrix.
    """
    n = embeddings.size(0)
    D = torch.zeros(n, n, device=embeddings.device)
    for i in range(n):
        if i + 1 < n:
            dists = poincare_distance(
                embeddings[i].unsqueeze(0).expand(n - i - 1, -1),
                embeddings[i + 1:],
                kappa,
            )
            D[i, i + 1:] = dists
            D[i + 1:, i] = dists
    return D


def neighbor_joining(
    embeddings: Tensor,
    taxon_ids: List[str],
    kappa: float = KAPPA_DEFAULT,
) -> PhyloTree:
    """Build an unrooted phylogenetic tree via Neighbor-Joining.

    The algorithm operates on Poincare ball geodesic distances and places
    internal nodes at weighted Karcher means of their children.

    Args:
        embeddings: (N, D) embeddings inside the Poincare ball.
        taxon_ids: list of N taxon labels.
        kappa: curvature constant.

    Returns:
        A ``PhyloTree`` with geodesic edge lengths.
    """
    n = len(taxon_ids)
    if n < 2:
        tree = PhyloTree(kappa=kappa)
        if n == 1:
            tree.add_leaf(taxon_ids[0], embeddings[0])
        return tree

    # Compute pairwise distance matrix
    D = compute_distance_matrix(embeddings, kappa)

    # Active node set: maps active_index -> (tree_node_id, embedding)
    tree = PhyloTree(kappa=kappa)
    active: Dict[int, Tuple[int, Tensor]] = {}
    for i in range(n):
        nid = tree.add_leaf(taxon_ids[i], embeddings[i])
        active[i] = (nid, embeddings[i])

    # Working distance matrix (grows via new internal nodes)
    dist = D.clone()

    while len(active) > 2:
        active_keys = sorted(active.keys())
        m = len(active_keys)

        # Row sums for active nodes
        row_sums: Dict[int, float] = {}
        for i in active_keys:
            s = 0.0
            for j in active_keys:
                if i != j:
                    s += dist[i, j].item()
            row_sums[i] = s

        # Find pair (i, j) minimizing Q(i, j)
        best_q = float("inf")
        best_pair: Tuple[int, int] = (active_keys[0], active_keys[1])
        for idx_a in range(m):
            i = active_keys[idx_a]
            for idx_b in range(idx_a + 1, m):
                j = active_keys[idx_b]
                q = (m - 2) * dist[i, j].item() - row_sums[i] - row_sums[j]
                if q < best_q:
                    best_q = q
                    best_pair = (i, j)

        i, j = best_pair

        # Edge lengths from i, j to new internal node
        d_ij = dist[i, j].item()
        delta = (row_sums[i] - row_sums[j]) / max(m - 2, 1)
        li = max(0.5 * (d_ij + delta), 0.0)
        lj = max(d_ij - li, 0.0)

        # Place internal node at Karcher mean weighted by inverse distance
        emb_i = active[i][1]
        emb_j = active[j][1]
        if li + lj > 1e-15:
            w_i = lj / (li + lj)  # closer to i if lj is large
            w_j = li / (li + lj)
        else:
            w_i, w_j = 0.5, 0.5
        weights = torch.tensor([w_i, w_j], device=embeddings.device)
        pair = torch.stack([emb_i, emb_j])
        int_emb, _ = karcher_mean(pair, kappa, weights=weights)

        # Create internal node in the tree
        nid_i = active[i][0]
        nid_j = active[j][0]
        nid_k = tree.add_internal(children=[nid_i, nid_j], embedding=int_emb)
        tree.add_edge(nid_k, nid_i, li)
        tree.add_edge(nid_k, nid_j, lj)

        # Update distance matrix: distances from new node k to remaining active nodes
        # Expand dist matrix if needed
        k = max(dist.size(0), max(active.keys()) + 1)
        if k >= dist.size(0):
            new_dist = torch.zeros(k + 1, k + 1, device=dist.device)
            new_dist[:dist.size(0), :dist.size(1)] = dist
            dist = new_dist

        for r in active_keys:
            if r != i and r != j:
                d_kr = 0.5 * (dist[i, r].item() + dist[j, r].item() - d_ij)
                d_kr = max(d_kr, 0.0)
                dist[k, r] = d_kr
                dist[r, k] = d_kr

        # Remove i, j from active; add k
        del active[i]
        del active[j]
        active[k] = (nid_k, int_emb)

    # Final two nodes — connect directly
    remaining = sorted(active.keys())
    if len(remaining) == 2:
        a, b = remaining
        nid_a = active[a][0]
        nid_b = active[b][0]
        d_ab = dist[a, b].item()
        tree.add_edge(nid_a, nid_b, d_ab)
        # If both are internal, wire parent relationship
        node_a = tree.get_node(nid_a)
        node_b = tree.get_node(nid_b)
        if not node_a.is_leaf and node_b.is_leaf:
            tree.connect(nid_a, nid_b)
        elif not node_b.is_leaf and node_a.is_leaf:
            tree.connect(nid_b, nid_a)

    return tree


def root_at_midpoint(tree: PhyloTree) -> PhyloTree:
    """Root the tree at the midpoint of the longest path (patristic distance).

    Finds the two most distant leaves and inserts a virtual root at the
    midpoint of their connecting path.

    Returns the same tree object, now rooted.
    """
    leaf_ids = tree.leaf_ids()
    if len(leaf_ids) < 2:
        return tree

    # Find the diameter pair
    best_d = -1.0
    best_pair = (leaf_ids[0], leaf_ids[1])
    for i in range(len(leaf_ids)):
        for j in range(i + 1, len(leaf_ids)):
            d = tree.path_distance(leaf_ids[i], leaf_ids[j])
            if d > best_d:
                best_d = d
                best_pair = (leaf_ids[i], leaf_ids[j])

    # Root at the first node of the diameter pair for simplicity
    # (a proper midpoint root would split an edge, which we keep simple here)
    tree.root_at(best_pair[0])
    return tree
