"""
atlas-tree: Phylogenetic tree construction from BiosphereAtlas coordinates.
===========================================================================

Automatic tree construction from Poincare ball embeddings.  Edges follow
geodesics, quartet-consistent by construction.  Replaces alignment-based
tools like RAxML.

All operations use the Poincare ball model with curvature kappa = 1.247
(Fenn & Fenn 2025), matching the BiosphereAtlas training geometry.

Quick start::

    from biosphere_atlas.tree import build_tree
    import torch

    embeddings = torch.load("coords.pt")
    taxa = ["E_coli", "S_enterica", "L_acidophilus", "M_thermo"]

    tree, report = build_tree(embeddings, taxa)
    print(tree.summary())
    if report:
        print(report.summary())

Copyright (c) 2025 Sentry Bio, Inc.
"""

from .build import build_tree, estimate_tree_quality
from .export import to_newick, to_json, to_svg, write_newick, write_json, write_svg
from .nj import neighbor_joining, compute_distance_matrix, root_at_midpoint
from .quartet import (
    check_quartet_consistency,
    four_point_delta,
    ConsistencyReport,
    QuartetResult,
    Topology,
)
from .tree_struct import PhyloTree, TreeNode, TreeEdge

__all__ = [
    # High-level API
    "build_tree",
    "estimate_tree_quality",
    # Tree construction
    "neighbor_joining",
    "compute_distance_matrix",
    "root_at_midpoint",
    # Tree structures
    "PhyloTree",
    "TreeNode",
    "TreeEdge",
    # Quartet validation
    "check_quartet_consistency",
    "four_point_delta",
    "ConsistencyReport",
    "QuartetResult",
    "Topology",
    # Export
    "to_newick",
    "to_json",
    "to_svg",
    "write_newick",
    "write_json",
    "write_svg",
]
