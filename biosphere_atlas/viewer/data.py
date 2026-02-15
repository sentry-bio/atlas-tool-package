"""
Data loading and ViewerData assembly.
=======================================

Unified interface for loading data from atlas-tree (PhyloTree),
atlas-place (ReferenceDB), or raw embeddings + taxa files.  All
sources are projected to a 2D Poincare disk for visualization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from biosphere_atlas.core.hyperbolic import KAPPA_DEFAULT, dist_from_origin
from .layout import RANK_COLORS, RANK_ORDER, RankBand, compute_rank_bands, estimate_rank_radii
from .projection import ProjectionResult, tangent_pca_projection


@dataclass
class ViewerData:
    """Complete data payload for the Poincare disk viewer.

    This is the bridge between the Python data layer and the HTML/JS
    frontend.  Everything the viewer needs is here.
    """

    # Core geometry
    coords_2d: Tensor
    """(N, 2) 2D Poincare disk coordinates."""

    taxon_ids: List[str]
    """N taxon labels."""

    edges: List[Tuple[int, int, float]]
    """(source_idx, target_idx, geodesic_length) for tree edges."""

    # Taxonomy
    ranks: List[int]
    """(N,) rank levels: 0=domain, ..., 6=species. -1 if unknown."""

    lineages: List[str]
    """(N,) semicolon-separated lineage strings."""

    rank_bands: List[RankBand]
    """Geodesic circles at rank boundaries."""

    # Metadata
    kappa: float = KAPPA_DEFAULT
    n_organisms: int = 0
    title: str = "BiosphereAtlas"
    projection_variance: List[float] = field(default_factory=list)

    def __post_init__(self):
        self.n_organisms = len(self.taxon_ids)

    def to_json_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dictionary for HTML embedding."""
        return {
            "coords": self.coords_2d.tolist(),
            "taxonIds": self.taxon_ids,
            "edges": self.edges,
            "ranks": self.ranks,
            "lineages": self.lineages,
            "rankBands": [
                {
                    "name": b.rank_name,
                    "radius": b.hyperbolic_radius,
                    "euclideanRadius": b.euclidean_radius,
                    "color": b.color,
                    "label": b.label,
                }
                for b in self.rank_bands
            ],
            "kappa": self.kappa,
            "nOrganisms": self.n_organisms,
            "title": self.title,
            "projectionVariance": self.projection_variance,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_json_dict())


# -- Loaders -------------------------------------------------------------------

def from_tree(
    source: Union[str, Path, Dict],
    kappa: float = KAPPA_DEFAULT,
    title: str = "PhyloTree Viewer",
) -> ViewerData:
    """Load ViewerData from an atlas-tree PhyloTree (JSON or dict).

    Extracts all nodes with embeddings (leaf + internal), projects to 2D,
    and preserves edges between included nodes.

    Args:
        source: path to tree.json, or a dict from PhyloTree.to_dict().
        kappa: curvature.
        title: viewer title.

    Returns:
        ViewerData ready for rendering.
    """
    if isinstance(source, (str, Path)):
        with open(source) as f:
            tree_dict = json.load(f)
    else:
        tree_dict = source

    # Extract all nodes with embeddings (not only leaves)
    nodes_with_embeddings = []
    node_id_to_idx: Dict[int, int] = {}
    embeddings_list = []
    taxon_ids = []
    ranks_list = []

    for node in tree_dict.get("nodes", []):
        if node.get("embedding"):
            idx = len(nodes_with_embeddings)
            node_id_to_idx[node["node_id"]] = idx
            nodes_with_embeddings.append(node)
            embeddings_list.append(torch.tensor(node["embedding"], dtype=torch.float32))
            taxon_ids.append(node.get("taxon_id") or f"node_{node['node_id']}")
            ranks_list.append(-1)  # Unknown rank from tree

    if not embeddings_list:
        return _empty_viewer_data(kappa, title)

    embeddings = torch.stack(embeddings_list)  # (N, D)

    # Project to 2D
    proj = tangent_pca_projection(embeddings, kappa)
    coords_2d = proj.coords_2d

    # Extract edges between included nodes
    edges = []
    for edge in tree_dict.get("edges", []):
        src = edge["source"]
        tgt = edge["target"]
        if src in node_id_to_idx and tgt in node_id_to_idx:
            edges.append((
                node_id_to_idx[src],
                node_id_to_idx[tgt],
                edge.get("length", 0.0),
            ))

    # Estimate rank bands from radial positions
    rank_radii = _default_rank_radii(coords_2d, kappa)
    rank_bands = compute_rank_bands(rank_radii, kappa)

    return ViewerData(
        coords_2d=coords_2d,
        taxon_ids=taxon_ids,
        edges=edges,
        ranks=ranks_list,
        lineages=[""] * len(taxon_ids),
        rank_bands=rank_bands,
        kappa=kappa,
        title=title,
        projection_variance=proj.variance_explained,
    )


def from_embeddings(
    embeddings: Union[str, Path, Tensor],
    taxa: Union[str, Path, List[str]],
    kappa: float = KAPPA_DEFAULT,
    lineages: Optional[Union[str, Path, List[str]]] = None,
    title: str = "Embedding Viewer",
) -> ViewerData:
    """Load ViewerData from raw embeddings and taxa list.

    Args:
        embeddings: path to .pt/.npy file, or (N, D) tensor.
        taxa: path to taxa file (one per line), or list of strings.
        kappa: curvature.
        lineages: optional lineage strings (semicolon-separated).
        title: viewer title.

    Returns:
        ViewerData ready for rendering.
    """
    # Load embeddings
    if isinstance(embeddings, (str, Path)):
        emb_path = Path(embeddings)
        if emb_path.suffix == ".pt":
            emb = torch.load(str(emb_path), map_location="cpu", weights_only=True)
        elif emb_path.suffix == ".npy":
            import numpy as np
            emb = torch.from_numpy(np.load(str(emb_path))).float()
        else:
            raise ValueError(f"Unsupported embedding format: {emb_path.suffix}")
    else:
        emb = embeddings

    # Load taxa
    if isinstance(taxa, (str, Path)):
        with open(taxa) as f:
            taxon_ids = [line.strip() for line in f if line.strip()]
    else:
        taxon_ids = list(taxa)

    # Load lineages
    lineage_list: List[str] = []
    if lineages is not None:
        if isinstance(lineages, (str, Path)):
            with open(lineages) as f:
                lineage_list = [line.strip() for line in f if line.strip()]
        else:
            lineage_list = list(lineages)

    if not lineage_list:
        lineage_list = [""] * len(taxon_ids)

    # Infer ranks from lineages
    ranks_list = []
    for lin in lineage_list:
        if lin:
            parts = lin.split(";")
            ranks_list.append(min(len(parts) - 1, 6))
        else:
            ranks_list.append(-1)

    # Project to 2D
    proj = tangent_pca_projection(emb, kappa)

    # Estimate rank bands
    if any(r >= 0 for r in ranks_list):
        rank_radii = estimate_rank_radii(proj.coords_2d, ranks_list, kappa)
    else:
        rank_radii = _default_rank_radii(proj.coords_2d, kappa)
    rank_bands = compute_rank_bands(rank_radii, kappa)

    return ViewerData(
        coords_2d=proj.coords_2d,
        taxon_ids=taxon_ids,
        edges=[],
        ranks=ranks_list,
        lineages=lineage_list,
        rank_bands=rank_bands,
        kappa=kappa,
        title=title,
        projection_variance=proj.variance_explained,
    )


# -- Helpers -------------------------------------------------------------------

def _default_rank_radii(
    coords_2d: Tensor, kappa: float = KAPPA_DEFAULT
) -> Dict[str, float]:
    """Estimate rank radii by evenly spacing across the radial range."""
    radii = dist_from_origin(coords_2d, kappa)
    r_max = radii.max().item()
    if r_max < 1e-6:
        r_max = 1.0

    n_ranks = len(RANK_ORDER)
    return {
        name: r_max * (i + 1) / (n_ranks + 1)
        for i, name in enumerate(RANK_ORDER)
    }


def _empty_viewer_data(kappa: float, title: str) -> ViewerData:
    return ViewerData(
        coords_2d=torch.zeros(0, 2),
        taxon_ids=[],
        edges=[],
        ranks=[],
        lineages=[],
        rank_bands=[],
        kappa=kappa,
        title=title,
    )
