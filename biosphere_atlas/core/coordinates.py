"""
BiosphereAtlas coordinate extraction.

Every sequence processed by atlas-chimera receives a coordinate in the
universal coordinate system for biology. This is the Trojan horse:
researchers come for chimera detection, and every sequence they run
gets a BiosphereAtlas address.

The coordinate system is 2D (r, theta) in the Poincare disk:
- r (radial): Evolutionary depth — distance from LUCA at the origin.
  Deeper lineages have larger r values.
- theta (angular): Phylogenetic direction — position among alternative
  evolutionary trajectories at a given depth.

Together, (r, theta) gives every organism a unique address in the
geometry of life, analogous to (latitude, longitude) on Earth.

The n=2 dimensionality is not a compression artifact but an intrinsic
property of evolution (Fenn & Fenn 2025): all evolutionary processes
navigate a strictly two-dimensional manifold embedded in hyperbolic space.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional

from biosphere_atlas.core.hyperbolic import (
    KAPPA_DEFAULT,
    dist_from_origin,
    karcher_mean,
)


@dataclass
class BiosphereCoordinate:
    """
    A position in the BiosphereAtlas coordinate system.

    Attributes:
        r: Radial coordinate (evolutionary depth from LUCA).
           Range: [0, inf). Deeper lineages have larger r.
        theta: Angular coordinate (phylogenetic direction).
           Range: [0, 2*pi). Position among alternative trajectories.
        r_euclidean: Euclidean radius in the Poincare disk (for visualization).
           Range: [0, 1). Closer to 1 = deeper in the tree.
        embedding_dim: Full embedding dimensionality (for advanced use).
        kappa: Curvature used for this coordinate.
    """
    r: float
    theta: float
    r_euclidean: float
    kappa: float = KAPPA_DEFAULT
    embedding_dim: int = 128

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON/TSV output."""
        return {
            "r": round(self.r, 6),
            "theta": round(self.theta, 6),
            "theta_degrees": round(np.degrees(self.theta), 2),
            "r_euclidean": round(self.r_euclidean, 6),
            "kappa": self.kappa,
        }

    def __repr__(self) -> str:
        return (
            f"BiosphereCoordinate(r={self.r:.4f}, "
            f"theta={np.degrees(self.theta):.1f} deg, "
            f"kappa={self.kappa})"
        )


def extract_coordinate(
    embedding: torch.Tensor,
    kappa: float = KAPPA_DEFAULT,
) -> BiosphereCoordinate:
    """
    Extract BiosphereAtlas (r, theta) coordinate from a high-dimensional embedding.

    The 2D coordinate is computed by:
    1. r = hyperbolic distance from origin (proper geodesic distance, not Euclidean norm)
    2. theta = angle of the embedding projected onto the first two principal components

    For the full BiosphereCodec model, the principal components are aligned with
    phylogenetic structure. For the lightweight version, we use the first two
    dimensions of the embedding as a proxy (these capture the dominant phylogenetic
    signal due to the hierarchical pooling architecture).

    Args:
        embedding: Poincare ball embedding, shape (dim,)
        kappa: Curvature parameter

    Returns:
        BiosphereCoordinate with (r, theta) address
    """
    # Radial: proper hyperbolic distance from origin
    r_hyp = dist_from_origin(embedding.unsqueeze(0), kappa).item()

    # Euclidean radius (for Poincare disk visualization)
    r_euc = embedding.norm().item()

    # Angular: use first two dimensions as a lightweight proxy.
    # TODO(v0.3): switch to tangent-space PCA projection for theta to stay
    # consistent with atlas-viewer and future depth-aware architectures.
    x = embedding[0].item()
    y = embedding[1].item()
    theta = np.arctan2(y, x) % (2 * np.pi)

    return BiosphereCoordinate(
        r=r_hyp,
        theta=theta,
        r_euclidean=r_euc,
        kappa=kappa,
        embedding_dim=embedding.shape[0],
    )


def extract_coordinates_batch(
    embeddings: torch.Tensor,
    kappa: float = KAPPA_DEFAULT,
) -> list:
    """
    Extract BiosphereAtlas coordinates for a batch of embeddings.

    Args:
        embeddings: Poincare ball embeddings, shape (n, dim)
        kappa: Curvature parameter

    Returns:
        List of BiosphereCoordinate objects
    """
    return [extract_coordinate(embeddings[i], kappa) for i in range(embeddings.shape[0])]
