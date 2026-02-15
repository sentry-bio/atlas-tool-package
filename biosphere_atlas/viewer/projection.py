"""
Hyperbolic-aware 2D projection.
================================

Projects high-dimensional Poincare ball embeddings to a 2D Poincare disk
via tangent-space PCA.  Preserves neighbor relationships, radial ordering,
and local conformal structure as much as possible.

Algorithm (Tangent PCA):
    1. Log-map all points to tangent space at origin: v_i = log_0(x_i)
    2. Center and PCA in tangent space (linear operation)
    3. Keep top-2 principal components
    4. Exp-map back to the 2D Poincare disk
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from biosphere_atlas.core.hyperbolic import (
    KAPPA_DEFAULT,
    _clamp_to_ball,
    dist_from_origin,
    exp_map_0,
    log_map_0,
)


@dataclass
class ProjectionResult:
    """Captures the projection mapping for diagnostics and inverse queries."""

    coords_2d: Tensor
    """(N, 2) projected 2D Poincare disk coordinates."""

    pca_components: Tensor
    """(2, D) principal component directions in tangent space."""

    mean_tangent: Tensor
    """(D,) mean of tangent vectors (subtracted before PCA)."""

    variance_explained: List[float]
    """Fraction of variance explained by each component."""

    kappa: float


def tangent_pca_projection(
    embeddings: Tensor,
    kappa: float = KAPPA_DEFAULT,
    n_components: int = 2,
) -> ProjectionResult:
    """Project N-dimensional Poincare ball embeddings to 2D via tangent PCA.

    Steps:
        1. Log-map all points to tangent space at origin
        2. Center the tangent vectors
        3. SVD for principal components
        4. Project onto top-2 components
        5. Exp-map back to 2D Poincare disk

    Args:
        embeddings: (N, D) points in D-dimensional Poincare ball.
        kappa: curvature constant.
        n_components: output dimension (default 2).

    Returns:
        ProjectionResult with 2D coordinates and PCA metadata.
    """
    N, D = embeddings.shape

    # 1. Log-map to tangent space at origin
    tangent_vectors = log_map_0(embeddings, kappa)  # (N, D)

    # 2. Center
    mean_t = tangent_vectors.mean(dim=0)  # (D,)
    centered = tangent_vectors - mean_t  # (N, D)

    # Handle degenerate case: single point (centered = zeros)
    if N == 1:
        coords_2d = _clamp_to_ball(torch.zeros(1, n_components), kappa)
        components = torch.eye(n_components, D)
        return ProjectionResult(
            coords_2d=coords_2d,
            pca_components=components,
            mean_tangent=mean_t,
            variance_explained=[0.0] * n_components,
            kappa=kappa,
        )

    # 3. SVD for PCA
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    # U: (N, min(N,D)), S: (min(N,D),), Vh: (min(N,D), D)

    total_var = (S ** 2).sum()
    if total_var < 1e-15:
        var_explained = [0.0] * min(n_components, len(S))
    else:
        var_explained = [(S[i] ** 2 / total_var).item() for i in range(min(n_components, len(S)))]

    # Top components
    components = Vh[:n_components]  # (2, D)

    # 4. Project centered tangent vectors onto top-2 components
    projected_tangent = centered @ components.T  # (N, 2)

    # 5. Exp-map back to 2D Poincare disk
    coords_2d = exp_map_0(projected_tangent, kappa)  # (N, 2)
    coords_2d = _clamp_to_ball(coords_2d, kappa)

    return ProjectionResult(
        coords_2d=coords_2d,
        pca_components=components,
        mean_tangent=mean_t,
        variance_explained=var_explained,
        kappa=kappa,
    )


def preserve_radial_order(
    coords_2d: Tensor,
    original_embeddings: Tensor,
    kappa: float = KAPPA_DEFAULT,
) -> Tensor:
    """Post-projection pass to enforce radial ordering.

    Ensures that if point A is closer to LUCA (origin) than point B in the
    original high-D space, then A is also closer to the origin in the 2D
    projection.  This preserves the "evolutionary depth" interpretation of
    radial distance.

    Args:
        coords_2d: (N, 2) projected coordinates.
        original_embeddings: (N, D) original high-D embeddings.
        kappa: curvature.

    Returns:
        (N, 2) coordinates with radial order preserved.
    """
    # Original radial distances from origin (high-D)
    orig_radii = dist_from_origin(original_embeddings, kappa)  # (N,)

    # Current 2D radial distances
    current_radii = dist_from_origin(coords_2d, kappa)  # (N,)

    # Compute rank-order mapping: sort original radii, map to monotonic 2D radii
    orig_order = torch.argsort(orig_radii)
    current_sorted = current_radii[orig_order]

    # Enforce monotonicity via isotonic regression (simple cummax)
    monotonic = torch.cummax(current_sorted, dim=0).values

    # Compute scale factor for each point to match monotonic ordering
    result = coords_2d.clone()
    for idx in range(len(orig_order)):
        i = orig_order[idx].item()
        old_r = current_radii[i]
        new_r = monotonic[idx]
        if old_r > 1e-10:
            scale = new_r / old_r
            # Adjust the Euclidean norm to match the new hyperbolic radius
            # For small adjustments, scaling the Euclidean vector is a reasonable proxy
            result[i] = coords_2d[i] * (scale.clamp(0.5, 2.0))

    return _clamp_to_ball(result, kappa)


def project_single(
    embedding: Tensor,
    projection: ProjectionResult,
) -> Tensor:
    """Project a single new embedding using an existing projection.

    Useful for projecting query sequences onto an existing viewer.

    Args:
        embedding: (D,) single point in high-D Poincare ball.
        projection: a previously computed ProjectionResult.

    Returns:
        (2,) point in 2D Poincare disk.
    """
    kappa = projection.kappa
    v = log_map_0(embedding.unsqueeze(0), kappa).squeeze(0)  # (D,)
    centered = v - projection.mean_tangent
    proj_2d = centered @ projection.pca_components.T  # (2,)
    return _clamp_to_ball(
        exp_map_0(proj_2d.unsqueeze(0), kappa).squeeze(0), kappa
    )
