"""
atlas-viewer: Interactive Poincare disk viewer for BiosphereAtlas.
==================================================================

The geometry IS the visualization.  The Poincare disk renders the
hyperbolic manifold directly — LUCA at the origin, evolutionary
divergence as radial distance, the boundary as the frontier of life.
Navigation via Mobius transformations smoothly recenters the view
while preserving all geometric relationships.

Quick start::

    from biosphere_atlas.viewer import from_tree, generate_viewer_html

    data = from_tree("tree.json")
    generate_viewer_html(data, "viewer.html")

All operations use the Poincare ball model with curvature kappa = 1.247
(Fenn & Fenn 2025), matching the BiosphereAtlas training geometry.

Copyright (c) 2025 Sentry Bio, Inc.
"""

from .data import ViewerData, from_tree, from_embeddings
from .mobius import Mobius2D, NavigationState
from .projection import (
    ProjectionResult,
    tangent_pca_projection,
    preserve_radial_order,
    project_single,
)
from .render import generate_viewer_html

__all__ = [
    # Data layer
    "ViewerData",
    "from_tree",
    "from_embeddings",
    # Projection
    "ProjectionResult",
    "tangent_pca_projection",
    "preserve_radial_order",
    "project_single",
    # Navigation
    "Mobius2D",
    "NavigationState",
    # Rendering
    "generate_viewer_html",
]
