"""
BiosphereAtlas v1.0.0 — Canonical coordinate system for biology.

One manifold. One curvature. Seven tools.

    core/     — KAPPA=5/4 datum, Poincaré geometry, (r,θ) coordinates
    chimera/  — Geometric chimera detection (Karcher mean + tangent bimodality)
    place/    — Geodesic hierarchical placement with conformal calibration
    hplg/     — Hyperbolic-Primary Likelihood-Gated classification
    novelty/  — Conformal novelty scoring
    tree/     — Phylogenetic tree construction
    viewer/   — 3D Poincaré ball visualization
"""

__version__ = "1.0.0"

from biosphere_atlas.core.hyperbolic import KAPPA_DEFAULT as KAPPA

