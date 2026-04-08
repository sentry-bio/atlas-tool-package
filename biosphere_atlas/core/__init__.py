"""
Core geometry, coordinates, and I/O utilities.

This is the foundation layer — the canonical coordinate system for biology.
KAPPA_DEFAULT = 1.25 (datum, fixed by BiosphereCoordinate v1.0 specification).
"""

from biosphere_atlas.core.hyperbolic import KAPPA_DEFAULT
from biosphere_atlas.core.coordinates import BiosphereCoordinate, extract_coordinate, extract_coordinates_batch

try:
    from biosphere_atlas.core.atlas import Atlas
except ImportError:
    Atlas = None  # encoder not available in lightweight installs

__all__ = ["KAPPA_DEFAULT", "Atlas", "BiosphereCoordinate", "extract_coordinate", "extract_coordinates_batch"]

