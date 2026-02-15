"""
atlas-chimera: Geometry-based chimera detection for metagenomic sequences.

Detects chimeric sequences by identifying geometric anomalies in hyperbolic
embedding space. Based on the BiosphereCodec architecture and the discovery
that biological information organizes in hyperbolic space with curvature
kappa = 1.247 +/- 0.003 (Fenn & Fenn 2025).

Unlike reference-based chimera detectors (UCHIME, ChimeraSlayer), this tool
identifies chimeras through coordinate-space anomalies — sequences that
occupy geometrically impossible positions, as if claiming two phylogenetic
addresses simultaneously.

Every sequence processed receives both a chimera score and a BiosphereAtlas
coordinate (r, theta) in the universal coordinate system for biology.

Reference:
    Fenn, R. & Fenn, A. (2025). Evolution as Active Geometry:
    A Universal Curvature Constant. bioRxiv.

License: MIT
"""

__version__ = "0.1.0"

from biosphere_atlas.chimera.detect import detect_chimeras, ChimeraResult
from biosphere_atlas.core.coordinates import BiosphereCoordinate

__all__ = ["detect_chimeras", "ChimeraResult", "BiosphereCoordinate"]
