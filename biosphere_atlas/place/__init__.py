"""
atlas-place: Phylogenetic placement via BiosphereAtlas hyperbolic coordinates.
==============================================================================

Drop-in replacement for pplacer / GTDB-Tk classify_wf.

    sequence → embedding → nearest-prototype → calibrated placement

Every query produces a BiosphereAtlas (r, θ) coordinate and a conformal
three-zone decision (accept / escalate / fallback) with formal coverage
guarantees.

Quick start:
    from biosphere_atlas.place import place_sequences, ReferenceDB

    ref = ReferenceDB.load("reference.pkl")
    results = place_sequences("input.fasta", ref)
"""

__version__ = "0.1.0"

from biosphere_atlas.place.calibrator import PlacementCalibrator
from biosphere_atlas.core.atlas import Atlas
from biosphere_atlas.place.place import place_embeddings, place_sequences, placement_summary
from biosphere_atlas.place.placer import PlacementEngine, PlacementResult
from biosphere_atlas.place.reference import Rank, RANKS, ReferenceDB

__all__ = [
    "place_sequences",
    "place_embeddings",
    "placement_summary",
    "PlacementEngine",
    "PlacementResult",
    "PlacementCalibrator",
    "Atlas",
    "ReferenceDB",
    "Rank",
    "RANKS",
]
