"""atlas-dark: Dark matter biology mapper via geodesic uncertainty fields.

Maps the unknown regions of the BiosphereAtlas — coordinate-space areas
with high uncertainty and sparse prototype coverage.  The uncertainty
field sigma_local(x) is computed from geodesic distances to k-nearest
reference prototypes.  High sigma = "the Atlas has never seen this."

Three-way triage for new genomes:
  Redundant:        d_geo < epsilon               -> link to existing prototype
  Novel-Certain:    d_geo > epsilon, sigma low     -> place via atlas-place
  Novel-Uncertain:  d_geo > epsilon, sigma high    -> true dark matter, needs training
"""

from biosphere_atlas.dark.field import UncertaintyField
from biosphere_atlas.dark.dark import DarkMatterMap, DarkRegion
from biosphere_atlas.dark.triage import triage_embedding, triage_genomes, TriageResult, TriageCategory

__all__ = [
    "UncertaintyField",
    "DarkMatterMap",
    "DarkRegion",
    "triage_embedding",
    "triage_genomes",
    "TriageResult",
    "TriageCategory",
]

