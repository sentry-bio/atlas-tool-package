"""
atlas-hplg: Hyperbolic-Primary, Likelihood-Gated taxonomic classifier.

Production-ready biological sequence classification with formal statistical
guarantees against confident mislabeling. Built on the BiosphereAtlas
coordinate system (kappa = 1.247).

Key insight: biological sequence space has multiple valid organizations
(functional at kappa~1.0, phylogenetic at kappa~1.2475), and the HPLG
three-zone decision framework exploits this geometry for calibrated
classification at every taxonomic rank.

Three-zone decisions:
  Accept:     A_r <= q_accept     -> Direct classification
  Escalation: q_accept < A_r <= q_fallback -> Compute ΔLLR for refinement
  Fallback:   A_r > q_fallback    -> Return to parent rank (graceful)

Formal guarantee: Coverage >= 1 - epsilon via Mondrian conformal prediction.
"""

__version__ = "0.1.0"

from biosphere_atlas.hplg.taxonomy import Rank, RANKS, Taxonomy
from biosphere_atlas.hplg.classifier import HPLGClassifier, ClassificationResult

__all__ = [
    "HPLGClassifier",
    "ClassificationResult",
    "Rank",
    "RANKS",
    "Taxonomy",
]
