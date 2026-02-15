"""
Nonconformity scorer for HPLG decisions.

The nonconformity score is the unified signal that drives three-zone decisions.
It combines geometric distance, margin, evolutionary depth, and optional
log-likelihood ratio into a single scalar per query:

  A_r = distance_best - eta[r] * margin + rho[r] * radius - zeta[r] * ΔLLR

Where:
  distance_best: Poincare distance to nearest prototype
  margin:        distance_best - distance_second (gap between top-2 candidates)
  radius:        dist_from_origin (evolutionary depth / radial coordinate)
  ΔLLR:          log-likelihood ratio between top-2 candidates (optional, expensive)

The per-rank weights (eta, rho, zeta) are learned during calibration. Their
interpretation:
  eta:   Margin importance — how much does the gap between best and second matter?
  rho:   Radius importance — are deeper lineages harder to classify?
  zeta:  ΔLLR importance — how much does the likelihood signal help?

Lower scores = more confident classification.
"""

import torch
from dataclasses import dataclass
from typing import Dict, Optional

from biosphere_atlas.hplg.taxonomy import Rank
from biosphere_atlas.core.hyperbolic import (
    KAPPA_DEFAULT,
    poincare_distance,
    dist_from_origin,
)


@dataclass
class ScorerWeights:
    """Per-rank weights for the nonconformity scorer."""
    eta: float = 1.0   # margin importance
    rho: float = 0.3   # radius importance
    zeta: float = 1.2  # ΔLLR importance


# Default weights (good starting point; fine-tune during calibration)
DEFAULT_SCORER_WEIGHTS = {
    Rank.DOMAIN: ScorerWeights(eta=1.0, rho=0.1, zeta=1.0),
    Rank.PHYLUM: ScorerWeights(eta=1.0, rho=0.2, zeta=1.1),
    Rank.CLASS: ScorerWeights(eta=1.0, rho=0.2, zeta=1.1),
    Rank.ORDER: ScorerWeights(eta=1.0, rho=0.25, zeta=1.2),
    Rank.FAMILY: ScorerWeights(eta=1.0, rho=0.25, zeta=1.2),
    Rank.GENUS: ScorerWeights(eta=1.0, rho=0.3, zeta=1.2),
    Rank.SPECIES: ScorerWeights(eta=1.0, rho=0.3, zeta=1.3),
}


class NonconformityScorer:
    """
    Computes unified nonconformity scores for HPLG classification.

    The scorer takes the geometric signals (distance, margin, radius)
    and optional likelihood signals (ΔLLR) and produces a single scalar
    that drives the three-zone decision.

    Usage:
        scorer = NonconformityScorer()
        score = scorer.compute(
            rank=Rank.SPECIES,
            best_dist=0.15,
            second_dist=0.45,
            radius=1.2,
        )
        # score is a float; lower = more confident
    """

    def __init__(
        self,
        weights: Optional[Dict[Rank, ScorerWeights]] = None,
        kappa: float = KAPPA_DEFAULT,
    ):
        self.weights = weights or DEFAULT_SCORER_WEIGHTS
        self.kappa = kappa

    def compute(
        self,
        rank: Rank,
        best_dist: float,
        second_dist: float,
        radius: float,
        delta_llr: Optional[float] = None,
    ) -> float:
        """
        Compute nonconformity score.

        A_r = best_dist - eta * margin + rho * radius - zeta * ΔLLR

        Args:
            rank: Taxonomic rank for weight lookup
            best_dist: Distance to nearest prototype
            second_dist: Distance to second-nearest prototype
            radius: Hyperbolic distance from origin (evolutionary depth)
            delta_llr: Optional log-likelihood ratio (None if not computed)

        Returns:
            Nonconformity score (lower = more confident)
        """
        w = self.weights.get(rank, ScorerWeights())

        margin = second_dist - best_dist  # positive means clear winner
        score = best_dist - w.eta * margin + w.rho * radius

        if delta_llr is not None:
            score -= w.zeta * delta_llr

        return score

    def compute_batch(
        self,
        rank: Rank,
        embeddings: torch.Tensor,
        prototype_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Vectorized nonconformity scoring for a batch of embeddings.

        Args:
            rank: Taxonomic rank
            embeddings: Query embeddings, shape (B, D)
            prototype_embeddings: Prototype embeddings, shape (N, D)

        Returns:
            Scores tensor, shape (B,)
        """
        w = self.weights.get(rank, ScorerWeights())

        # Compute all pairwise distances: (B, N)
        dists = poincare_distance(
            embeddings.unsqueeze(1),
            prototype_embeddings.unsqueeze(0),
            self.kappa,
        )

        # Top-2 distances
        top2_dists, top2_idx = dists.topk(min(2, dists.size(1)), dim=1, largest=False)
        best_dist = top2_dists[:, 0]

        if top2_dists.size(1) >= 2:
            second_dist = top2_dists[:, 1]
            margin = second_dist - best_dist
        else:
            margin = torch.zeros_like(best_dist)

        # Evolutionary depth
        radius = dist_from_origin(embeddings, self.kappa)

        # Unified score (no ΔLLR in batch mode — that's the escalation path)
        scores = best_dist - w.eta * margin + w.rho * radius

        return scores, top2_idx[:, 0]

    def state_dict(self) -> dict:
        """Serialize weights for checkpointing."""
        return {
            r.value: {"eta": w.eta, "rho": w.rho, "zeta": w.zeta}
            for r, w in self.weights.items()
        }

    @classmethod
    def from_state_dict(cls, state: dict, kappa: float = KAPPA_DEFAULT) -> "NonconformityScorer":
        """Load weights from checkpoint."""
        weights = {
            Rank(r_val): ScorerWeights(**w)
            for r_val, w in state.items()
        }
        return cls(weights=weights, kappa=kappa)
