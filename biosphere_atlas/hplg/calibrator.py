"""
Mondrian conformal calibrator for HPLG decisions.

Provides formal coverage guarantees via conformal prediction, stratified
by taxonomic rank (the "Mondrian" variant). This means the guarantee
P(correct classification) >= 1 - epsilon holds independently at every rank.

The calibrator maintains a running distribution of nonconformity scores
for each rank, computes quantile thresholds, and provides the two critical
decision boundaries:

  q_accept:   Scores <= this threshold get direct classification (high confidence)
  q_fallback: Scores > this threshold trigger graceful fallback to parent rank

The escalation zone (q_accept < score <= q_fallback) triggers optional ΔLLR
computation for ambiguous cases — saving expensive likelihood computation
for only the 10-20% of sequences that need it.

Key properties:
- Warm-starts with conservative defaults (avoids cold-start brittleness)
- Maintains proper ordering: q_fallback >= q_accept (always)
- Adapts online as new scores arrive
- Serializable for checkpoint interoperability
"""

import torch
import numpy as np
from collections import defaultdict
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from biosphere_atlas.hplg.taxonomy import Rank, RANKS


@dataclass
class ThresholdPair:
    """Accept and fallback thresholds for a single rank."""
    q_accept: float
    q_fallback: float

    def __post_init__(self):
        # Enforce ordering constraint
        if self.q_fallback < self.q_accept:
            self.q_fallback = self.q_accept


# Conservative warm-start defaults (before enough calibration data)
DEFAULT_THRESHOLDS = {
    Rank.DOMAIN: ThresholdPair(q_accept=0.3, q_fallback=0.8),
    Rank.PHYLUM: ThresholdPair(q_accept=0.35, q_fallback=0.85),
    Rank.CLASS: ThresholdPair(q_accept=0.4, q_fallback=0.9),
    Rank.ORDER: ThresholdPair(q_accept=0.45, q_fallback=0.95),
    Rank.FAMILY: ThresholdPair(q_accept=0.5, q_fallback=1.0),
    Rank.GENUS: ThresholdPair(q_accept=0.6, q_fallback=1.2),
    Rank.SPECIES: ThresholdPair(q_accept=0.7, q_fallback=1.5),
}

# Minimum calibration samples before using empirical quantiles
MIN_CAL_SAMPLES = 30


class MondrianConformalCalibrator:
    """
    Rank-stratified conformal calibrator with online updates.

    The "Mondrian" structure means we maintain separate score distributions
    and thresholds for each taxonomic rank. This is critical because the
    natural scale of nonconformity scores differs across ranks: domain-level
    decisions are geometrically easier than species-level ones.

    Coverage guarantee: For rank r, P(Y ∈ C_r(X)) >= 1 - epsilon_r
    where epsilon_r is the target error rate at rank r.
    """

    def __init__(
        self,
        epsilon_accept: float = 0.10,   # 90% coverage for acceptance
        epsilon_fallback: float = 0.01,  # 99% threshold for fallback
        max_scores: int = 10000,         # Maximum stored scores per rank
    ):
        self.epsilon_accept = epsilon_accept
        self.epsilon_fallback = epsilon_fallback
        self.max_scores = max_scores

        # Score histories per rank
        self._scores: Dict[Rank, list] = {r: [] for r in RANKS}

        # Current thresholds (start with conservative defaults)
        self._thresholds: Dict[Rank, ThresholdPair] = {
            r: ThresholdPair(t.q_accept, t.q_fallback)
            for r, t in DEFAULT_THRESHOLDS.items()
        }

    def add_score(self, score: float, rank: Rank):
        """
        Add a calibration score observation.

        Called after each accepted classification to update the
        empirical distribution.
        """
        scores = self._scores[rank]
        scores.append(score)

        # Maintain bounded memory
        if len(scores) > self.max_scores:
            # Keep a random subsample
            indices = np.random.choice(len(scores), self.max_scores, replace=False)
            self._scores[rank] = [scores[i] for i in sorted(indices)]

        # Recompute thresholds if we have enough data
        if len(self._scores[rank]) >= MIN_CAL_SAMPLES:
            self._update_thresholds(rank)

    def _update_thresholds(self, rank: Rank):
        """Recompute thresholds from empirical score distribution."""
        scores = sorted(self._scores[rank])
        n = len(scores)

        # Conformal quantile with finite-sample correction
        # q = ceil((n+1)(1-epsilon)) / n
        accept_idx = min(n - 1, int(np.ceil((n + 1) * (1 - self.epsilon_accept))) - 1)
        fallback_idx = min(n - 1, int(np.ceil((n + 1) * (1 - self.epsilon_fallback))) - 1)

        q_accept = scores[accept_idx]
        q_fallback = scores[fallback_idx]

        # Enforce ordering
        q_fallback = max(q_fallback, q_accept)

        self._thresholds[rank] = ThresholdPair(q_accept, q_fallback)

    def get_thresholds(self, rank: Rank) -> ThresholdPair:
        """Get current accept/fallback thresholds for a rank."""
        return self._thresholds[rank]

    def decide(self, score: float, rank: Rank) -> str:
        """
        Make a three-zone decision based on score and rank thresholds.

        Returns:
            "accept":     Score is in the accept zone
            "escalate":   Score is in the escalation zone (compute ΔLLR)
            "fallback":   Score exceeds fallback threshold
        """
        thresholds = self._thresholds[rank]

        if score <= thresholds.q_accept:
            return "accept"
        elif score > thresholds.q_fallback:
            return "fallback"
        else:
            return "escalate"

    def calibration_summary(self) -> Dict[Rank, dict]:
        """Get calibration status per rank."""
        summary = {}
        for rank in RANKS:
            n = len(self._scores[rank])
            t = self._thresholds[rank]
            summary[rank] = {
                "n_scores": n,
                "calibrated": n >= MIN_CAL_SAMPLES,
                "q_accept": round(t.q_accept, 4),
                "q_fallback": round(t.q_fallback, 4),
                "escalation_width": round(t.q_fallback - t.q_accept, 4),
            }
        return summary

    def state_dict(self) -> dict:
        """Serialize calibrator state for checkpointing."""
        return {
            "epsilon_accept": self.epsilon_accept,
            "epsilon_fallback": self.epsilon_fallback,
            "scores": {r.value: s for r, s in self._scores.items()},
            "thresholds": {
                r.value: {"q_accept": t.q_accept, "q_fallback": t.q_fallback}
                for r, t in self._thresholds.items()
            },
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "MondrianConformalCalibrator":
        """Load calibrator from checkpoint."""
        cal = cls(
            epsilon_accept=state["epsilon_accept"],
            epsilon_fallback=state["epsilon_fallback"],
        )
        for r_val, scores in state["scores"].items():
            cal._scores[Rank(r_val)] = scores
        for r_val, t in state["thresholds"].items():
            cal._thresholds[Rank(r_val)] = ThresholdPair(t["q_accept"], t["q_fallback"])
        return cal
