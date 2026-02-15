"""
Conformal calibration for phylogenetic placement.
==================================================

Implements rank-stratified conformal prediction with three-zone decisions,
following the atlas-hplg pattern adapted for placement:

    Accept   :  A ≤ q_accept     →  high-confidence single placement
    Escalate :  q_accept < A ≤ q_fallback  →  prediction set of plausible taxa
    Fallback :  A > q_fallback   →  abstain / escalate to coarser rank

The nonconformity score A for placement combines:
    A = d_best - η·margin + ρ·r_evolutionary

where:
    d_best  = geodesic distance to nearest prototype
    margin  = d_2nd - d_best  (gap between 1st and 2nd nearest)
    r       = radial coordinate (evolutionary depth from LUCA)

Conformal quantiles are computed per-rank with finite-sample correction:
    q = ⌈(n+1)(1-ε)⌉ / n
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from biosphere_atlas.place.reference import Rank, RANKS


# ── Nonconformity scoring ────────────────────────────────────────────────────

@dataclass
class ScorerWeights:
    """Per-rank weights for the nonconformity score."""

    eta: float = 1.0
    """Weight on margin (always 1.0 — margin is the primary discriminant)."""

    rho: float = 0.1
    """Weight on radial coordinate.  Higher for deeper ranks (more novel sequences
    at tips should incur higher nonconformity)."""


# Default per-rank weights: margin weight fixed, radial weight increases with depth
DEFAULT_WEIGHTS: Dict[Rank, ScorerWeights] = {
    Rank.DOMAIN:  ScorerWeights(eta=1.0, rho=0.05),
    Rank.PHYLUM:  ScorerWeights(eta=1.0, rho=0.08),
    Rank.CLASS:   ScorerWeights(eta=1.0, rho=0.10),
    Rank.ORDER:   ScorerWeights(eta=1.0, rho=0.12),
    Rank.FAMILY:  ScorerWeights(eta=1.0, rho=0.15),
    Rank.GENUS:   ScorerWeights(eta=1.0, rho=0.20),
    Rank.SPECIES: ScorerWeights(eta=1.0, rho=0.25),
}


def compute_nonconformity(
    best_distance: float,
    margin: float,
    atlas_r: float,
    rank: Rank = Rank.SPECIES,
    weights: Optional[Dict[Rank, ScorerWeights]] = None,
) -> float:
    """
    Compute the nonconformity score for a placement.

        A = d_best - η·margin + ρ·r

    Lower A → more conforming (better placement).  A placement with
    a small nearest distance, large margin, and moderate evolutionary
    depth is highly conforming.
    """
    w = (weights or DEFAULT_WEIGHTS).get(rank, ScorerWeights())
    # Clamp margin to avoid inf in score
    m = min(margin, 20.0)
    return best_distance - w.eta * m + w.rho * atlas_r


# ── Threshold pair ───────────────────────────────────────────────────────────

@dataclass
class ThresholdPair:
    """Conformal thresholds for three-zone decisions."""

    q_accept: float
    """Nonconformity quantile for acceptance."""

    q_fallback: float
    """Nonconformity quantile for fallback (must be > q_accept)."""

    def __post_init__(self):
        if self.q_accept > self.q_fallback:
            self.q_accept, self.q_fallback = self.q_fallback, self.q_accept

    def decide(self, score: float) -> str:
        """Return 'accept', 'escalate', or 'fallback'."""
        if score <= self.q_accept:
            return "accept"
        elif score <= self.q_fallback:
            return "escalate"
        else:
            return "fallback"


# Conservative defaults before calibration (wide acceptance)
DEFAULT_THRESHOLDS: Dict[Rank, ThresholdPair] = {
    Rank.DOMAIN:  ThresholdPair(q_accept=0.8, q_fallback=2.0),
    Rank.PHYLUM:  ThresholdPair(q_accept=1.0, q_fallback=2.5),
    Rank.CLASS:   ThresholdPair(q_accept=1.2, q_fallback=3.0),
    Rank.ORDER:   ThresholdPair(q_accept=1.4, q_fallback=3.5),
    Rank.FAMILY:  ThresholdPair(q_accept=1.6, q_fallback=4.0),
    Rank.GENUS:   ThresholdPair(q_accept=2.0, q_fallback=5.0),
    Rank.SPECIES: ThresholdPair(q_accept=2.5, q_fallback=6.0),
}


# ── Conformal calibrator ────────────────────────────────────────────────────

class PlacementCalibrator:
    """
    Rank-stratified conformal prediction for placement confidence.

    Online calibration: feed in (score, rank) pairs from a calibration set,
    then use the calibrated thresholds for three-zone decisions on new queries.

    Coverage guarantee:  P(correct placement ∈ prediction set) ≥ 1 - ε
    at each rank, provided the calibration set is exchangeable with the
    test distribution.
    """

    def __init__(
        self,
        epsilon: float = 0.10,
        fallback_epsilon: float = 0.01,
        weights: Optional[Dict[Rank, ScorerWeights]] = None,
    ):
        """
        Args:
            epsilon: target miscoverage rate for acceptance.
            fallback_epsilon: target miscoverage rate for fallback
                (stricter — we rarely want to abstain on correct placements).
            weights: optional per-rank scorer weights.
        """
        self.epsilon = epsilon
        self.fallback_epsilon = fallback_epsilon
        self.weights = weights or DEFAULT_WEIGHTS

        # Calibration scores per rank
        self._scores: Dict[Rank, List[float]] = {r: [] for r in RANKS}

        # Computed thresholds (start with conservative defaults)
        self._thresholds: Dict[Rank, ThresholdPair] = dict(DEFAULT_THRESHOLDS)

        # Track calibration state
        self._calibrated = False
        self._total_scores = 0

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def total_calibration_scores(self) -> int:
        return self._total_scores

    # ── Calibration ──────────────────────────────────────────────────────

    def add_score(self, score: float, rank: Rank) -> None:
        """Add a nonconformity score from the calibration set."""
        self._scores[rank].append(score)
        self._total_scores += 1
        self._update_thresholds(rank)

    def add_scores_batch(
        self,
        scores: List[float],
        ranks: List[Rank],
    ) -> None:
        """Add a batch of calibration scores."""
        for score, rank in zip(scores, ranks):
            self._scores[rank].append(score)
            self._total_scores += 1
        # Update all affected ranks
        for rank in set(ranks):
            self._update_thresholds(rank)

    def _update_thresholds(self, rank: Rank) -> None:
        """Recompute conformal quantiles for a given rank."""
        scores = sorted(self._scores[rank])
        n = len(scores)
        if n < 5:
            # Not enough data — keep defaults
            return

        # Conformal quantile with finite-sample correction:
        #   q_{1-ε} = scores[⌈(n+1)(1-ε)⌉ - 1]
        accept_idx = min(math.ceil((n + 1) * (1 - self.epsilon)) - 1, n - 1)
        fallback_idx = min(math.ceil((n + 1) * (1 - self.fallback_epsilon)) - 1, n - 1)

        self._thresholds[rank] = ThresholdPair(
            q_accept=scores[max(0, accept_idx)],
            q_fallback=scores[max(0, fallback_idx)],
        )
        self._calibrated = True

    # ── Decision making ──────────────────────────────────────────────────

    def decide(self, score: float, rank: Rank) -> str:
        """
        Three-zone decision for a nonconformity score.

        Returns: 'accept', 'escalate', or 'fallback'.
        """
        thresholds = self._thresholds.get(rank, DEFAULT_THRESHOLDS[Rank.SPECIES])
        return thresholds.decide(score)

    def confidence(self, score: float, rank: Rank) -> float:
        """
        Calibrated confidence ∈ [0, 1].

        Uses the empirical p-value:  conf = 1 - (rank_of_score / (n + 1))
        If uncalibrated, returns a heuristic based on score magnitude.
        """
        scores = self._scores.get(rank, [])
        if len(scores) < 5:
            # Heuristic: sigmoid of negative score
            return 1.0 / (1.0 + math.exp(score))

        n = len(scores)
        # Count how many calibration scores are ≥ this score
        rank_of_score = sum(1 for s in scores if s >= score)
        return rank_of_score / (n + 1)

    def prediction_set_size(self, score: float, rank: Rank) -> int:
        """
        Estimate the size of the conformal prediction set.

        In the accept zone, the set is a singleton.
        In escalate, estimate from calibration distribution.
        In fallback, the set is "unbounded" (return -1).
        """
        zone = self.decide(score, rank)
        if zone == "accept":
            return 1
        elif zone == "fallback":
            return -1  # unbounded / abstain
        else:
            # Escalate: estimate set size from score distribution
            scores = self._scores.get(rank, [])
            if not scores:
                return 3  # conservative default
            thresholds = self._thresholds.get(rank)
            if thresholds is None:
                return 3
            # Rough estimate: linear interpolation in escalate zone
            width = thresholds.q_fallback - thresholds.q_accept
            if width <= 0:
                return 2
            position = (score - thresholds.q_accept) / width
            return max(2, int(2 + position * 8))  # 2–10

    # ── Apply to placement results ───────────────────────────────────────

    def calibrate_placement(
        self,
        placement: "PlacementResult",
        rank: Optional[Rank] = None,
    ) -> "PlacementResult":
        """
        Annotate a PlacementResult with conformal zone, confidence,
        and prediction set size.

        If rank is not specified, uses the rank of the best candidate.
        """
        from biosphere_atlas.place.placer import PlacementResult

        if not placement.candidates:
            placement.zone = "fallback"
            placement.confidence = 0.0
            placement.prediction_set_size = -1
            return placement

        if rank is None:
            rank = placement.candidates[0].rank

        score = compute_nonconformity(
            best_distance=placement.best_distance,
            margin=placement.margin,
            atlas_r=placement.atlas_r,
            rank=rank,
            weights=self.weights,
        )

        placement.zone = self.decide(score, rank)
        placement.confidence = self.confidence(score, rank)
        placement.prediction_set_size = self.prediction_set_size(score, rank)

        return placement

    def calibrate_batch(
        self,
        placements: List["PlacementResult"],
        ranks: Optional[List[Rank]] = None,
    ) -> List["PlacementResult"]:
        """Calibrate a batch of placement results."""
        for i, p in enumerate(placements):
            r = ranks[i] if ranks else None
            self.calibrate_placement(p, rank=r)
        return placements

    # ── Serialization ────────────────────────────────────────────────────

    def state_dict(self) -> Dict:
        """Export calibrator state."""
        return {
            "epsilon": self.epsilon,
            "fallback_epsilon": self.fallback_epsilon,
            "scores": {r.value: s for r, s in self._scores.items()},
            "thresholds": {
                r.value: {"q_accept": t.q_accept, "q_fallback": t.q_fallback}
                for r, t in self._thresholds.items()
            },
        }

    @classmethod
    def from_state_dict(cls, state: Dict) -> "PlacementCalibrator":
        """Restore calibrator from state dict."""
        cal = cls(
            epsilon=state["epsilon"],
            fallback_epsilon=state["fallback_epsilon"],
        )
        for rank_val, scores in state["scores"].items():
            rank = Rank(int(rank_val))
            cal._scores[rank] = scores
            cal._total_scores += len(scores)
            cal._update_thresholds(rank)
        return cal
