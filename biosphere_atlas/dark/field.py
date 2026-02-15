"""Continuous uncertainty field over the Poincare ball.

sigma_local(x) measures how well-covered a point x is by existing
reference prototypes.  High sigma = sparse region = dark matter.

The field is computed from k-nearest prototype geodesic distances
using harmonic-mean weighting, which naturally penalizes isolation
(a single nearby prototype with many distant ones still yields high sigma).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from biosphere_atlas.core.hyperbolic import KAPPA_DEFAULT, poincare_distance
from biosphere_atlas.place.reference import Rank, ReferenceDB


@dataclass
class FieldStats:
    """Global statistics of the uncertainty field over the reference set."""
    mean_sigma: float
    median_sigma: float
    std_sigma: float
    q95_sigma: float
    q99_sigma: float
    n_prototypes: int


class UncertaintyField:
    """Continuous geodesic uncertainty field from a ReferenceDB.

    For any point x in the Poincare ball, sigma_local(x) is the
    harmonic mean of geodesic distances to the k nearest prototypes.
    This gives a manifold-aware measure of local coverage:
    - Low sigma: well-charted territory (many nearby prototypes)
    - High sigma: dark matter (sparse or absent coverage)

    Args:
        ref_db: atlas-place ReferenceDB with prototypes.
        rank: which rank-level prototypes to use.
        k: number of nearest neighbors for sigma estimation.
        kappa: curvature override (default: from ref_db).
    """

    def __init__(
        self,
        ref_db: ReferenceDB,
        rank: Rank = Rank.FAMILY,
        k: int = 5,
        kappa: Optional[float] = None,
    ):
        self.ref_db = ref_db
        self.rank = rank
        self.k = k
        self.kappa = float(kappa if kappa is not None else ref_db.kappa)

        ids, emb = ref_db.get_prototypes_at_rank(rank)
        self._ids = ids
        self._emb = emb  # (N, D)
        self._n = emb.shape[0] if emb.numel() > 0 else 0

        # Pre-compute leave-one-out sigmas for threshold calibration
        self._prototype_sigmas: Optional[Tensor] = None
        self._stats: Optional[FieldStats] = None

    @property
    def n_prototypes(self) -> int:
        return self._n

    def sigma(self, x: Tensor) -> float:
        """Compute sigma_local for a single point x.

        Args:
            x: (D,) Poincare ball embedding.

        Returns:
            sigma_local (harmonic mean of k-nearest distances).
        """
        if self._n == 0:
            return float("inf")
        x = x.detach().to(self._emb.device).float()
        q = x.unsqueeze(0).expand_as(self._emb)
        d = poincare_distance(q, self._emb, kappa=self.kappa)
        k = min(self.k, self._n)
        topd, _ = torch.topk(d, k=k, largest=False)
        # Harmonic mean: k / sum(1/d_i) — naturally penalizes isolation
        inv_sum = (1.0 / topd.clamp_min(1e-15)).sum()
        return float(k / inv_sum.item())

    def sigma_batch(self, x: Tensor) -> Tensor:
        """Compute sigma_local for a batch of points.

        Args:
            x: (B, D) embeddings.

        Returns:
            (B,) sigma values.
        """
        if self._n == 0:
            return torch.full((x.shape[0],), float("inf"))
        x = x.detach().to(self._emb.device).float()
        # Pairwise distances (B, N)
        B = x.shape[0]
        N = self._emb.shape[0]
        q = x.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
        r = self._emb.unsqueeze(0).expand(B, N, -1).reshape(B * N, -1)
        d_flat = poincare_distance(q, r, kappa=self.kappa)
        d = d_flat.reshape(B, N)
        k = min(self.k, N)
        topd, _ = torch.topk(d, k=k, largest=False, dim=1)  # (B, k)
        inv_sum = (1.0 / topd.clamp_min(1e-15)).sum(dim=1)  # (B,)
        return k / inv_sum

    def compute_prototype_sigmas(self) -> Tensor:
        """Leave-one-out sigma for each prototype (for threshold calibration)."""
        if self._prototype_sigmas is not None:
            return self._prototype_sigmas
        if self._n < 2:
            self._prototype_sigmas = torch.zeros(max(self._n, 0))
            return self._prototype_sigmas

        sigmas = []
        for i in range(self._n):
            q = self._emb[i].unsqueeze(0).expand_as(self._emb)
            d = poincare_distance(q, self._emb, kappa=self.kappa)
            d[i] = float("inf")  # exclude self
            k = min(self.k, self._n - 1)
            topd, _ = torch.topk(d, k=k, largest=False)
            inv_sum = (1.0 / topd.clamp_min(1e-15)).sum()
            sigmas.append(float(k / inv_sum.item()))

        self._prototype_sigmas = torch.tensor(sigmas)
        return self._prototype_sigmas

    def stats(self) -> FieldStats:
        """Global field statistics from leave-one-out prototype sigmas."""
        if self._stats is not None:
            return self._stats
        s = self.compute_prototype_sigmas()
        if s.numel() == 0:
            self._stats = FieldStats(0, 0, 0, 0, 0, 0)
            return self._stats
        self._stats = FieldStats(
            mean_sigma=float(s.mean()),
            median_sigma=float(s.median()),
            std_sigma=float(s.std()) if s.numel() > 1 else 0.0,
            q95_sigma=float(s.quantile(0.95)),
            q99_sigma=float(s.quantile(0.99)),
            n_prototypes=self._n,
        )
        return self._stats

