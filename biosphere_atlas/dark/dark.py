"""Dark matter map: identifies under-explored regions of the manifold.

A DarkMatterMap partitions the reference prototypes into dark regions
(high sigma) and charted regions (low sigma) based on the uncertainty
field.  Each region carries its prototype IDs and local sigma statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from biosphere_atlas.dark.field import UncertaintyField
from biosphere_atlas.place.reference import Rank


@dataclass
class DarkRegion:
    """A contiguous region of high uncertainty in the manifold."""
    prototype_ids: List[str]
    mean_sigma: float
    max_sigma: float
    n_prototypes: int


@dataclass
class DarkMatterMap:
    """Partition of the reference set into dark and charted regions.

    Attributes:
        dark_threshold: sigma above which a prototype is 'dark'.
        dark_regions: list of DarkRegion objects.
        charted_ids: prototype IDs with sigma below threshold.
        field_stats: global field statistics.
        summary: human-readable summary dict.
    """
    dark_threshold: float
    dark_regions: List[DarkRegion]
    charted_ids: List[str]
    field_stats: dict
    summary: dict

    @classmethod
    def from_field(
        cls,
        uf: UncertaintyField,
        dark_quantile: float = 0.95,
    ) -> "DarkMatterMap":
        """Build the dark matter map from an uncertainty field.

        Args:
            uf: pre-computed UncertaintyField.
            dark_quantile: percentile above which prototypes are 'dark'.

        Returns:
            DarkMatterMap with classified regions.
        """
        stats = uf.stats()
        sigmas = uf.compute_prototype_sigmas()

        if sigmas.numel() == 0:
            return cls(
                dark_threshold=0.0,
                dark_regions=[],
                charted_ids=[],
                field_stats={},
                summary={"n_dark": 0, "n_charted": 0},
            )

        threshold = float(sigmas.quantile(dark_quantile))
        dark_mask = sigmas > threshold
        charted_mask = ~dark_mask

        dark_ids = [uf._ids[i] for i in range(len(uf._ids)) if dark_mask[i]]
        charted_ids = [uf._ids[i] for i in range(len(uf._ids)) if charted_mask[i]]
        dark_sigmas = sigmas[dark_mask]

        # For v0.1, treat all dark prototypes as one region
        # (proper geodesic clustering can come in v0.2)
        dark_regions = []
        if dark_ids:
            dark_regions.append(DarkRegion(
                prototype_ids=dark_ids,
                mean_sigma=float(dark_sigmas.mean()),
                max_sigma=float(dark_sigmas.max()),
                n_prototypes=len(dark_ids),
            ))

        summary = {
            "n_dark": len(dark_ids),
            "n_charted": len(charted_ids),
            "n_total": len(uf._ids),
            "dark_threshold": threshold,
            "pct_dark": len(dark_ids) / max(1, len(uf._ids)) * 100,
            "mean_sigma_dark": float(dark_sigmas.mean()) if dark_ids else 0.0,
            "mean_sigma_charted": float(sigmas[charted_mask].mean()) if charted_ids else 0.0,
        }

        return cls(
            dark_threshold=threshold,
            dark_regions=dark_regions,
            charted_ids=charted_ids,
            field_stats={
                "mean": stats.mean_sigma,
                "median": stats.median_sigma,
                "std": stats.std_sigma,
                "q95": stats.q95_sigma,
                "q99": stats.q99_sigma,
            },
            summary=summary,
        )

    def to_dict(self) -> dict:
        return {
            "dark_threshold": self.dark_threshold,
            "dark_regions": [
                {
                    "prototype_ids": r.prototype_ids,
                    "mean_sigma": r.mean_sigma,
                    "max_sigma": r.max_sigma,
                    "n_prototypes": r.n_prototypes,
                }
                for r in self.dark_regions
            ],
            "n_charted": len(self.charted_ids),
            "field_stats": self.field_stats,
            "summary": self.summary,
        }

