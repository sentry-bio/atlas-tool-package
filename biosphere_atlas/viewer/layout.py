"""
Layout, level-of-detail, and annotation.
==========================================

Compute rank bands (geodesic circles), geodesic arcs between points,
level-of-detail filtering, and label placement for the Poincare disk
viewer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from biosphere_atlas.core.hyperbolic import (
    KAPPA_DEFAULT,
    ball_radius,
    dist_from_origin,
    geodesic_interpolation,
    poincare_distance,
)


# -- Rank bands ----------------------------------------------------------------

# GTDB-standard rank colors (domain → species)
RANK_COLORS = {
    "domain": "#e74c3c",   # red
    "phylum": "#e67e22",   # orange
    "class": "#f1c40f",    # yellow
    "order": "#2ecc71",    # green
    "family": "#3498db",   # blue
    "genus": "#9b59b6",    # purple
    "species": "#34495e",  # dark grey
}

RANK_ORDER = ["domain", "phylum", "class", "order", "family", "genus", "species"]


@dataclass
class RankBand:
    """A geodesic circle representing a taxonomic rank boundary."""

    rank_name: str
    hyperbolic_radius: float
    """Geodesic distance from origin (LUCA) defining this rank boundary."""
    color: str
    label: str

    @property
    def euclidean_radius(self) -> float:
        """Euclidean radius in the Poincare disk model.

        For a hyperbolic circle centered at origin with geodesic radius r,
        the Euclidean radius in the Poincare disk is:

            r_e = tanh(sqrt(kappa) * r / 2) / sqrt(kappa)
        """
        sqk = math.sqrt(KAPPA_DEFAULT)
        return math.tanh(sqk * self.hyperbolic_radius / 2.0) / sqk


def compute_rank_bands(
    radii_per_rank: Dict[str, float],
    kappa: float = KAPPA_DEFAULT,
) -> List[RankBand]:
    """Generate rank bands from a mapping of rank name → geodesic radius.

    Args:
        radii_per_rank: e.g. {"domain": 0.5, "phylum": 1.0, ...}
        kappa: curvature.

    Returns:
        Sorted list of RankBand objects.
    """
    bands = []
    for rank_name in RANK_ORDER:
        if rank_name in radii_per_rank:
            color = RANK_COLORS.get(rank_name, "#888888")
            bands.append(RankBand(
                rank_name=rank_name,
                hyperbolic_radius=radii_per_rank[rank_name],
                color=color,
                label=rank_name.capitalize(),
            ))
    bands.sort(key=lambda b: b.hyperbolic_radius)
    return bands


def estimate_rank_radii(
    coords_2d: Tensor,
    ranks: List[int],
    kappa: float = KAPPA_DEFAULT,
) -> Dict[str, float]:
    """Estimate rank boundary radii from actual point positions.

    For each rank level, computes the median geodesic distance from
    origin of all points at that rank.

    Args:
        coords_2d: (N, 2) disk coordinates.
        ranks: (N,) rank levels (0=domain, 6=species).
        kappa: curvature.

    Returns:
        Dict mapping rank name to geodesic radius.
    """
    radii = dist_from_origin(coords_2d, kappa)  # (N,)
    result = {}
    for level, name in enumerate(RANK_ORDER):
        mask = torch.tensor([r == level for r in ranks])
        if mask.any():
            result[name] = radii[mask].median().item()
    return result


# -- Geodesic arcs -------------------------------------------------------------

def geodesic_arc_points(
    p1: Tensor,
    p2: Tensor,
    kappa: float = KAPPA_DEFAULT,
    n_samples: int = 30,
) -> Tensor:
    """Sample points along the geodesic arc from p1 to p2.

    In the Poincare disk, geodesics are circular arcs orthogonal to
    the boundary circle.  We compute the arc via manifold-safe
    interpolation (exp_x(t * log_x(y))).

    Args:
        p1: (2,) start point in disk.
        p2: (2,) end point in disk.
        kappa: curvature.
        n_samples: number of points along the arc.

    Returns:
        (n_samples, 2) points along the geodesic.
    """
    ts = torch.linspace(0, 1, n_samples)
    points = []
    for t in ts:
        pt = geodesic_interpolation(p1, p2, t.item(), kappa)
        points.append(pt)
    return torch.stack(points)


# -- Level of detail -----------------------------------------------------------

def visible_points(
    all_points: Tensor,
    all_labels: List[str],
    ranks: Optional[List[int]] = None,
    center: Optional[Tensor] = None,
    max_visible: int = 500,
    kappa: float = KAPPA_DEFAULT,
) -> Tuple[Tensor, List[str], List[int]]:
    """Select which points to render based on proximity to view center.

    Points closer to the center (post-Mobius-transform) get priority.
    Higher-rank taxa (domain, phylum) are always included; lower-rank
    taxa (genus, species) are filtered by distance.

    Args:
        all_points: (N, 2) disk coordinates (already Mobius-transformed).
        all_labels: N taxon labels.
        ranks: optional N rank levels (lower rank = higher priority).
        center: view center in disk coords (default origin).
        max_visible: maximum points to return.
        kappa: curvature.

    Returns:
        (visible_coords, visible_labels, original_indices)
    """
    N = all_points.size(0)
    if N <= max_visible:
        return all_points, all_labels, list(range(N))

    if center is None:
        center = torch.zeros(2)

    # Distance from center
    dists = poincare_distance(
        all_points,
        center.unsqueeze(0).expand(N, -1),
        kappa,
    )  # (N,)

    # Priority: rank bonus (domain=6 bonus, species=0) + inverse distance
    if ranks is not None:
        rank_bonus = torch.tensor([(6 - r) * 10.0 for r in ranks])
    else:
        rank_bonus = torch.zeros(N)

    priority = rank_bonus - dists
    _, top_indices = priority.topk(min(max_visible, N))
    top_indices = top_indices.sort().values  # Restore original order

    indices = top_indices.tolist()
    return (
        all_points[top_indices],
        [all_labels[i] for i in indices],
        indices,
    )


# -- Label placement -----------------------------------------------------------

@dataclass
class LabelInfo:
    """Positioned label for rendering."""

    text: str
    x: float
    y: float
    visible: bool = True
    font_size: int = 11


def compute_label_positions(
    points_screen: List[Tuple[float, float]],
    labels: List[str],
    canvas_size: int = 900,
    label_offset: float = 12.0,
    min_separation: float = 14.0,
    max_labels: int = 40,
) -> List[LabelInfo]:
    """Compute label positions with collision avoidance.

    Places labels radially outward from each point, then greedily
    hides colliding labels (lower-priority labels hidden first).

    Args:
        points_screen: (M,) list of (x_px, y_px) screen coordinates.
        labels: M label strings.
        canvas_size: canvas width/height pixels.
        label_offset: pixel offset from point to label.
        min_separation: minimum pixels between label anchors.
        max_labels: cap on visible labels.

    Returns:
        List of LabelInfo with positions and visibility.
    """
    cx, cy = canvas_size / 2, canvas_size / 2
    result: List[LabelInfo] = []
    occupied: List[Tuple[float, float]] = []

    for i, (sx, sy) in enumerate(points_screen):
        if i >= len(labels):
            break

        # Place label radially outward from center
        dx, dy = sx - cx, sy - cy
        dist = math.sqrt(dx * dx + dy * dy) + 1e-10
        offset_x = dx / dist * label_offset
        offset_y = dy / dist * label_offset

        lx = sx + offset_x
        ly = sy + offset_y

        # Check collisions
        visible = len(occupied) < max_labels
        if visible:
            for ox, oy in occupied:
                if math.sqrt((lx - ox) ** 2 + (ly - oy) ** 2) < min_separation:
                    visible = False
                    break

        if visible:
            occupied.append((lx, ly))

        result.append(LabelInfo(text=labels[i], x=lx, y=ly, visible=visible))

    return result
