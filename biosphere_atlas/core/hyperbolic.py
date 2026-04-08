"""
Poincaré ball geometry for BiosphereAtlas ecosystem.
====================================================

All operations use the Poincaré ball model with curvature κ = 1.247
(Fenn & Fenn 2025).  Ball radius R = 1/√κ ≈ 0.8955.

Shared convention across atlas-chimera, atlas-hplg, and atlas-place.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

# ── Universal constants ──────────────────────────────────────────────────────

KAPPA_DEFAULT: float = 1.25
"""BiosphereCoordinate v1.0 datum constant (κ = 5/4, exactly). Fixed by specification."""

BOUNDARY_EPS: float = 1e-5
"""Safety margin inside the ball boundary."""


def ball_radius(kappa: float = KAPPA_DEFAULT) -> float:
    """Radius of the Poincaré ball: R = 1/√κ."""
    return 1.0 / math.sqrt(kappa)


# ── Ball constraint ──────────────────────────────────────────────────────────

def _clamp_to_ball(x: Tensor, kappa: float = KAPPA_DEFAULT) -> Tensor:
    """Project points back inside the Poincaré ball if numerical drift occurs."""
    R = ball_radius(kappa)
    max_norm = R - BOUNDARY_EPS
    norms = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    scale = (max_norm / norms).clamp_max(1.0)
    return x * scale


# ── Distance ─────────────────────────────────────────────────────────────────

def poincare_distance(u: Tensor, v: Tensor, kappa: float = KAPPA_DEFAULT) -> Tensor:
    """
    Geodesic distance on the Poincaré ball.

        d(u, v) = (2/√κ) · arcsinh(√κ · ‖u−v‖ / √((1−κ‖u‖²)(1−κ‖v‖²)))

    Uses the arcsinh formulation (matching V13 training script) which avoids
    Möbius subtraction and is numerically symmetric by construction.
    Returns shape (...,).
    """
    sqk = math.sqrt(kappa)
    diff_norm_sq = ((u - v) ** 2).sum(dim=-1)
    u_norm_sq = (u * u).sum(dim=-1)
    v_norm_sq = (v * v).sum(dim=-1)
    denom = (1.0 - kappa * u_norm_sq) * (1.0 - kappa * v_norm_sq)
    # Use 1e-15 floor on numerator to avoid inflating self-distance near boundary.
    # The 1e-8 denom clamp prevents division by zero for boundary points.
    return (2.0 / sqk) * torch.asinh(
        sqk * diff_norm_sq.clamp_min(1e-15).sqrt() / denom.clamp_min(1e-8).sqrt()
    )


def dist_from_origin(x: Tensor, kappa: float = KAPPA_DEFAULT) -> Tensor:
    """
    Hyperbolic distance from the origin (LUCA).

        r = (2/√κ) · arctanh(√κ · ‖x‖)

    Returns shape (...,).
    """
    norm = x.norm(dim=-1).clamp(min=1e-15, max=ball_radius(kappa) - 1e-7)
    sqk = math.sqrt(kappa)
    return (2.0 / sqk) * torch.atanh(sqk * norm)


# ── Möbius operations ────────────────────────────────────────────────────────

def mobius_addition(u: Tensor, v: Tensor, kappa: float = KAPPA_DEFAULT) -> Tensor:
    """
    Möbius addition in the Poincaré ball with curvature -κ.

        u ⊕ v = ((1 + 2κ<u,v> + κ‖v‖²)u + (1 - κ‖u‖²)v)
                 / (1 + 2κ<u,v> + κ²‖u‖²‖v‖²)
    """
    u2 = (u * u).sum(dim=-1, keepdim=True)
    v2 = (v * v).sum(dim=-1, keepdim=True)
    uv = (u * v).sum(dim=-1, keepdim=True)

    num = (1.0 + 2.0 * kappa * uv + kappa * v2) * u + (1.0 - kappa * u2) * v
    den = 1.0 + 2.0 * kappa * uv + kappa ** 2 * u2 * v2
    result = num / den.clamp_min(1e-15)
    return _clamp_to_ball(result, kappa)


# ── Exponential / logarithmic maps ──────────────────────────────────────────

def log_map_0(y: Tensor, kappa: float = KAPPA_DEFAULT) -> Tensor:
    """Logarithmic map at the origin: log_0(y)."""
    y_norm = y.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    sqk = math.sqrt(kappa)
    coeff = torch.atanh(sqk * y_norm.clamp(max=ball_radius(kappa) - 1e-7)) / (sqk * y_norm)
    return coeff * y


def exp_map_0(v: Tensor, kappa: float = KAPPA_DEFAULT) -> Tensor:
    """Exponential map at the origin: exp_0(v)."""
    v_norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    sqk = math.sqrt(kappa)
    coeff = torch.tanh(sqk * v_norm) / (sqk * v_norm)
    return _clamp_to_ball(coeff * v, kappa)


def log_map(x: Tensor, y: Tensor, kappa: float = KAPPA_DEFAULT) -> Tensor:
    """
    Logarithmic map at x: log_x(y) ∈ T_x B.

        log_x(y) = (2 / (√κ · λ_x)) · arctanh(√κ · ‖(-x)⊕y‖) · ((-x)⊕y) / ‖(-x)⊕y‖
                 = (2 / λ_x) · log_0((-x) ⊕ y)

    where 2/λ_x = 1 - κ‖x‖².
    """
    neg_x_plus_y = mobius_addition(-x, y, kappa=kappa)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    inv_scale = (1.0 - kappa * x2).clamp_min(1e-15)
    return inv_scale * log_map_0(neg_x_plus_y, kappa)


def exp_map(x: Tensor, v: Tensor, kappa: float = KAPPA_DEFAULT) -> Tensor:
    """
    Exponential map at x: exp_x(v) ∈ B.

        exp_x(v) = x ⊕ tanh(√κ · λ_x · ‖v‖ / 2) · v / (√κ · ‖v‖)
                 = x ⊕ exp_0(λ_x · v / 2)
    """
    lam = _lambda_x(x, kappa)
    scaled_v = v * lam / 2.0
    return mobius_addition(x, exp_map_0(scaled_v, kappa), kappa=kappa)


def _lambda_x(x: Tensor, kappa: float = KAPPA_DEFAULT) -> Tensor:
    """Conformal factor: λ_x = 2 / (1 - κ‖x‖²)."""
    x2 = (x * x).sum(dim=-1, keepdim=True)
    return 2.0 / (1.0 - kappa * x2).clamp_min(1e-15)


# ── Geodesic interpolation ──────────────────────────────────────────────────

def geodesic_interpolation(
    x: Tensor, y: Tensor, t: float, kappa: float = KAPPA_DEFAULT
) -> Tensor:
    """
    Manifold-safe interpolation along the geodesic from x to y.

        γ(t) = exp_x(t · log_x(y))

    t=0 → x,  t=1 → y.
    """
    v = log_map(x, y, kappa=kappa)
    return exp_map(x, t * v, kappa=kappa)


# ── Karcher (Fréchet) mean ───────────────────────────────────────────────────

def karcher_mean(
    points: Tensor,
    kappa: float = KAPPA_DEFAULT,
    max_iter: int = 50,
    tol: float = 1e-8,
    weights: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Riemannian center of mass on the Poincaré ball.

    Args:
        points: (N, D) embeddings inside the ball.
        kappa: curvature.
        max_iter: iteration cap.
        tol: convergence threshold on mean tangent norm.
        weights: optional (N,) importance weights.

    Returns:
        (mean, variance) where variance = weighted mean squared geodesic distance.
    """
    if weights is None:
        weights = torch.ones(points.size(0), device=points.device)
    weights = weights / weights.sum()

    # Initialize at weighted Euclidean centroid projected into ball
    mu = _clamp_to_ball((weights.unsqueeze(-1) * points).sum(dim=0), kappa)

    for _ in range(max_iter):
        # Compute tangent vectors from mu to each point
        tangents = log_map(mu.unsqueeze(0).expand_as(points), points, kappa=kappa)
        # Weighted mean tangent
        mean_tangent = (weights.unsqueeze(-1) * tangents).sum(dim=0)

        if mean_tangent.norm() < tol:
            break

        # Step along mean tangent
        mu = exp_map(mu, mean_tangent, kappa=kappa)
        mu = _clamp_to_ball(mu, kappa)

    # Variance = weighted mean squared geodesic distance
    dists = poincare_distance(mu.unsqueeze(0).expand_as(points), points, kappa=kappa)
    variance = (weights * dists ** 2).sum()

    return mu, variance
