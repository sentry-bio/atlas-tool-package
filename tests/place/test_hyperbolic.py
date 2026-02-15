"""
Tests for Poincaré ball geometry.

Uses moderate interior points (scale=0.3) for precision-sensitive tests.
Points near the ball boundary lose numerical accuracy — this is inherent
to the Poincaré model and matches V13 training behavior.
"""

import math

import pytest
import torch

from biosphere_atlas.place.hyperbolic import (
    KAPPA_DEFAULT,
    _clamp_to_ball,
    ball_radius,
    dist_from_origin,
    exp_map,
    exp_map_0,
    geodesic_interpolation,
    karcher_mean,
    log_map,
    log_map_0,
    mobius_addition,
    poincare_distance,
)


KAPPA = KAPPA_DEFAULT
R = ball_radius(KAPPA)


def _ball_point(dim: int = 8, scale: float = 0.3) -> torch.Tensor:
    """Generate a random point inside the Poincaré ball.

    scale=0.3 keeps points in the interior where all operations are
    numerically well-conditioned.  Use scale=0.6+ only for boundary tests.
    """
    p = torch.randn(dim) * scale / math.sqrt(KAPPA)
    return _clamp_to_ball(p, KAPPA)


class TestBallConstraints:
    def test_ball_radius(self):
        assert abs(R - 1.0 / math.sqrt(KAPPA)) < 1e-6

    def test_clamp_keeps_interior(self):
        p = torch.randn(8) * 10  # Way outside
        clamped = _clamp_to_ball(p, KAPPA)
        assert clamped.norm().item() < R

    def test_clamp_preserves_interior(self):
        p = _ball_point()
        clamped = _clamp_to_ball(p, KAPPA)
        assert torch.allclose(p, clamped, atol=1e-6)


class TestDistance:
    def test_identity(self):
        p = _ball_point()
        d = poincare_distance(p, p, KAPPA)
        assert d.item() < 1e-3

    def test_symmetry(self):
        a, b = _ball_point(), _ball_point()
        d_ab = poincare_distance(a, b, KAPPA)
        d_ba = poincare_distance(b, a, KAPPA)
        # asinh form is symmetric by construction
        assert abs(d_ab.item() - d_ba.item()) < 1e-5

    def test_triangle_inequality(self):
        a, b, c = _ball_point(), _ball_point(), _ball_point()
        d_ab = poincare_distance(a, b, KAPPA).item()
        d_bc = poincare_distance(b, c, KAPPA).item()
        d_ac = poincare_distance(a, c, KAPPA).item()
        assert d_ac <= d_ab + d_bc + 1e-4

    def test_non_negative(self):
        a, b = _ball_point(), _ball_point()
        d = poincare_distance(a, b, KAPPA)
        assert d.item() >= -1e-5

    def test_batch(self):
        A = torch.stack([_ball_point() for _ in range(4)])
        B = torch.stack([_ball_point() for _ in range(4)])
        dists = poincare_distance(A, B, KAPPA)
        assert dists.shape == (4,)
        assert (dists >= -1e-5).all()

    def test_dist_from_origin(self):
        origin = torch.zeros(8)
        p = _ball_point()
        d1 = dist_from_origin(p, KAPPA)
        d2 = poincare_distance(origin, p, KAPPA)
        # atanh vs asinh forms agree within ~0.01 for interior points
        assert abs(d1.item() - d2.item()) < 0.01


class TestMaps:
    def test_exp_log_invertibility_at_origin(self):
        v = torch.randn(8) * 0.3
        y = exp_map_0(v, KAPPA)
        v_rec = log_map_0(y, KAPPA)
        assert torch.allclose(v, v_rec, atol=1e-4)

    def test_exp_log_invertibility_general(self):
        x = _ball_point(scale=0.2)
        y = _ball_point(scale=0.2)
        v = log_map(x, y, KAPPA)
        y_rec = exp_map(x, v, KAPPA)
        assert torch.allclose(y, y_rec, atol=1e-3)

    def test_exp_stays_in_ball(self):
        x = _ball_point()
        v = torch.randn(8) * 0.3
        y = exp_map(x, v, KAPPA)
        assert y.norm().item() < R


class TestGeodesic:
    def test_endpoints(self):
        a = _ball_point(scale=0.2)
        b = _ball_point(scale=0.2)
        g0 = geodesic_interpolation(a, b, 0.0, KAPPA)
        g1 = geodesic_interpolation(a, b, 1.0, KAPPA)
        assert torch.allclose(g0, a, atol=1e-4)
        assert torch.allclose(g1, b, atol=1e-2)

    def test_midpoint_inside_ball(self):
        a = _ball_point()
        b = _ball_point()
        mid = geodesic_interpolation(a, b, 0.5, KAPPA)
        assert mid.norm().item() < R

    def test_stays_in_ball(self):
        a = _ball_point()
        b = _ball_point()
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            g = geodesic_interpolation(a, b, t, KAPPA)
            assert g.norm().item() < R


class TestKarcherMean:
    def test_single_point(self):
        p = _ball_point()
        mean, var = karcher_mean(p.unsqueeze(0), KAPPA)
        assert torch.allclose(mean, p, atol=1e-4)
        assert var.item() < 1e-3

    def test_stays_in_ball(self):
        points = torch.stack([_ball_point() for _ in range(10)])
        mean, var = karcher_mean(points, KAPPA)
        assert mean.norm().item() < R

    def test_variance_non_negative(self):
        points = torch.stack([_ball_point() for _ in range(10)])
        _, var = karcher_mean(points, KAPPA)
        assert var.item() >= -1e-8

    def test_converges_for_cluster(self):
        # Points clustered near a center — use small scale for precision
        center = _ball_point(scale=0.15)
        points = torch.stack([
            _clamp_to_ball(center + torch.randn(8) * 0.01, KAPPA)
            for _ in range(20)
        ])
        mean, var = karcher_mean(points, KAPPA)
        d = poincare_distance(mean, center, KAPPA)
        assert d.item() < 0.2  # Mean should be near center
