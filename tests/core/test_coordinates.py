"""
Tests for hyperbolic geometry operations.

Key: For kappa=1.247, the Poincare ball has radius R = 1/sqrt(1.247) ~ 0.896.
All test points must satisfy ||x|| < R.
"""

import torch
import pytest
import numpy as np

from biosphere_atlas.chimera.hyperbolic import (
    KAPPA_DEFAULT,
    ball_radius,
    poincare_distance,
    dist_from_origin,
    log_map_0,
    exp_map_0,
    log_map,
    exp_map,
    karcher_mean,
    _clamp_to_ball,
)

R = ball_radius(KAPPA_DEFAULT)  # ~0.896


def _make_ball_points(n, dim=128, scale=0.3, seed=42):
    """Generate random points inside the Poincare ball for kappa=1.247."""
    torch.manual_seed(seed)
    raw = torch.randn(n, dim)
    # Normalize to unit vectors then scale to be well inside the ball
    raw = raw / raw.norm(dim=-1, keepdim=True)
    # Random radii well inside the ball
    radii = torch.rand(n, 1) * scale * R
    return raw * radii


@pytest.fixture
def random_points():
    return _make_ball_points(10, scale=0.5)


class TestBallConstraints:
    def test_ball_radius(self):
        assert abs(R - 0.8955) < 0.001

    def test_clamp_keeps_interior_points(self):
        pts = _make_ball_points(5, scale=0.3)
        clamped = _clamp_to_ball(pts)
        assert torch.allclose(pts, clamped, atol=1e-6)

    def test_clamp_projects_exterior_points(self):
        pts = torch.randn(5, 128) * 5  # Way outside
        clamped = _clamp_to_ball(pts)
        norms = clamped.norm(dim=-1)
        assert (norms < R).all()


class TestPoincareDistance:
    def test_symmetry(self, random_points):
        u, v = random_points[:5], random_points[5:]
        d_uv = poincare_distance(u, v)
        d_vu = poincare_distance(v, u)
        assert torch.allclose(d_uv, d_vu, atol=1e-5)

    def test_identity(self, random_points):
        d = poincare_distance(random_points, random_points)
        assert torch.allclose(d, torch.zeros_like(d), atol=1e-3)

    def test_positivity(self, random_points):
        u, v = random_points[:5], random_points[5:]
        d = poincare_distance(u, v)
        assert (d > 0).all()

    def test_triangle_inequality(self, random_points):
        u = random_points[:3]
        v = random_points[3:6]
        w = random_points[6:9]
        d_uw = poincare_distance(u, w)
        d_uv = poincare_distance(u, v)
        d_vw = poincare_distance(v, w)
        assert (d_uw <= d_uv + d_vw + 1e-4).all()

    def test_origin_distance_consistency(self):
        x = _make_ball_points(5, scale=0.3, seed=0)
        origin = torch.zeros_like(x)
        d_general = poincare_distance(x, origin)
        d_origin = dist_from_origin(x)
        assert torch.allclose(d_general, d_origin, atol=1e-3)


class TestExpLogMaps:
    def test_exp_log_inverse_at_origin(self):
        x = _make_ball_points(5, scale=0.4)
        v = log_map_0(x)
        x_recovered = exp_map_0(v)
        assert torch.allclose(x, x_recovered, atol=1e-3)

    def test_log_exp_inverse_at_origin(self):
        torch.manual_seed(42)
        v = torch.randn(5, 128) * 0.5
        x = exp_map_0(v)
        v_recovered = log_map_0(x)
        assert torch.allclose(v, v_recovered, atol=1e-3)


class TestKarcherMean:
    def test_single_point(self):
        point = _make_ball_points(1, scale=0.3)
        mean, var = karcher_mean(point)
        assert torch.allclose(mean, point.squeeze(0), atol=1e-3)
        assert var < 0.01

    def test_symmetric_points(self):
        x = torch.zeros(2, 128)
        x[0, 0] = 0.3
        x[1, 0] = -0.3
        mean, _ = karcher_mean(x)
        assert mean.norm().item() < 0.1

    def test_variance_increases_with_spread(self):
        tight = _make_ball_points(20, scale=0.05, seed=42)
        spread = _make_ball_points(20, scale=0.5, seed=42)

        _, var_tight = karcher_mean(tight)
        _, var_spread = karcher_mean(spread)
        assert var_spread > var_tight

    def test_convergence(self, random_points):
        mean, var = karcher_mean(random_points, max_iter=200, tol=1e-10)
        assert not torch.isnan(mean).any()
        assert mean.norm().item() < R
        assert 0 <= var < 100


class TestCurvatureConstant:
    def test_default_kappa(self):
        assert abs(KAPPA_DEFAULT - 1.247) < 0.001

    def test_kappa_affects_distance(self):
        pts = _make_ball_points(2, scale=0.2)
        d_low = poincare_distance(pts[0:1], pts[1:2], kappa=1.0)
        d_high = poincare_distance(pts[0:1], pts[1:2], kappa=1.5)
        assert not torch.allclose(d_low, d_high)
