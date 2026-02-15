"""
Tests for Poincare ball geometry (atlas-tree copy).
"""

import math

import pytest
import torch

from biosphere_atlas.tree.hyperbolic import (
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
    poincare_distance,
)


KAPPA = KAPPA_DEFAULT
R = ball_radius(KAPPA)


def _ball_point(dim: int = 8, scale: float = 0.3) -> torch.Tensor:
    p = torch.randn(dim) * scale / math.sqrt(KAPPA)
    return _clamp_to_ball(p, KAPPA)


class TestDistance:
    def test_identity(self):
        p = _ball_point()
        d = poincare_distance(p, p, KAPPA)
        assert d.item() < 1e-3

    def test_symmetry(self):
        a, b = _ball_point(), _ball_point()
        d_ab = poincare_distance(a, b, KAPPA)
        d_ba = poincare_distance(b, a, KAPPA)
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


class TestMaps:
    def test_exp_log_invertibility(self):
        x = _ball_point(scale=0.2)
        y = _ball_point(scale=0.2)
        v = log_map(x, y, KAPPA)
        y_rec = exp_map(x, v, KAPPA)
        assert torch.allclose(y, y_rec, atol=1e-3)

    def test_geodesic_endpoints(self):
        a = _ball_point(scale=0.2)
        b = _ball_point(scale=0.2)
        g0 = geodesic_interpolation(a, b, 0.0, KAPPA)
        g1 = geodesic_interpolation(a, b, 1.0, KAPPA)
        assert torch.allclose(g0, a, atol=1e-4)
        assert torch.allclose(g1, b, atol=1e-2)


class TestKarcherMean:
    def test_single_point(self):
        p = _ball_point()
        mean, var = karcher_mean(p.unsqueeze(0), KAPPA)
        assert torch.allclose(mean, p, atol=1e-4)

    def test_stays_in_ball(self):
        points = torch.stack([_ball_point() for _ in range(10)])
        mean, var = karcher_mean(points, KAPPA)
        assert mean.norm().item() < R
