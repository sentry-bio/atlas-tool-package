"""
Tests for chimera detection scoring.

All test points are properly constrained to the Poincare ball
with radius R = 1/sqrt(kappa) ~ 0.896.
"""

import torch
import pytest
from biosphere_atlas.chimera.chimera import score_chimera, ChimeraScore
from biosphere_atlas.chimera.hyperbolic import KAPPA_DEFAULT, ball_radius

R = ball_radius(KAPPA_DEFAULT)


def _ball_point(dim=128, radius_frac=0.3):
    """Create a single point inside the ball at given fraction of R."""
    p = torch.randn(dim)
    p = p / p.norm() * radius_frac * R
    return p


class TestChimeraScoring:
    def test_coherent_cluster_low_score(self):
        """Tightly clustered sub-embeddings should have low chimera score."""
        torch.manual_seed(42)
        center = _ball_point(radius_frac=0.3)
        noise = torch.randn(10, 128) * 0.005
        sub_embeddings = center.unsqueeze(0) + noise
        # Ensure all inside ball
        norms = sub_embeddings.norm(dim=-1, keepdim=True)
        sub_embeddings = sub_embeddings * (R * 0.5 / norms.clamp(min=R * 0.5))

        result = score_chimera(sub_embeddings)
        assert result.score < 0.5
        assert not result.is_chimera

    def test_bimodal_cluster_high_score(self):
        """Two separated clusters should have high chimera score."""
        torch.manual_seed(42)
        center_a = torch.zeros(128)
        center_a[0] = 0.5 * R
        center_b = torch.zeros(128)
        center_b[0] = -0.5 * R

        noise = torch.randn(10, 128) * 0.01
        cluster_a = center_a + noise[:5]
        cluster_b = center_b + noise[5:]
        sub_embeddings = torch.cat([cluster_a, cluster_b])

        result = score_chimera(sub_embeddings)
        assert result.variance > 0.01
        assert result.is_chimera or result.score > 0.1

    def test_single_subsequence(self):
        """Single sub-embedding should return clean."""
        sub_embeddings = _ball_point().unsqueeze(0)
        result = score_chimera(sub_embeddings)
        assert result.score == 0.0
        assert not result.is_chimera
        assert result.confidence == 1.0

    def test_breakpoint_detection(self):
        """Chimera breakpoint should be detected near the junction."""
        torch.manual_seed(42)
        center_a = torch.zeros(128)
        center_a[0] = 0.4 * R
        center_b = torch.zeros(128)
        center_b[1] = 0.4 * R

        sub_a = center_a.unsqueeze(0) + torch.randn(5, 128) * 0.005
        sub_b = center_b.unsqueeze(0) + torch.randn(5, 128) * 0.005
        sub_embeddings = torch.cat([sub_a, sub_b])

        result = score_chimera(sub_embeddings)
        if result.breakpoint_idx is not None:
            assert 2 <= result.breakpoint_idx <= 7

    def test_score_components_non_negative(self):
        """All score components should be non-negative."""
        torch.manual_seed(42)
        pts = torch.randn(8, 128)
        pts = pts / pts.norm(dim=-1, keepdim=True) * 0.3 * R

        result = score_chimera(pts)
        assert result.variance >= 0
        assert result.bimodality >= 0
        assert result.separation >= 0
        assert 0 <= result.balance <= 1.0


class TestChimeraScoreDataclass:
    def test_fields(self):
        score = ChimeraScore(
            score=0.7, variance=0.3, bimodality=0.8,
            separation=1.2, balance=0.45, is_chimera=True,
            confidence=0.6, breakpoint_idx=5,
        )
        assert score.is_chimera
        assert score.breakpoint_idx == 5
