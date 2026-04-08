"""
BiosphereCoordinate v1.0 Specification-Compliant Extractor
==========================================================

This module implements the canonical coordinate extraction following the
BiosphereCoordinate v1.0 Specification.md.

Unlike model-specific coordinate extraction, this is **infrastructure** - it
defines coordinates independently of any particular model implementation, similar
to how WGS84 defines geodetic coordinates independently of GPS receivers.

Key Features:
- Fixed curvature κ = 1.25 (specification constant)
- Tree topology-based reference frame (domain centroids)
- E. coli K-12 MG1655 anchor at θ = 0°
- Centered tangent-space projection to isolate angular structure
- Versioned datum with transformation functions

Usage:
    from biosphere_atlas.core.coordinates_v1 import BiosphereCoordinateV1

    extractor = BiosphereCoordinateV1()
    coord = extractor.extract(embedding)

    print(f"BiosphereCoordinate: r={coord.r:.3f} nats, θ={coord.theta:.1f}°")
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor

from biosphere_atlas.core.hyperbolic import log_map_0


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

KAPPA_V1_0 = 1.25  # Fixed by specification v1.0
R_BALL_V1_0 = 2.0 / math.sqrt(5)  # Poincaré ball radius ≈ 0.8944


# ═══════════════════════════════════════════════════════════════════════
# COORDINATE DATACLASS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BiosphereCoordinateV1Data:
    """
    A position in the BiosphereCoordinate v1.0 coordinate system.

    Attributes:
        r: Radial coordinate (nats) - hyperbolic distance from LUCA origin
        theta: Angular coordinate (degrees) [0°, 360°) - phylogenetic direction
        r_euclidean: Euclidean norm (for compatibility and visualization)
        sigma_r: Radial uncertainty estimate (nats)
        sigma_theta: Angular uncertainty estimate (degrees)
        kappa: Specification curvature value (always 1.25 for v1.0)
        embedding_dim: Dimension of input embedding
        specification_version: Version string (e.g., "1.0")
    """
    r: float
    theta: float
    r_euclidean: float
    sigma_r: Optional[float] = None
    sigma_theta: Optional[float] = None
    kappa: float = KAPPA_V1_0
    embedding_dim: int = 129
    specification_version: str = "1.0"

    def to_dict(self):
        """Convert to dictionary for serialization.

        Includes 'theta_degrees' for backward compatibility with the chimera
        TSV writer (core/io.py) which reads r['coordinate']['theta_degrees'].
        In v1.0 theta is already in degrees, so theta_degrees == theta.
        """
        return {
            'r': round(self.r, 6),
            'theta': round(self.theta, 6),
            'theta_degrees': round(self.theta, 2),
            'r_euclidean': round(self.r_euclidean, 6),
            'sigma_r': round(self.sigma_r, 6) if self.sigma_r is not None else None,
            'sigma_theta': round(self.sigma_theta, 6) if self.sigma_theta is not None else None,
            'kappa': self.kappa,
            'embedding_dim': self.embedding_dim,
            'specification_version': self.specification_version,
        }

    def __repr__(self):
        sr = f"{self.sigma_r:.3f}" if self.sigma_r is not None else "N/A"
        st = f"{self.sigma_theta:.1f}" if self.sigma_theta is not None else "N/A"
        return (f"BiosphereCoordinate(r={self.r:.3f} nats, θ={self.theta:.1f}°, "
                f"σ_r={sr}, σ_θ={st}°)")


# ═══════════════════════════════════════════════════════════════════════
# COORDINATE EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════

class BiosphereCoordinateV1:
    """
    Extracts BiosphereCoordinate v1.0 compliant (r, θ) from model embeddings.

    The coordinate system is defined by:
    1. Fixed curvature κ = 1.25 (theoretical attractor)
    2. Centered tangent-space projection to isolate angular structure
    3. Reference plane from domain centroids (Gram-Schmidt on deviations)
    4. E. coli K-12 MG1655 anchor at θ = 0°
    5. Hyperbolic distance formula for radial coordinate

    Extraction protocol for a ball point z:
        1. v = log_0(z)                    # tangent vector at origin
        2. v_c = v - v_grand               # center (remove radial bias)
        3. z_proj = P @ v_c                # project to 2D reference plane
        4. θ_raw = atan2(y, x)             # raw angle
        5. θ = (θ_raw - θ_Ecoli) mod 360   # E. coli anchored at 0°
        6. r = (2/√κ) arctanh(√κ ||z||)    # hyperbolic distance
    """

    def __init__(self, reference_file: Optional[Union[str, Path]] = None):
        if reference_file is None:
            reference_file = (
                Path(__file__).parent / "data" /
                "BiosphereCoordinate_v1.0_Reference.json"
            )

        self.reference_file = Path(reference_file)

        if not self.reference_file.exists():
            raise FileNotFoundError(
                f"Reference realization not found at {self.reference_file}. "
                f"Generate it using compute_reference_realization.py"
            )

        with open(self.reference_file) as f:
            self.reference = json.load(f)

        self.kappa = self.reference.get('kappa', KAPPA_V1_0)
        self.version = self.reference.get('version', '1.0')
        self.embedding_dim = self.reference.get('embedding_dim', 129)

        if self.kappa != KAPPA_V1_0:
            raise ValueError(
                f"Reference kappa {self.kappa} does not match specification "
                f"constant {KAPPA_V1_0}. Reference file may be corrupted."
            )

        # Load projection matrix P (2×d)
        self.P = torch.tensor(
            self.reference['projection_matrix_P'],
            dtype=torch.float32
        )
        if self.P.shape[0] != 2:
            raise ValueError(f"Projection matrix must be 2×d, got {self.P.shape}")

        # Load tangent-space grand centroid for centering
        self.v_grand = torch.tensor(
            self.reference['tangent_space_grand_centroid'],
            dtype=torch.float32
        )

        # Load E. coli anchor offset
        ecoli_anchor = self.reference['reference_anchors'].get('E_coli_K12_MG1655')
        if ecoli_anchor is None:
            raise ValueError("E. coli anchor not found in reference")
        self.theta_ecoli = ecoli_anchor.get('theta_raw', 0.0)

        # Load domain centroids for uncertainty estimation
        self.domain_centroids = {}
        for domain, data in self.reference.get('domain_centroids', {}).items():
            self.domain_centroids[domain] = {
                'r': data['r'],
                'theta': data['theta'],
                'embedding': torch.tensor(data['embedding'], dtype=torch.float32),
            }

    def extract(self, embedding: Union[Tensor, np.ndarray]) -> BiosphereCoordinateV1Data:
        """
        Extract certified BiosphereCoordinate v1.0 from a Poincaré ball embedding.

        Args:
            embedding: (d,) embedding vector (||embedding|| < R_ball ≈ 0.8944)

        Returns:
            BiosphereCoordinateV1Data with (r, θ, σ_r, σ_θ)
        """
        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding).float()

        if embedding.dim() != 1:
            raise ValueError(f"Embedding must be 1D, got shape {embedding.shape}")

        if embedding.shape[0] != self.P.shape[1]:
            raise ValueError(
                f"Embedding dimension {embedding.shape[0]} does not match "
                f"projection matrix {self.P.shape[1]}"
            )

        norm = embedding.norm().item()
        if norm >= R_BALL_V1_0:
            raise ValueError(
                f"Embedding norm {norm:.6f} exceeds Poincaré ball radius "
                f"{R_BALL_V1_0:.6f}."
            )

        r = self._extract_radial(embedding)
        theta = self._extract_angular(embedding)
        sigma_r, sigma_theta = self._estimate_uncertainty(embedding, r, theta)

        return BiosphereCoordinateV1Data(
            r=r,
            theta=theta,
            r_euclidean=norm,
            sigma_r=sigma_r,
            sigma_theta=sigma_theta,
            kappa=self.kappa,
            embedding_dim=embedding.shape[0],
            specification_version=self.version,
        )

    def _extract_radial(self, embedding: Tensor) -> float:
        """r = (2/√κ) arctanh(√κ · ||z||)"""
        norm = embedding.norm().item()
        sqk = math.sqrt(self.kappa)
        return (2.0 / sqk) * math.atanh(sqk * norm)

    def _extract_angular(self, embedding: Tensor) -> float:
        """
        θ extraction via centered tangent-space projection:
          1. v = log_0(z)
          2. v_c = v - v_grand
          3. z_proj = P @ v_c
          4. θ = (atan2(y, x) - θ_Ecoli) mod 360
        """
        v = log_map_0(embedding.unsqueeze(0), kappa=self.kappa).squeeze(0)
        v_centered = v - self.v_grand
        z_proj = self.P @ v_centered
        x, y = z_proj[0].item(), z_proj[1].item()
        theta_raw = math.degrees(math.atan2(y, x))
        theta = (theta_raw - self.theta_ecoli) % 360.0
        # Guard against floating-point edge: values very close to 360 → 0.0
        # (JSON serialization of theta_raw introduces ~1e-8 rounding)
        if theta > 360.0 - 1e-6:
            theta = 0.0
        return theta

    def _estimate_uncertainty(
        self, embedding: Tensor, r: float, theta: float,
    ) -> tuple[float, float]:
        """Estimate coordinate uncertainties from distance to domain centroids."""
        if not self.domain_centroids:
            return 0.05, 5.0

        distances_r = []
        distances_theta = []
        for data in self.domain_centroids.values():
            distances_r.append(abs(r - data['r']))
            dtheta = abs(theta - data['theta'])
            distances_theta.append(min(dtheta, 360.0 - dtheta))

        min_dr = min(distances_r)
        min_dtheta = min(distances_theta)

        sigma_r = max(0.01, 0.1 * min_dr + 0.02)
        sigma_theta = max(1.0, 0.5 * min_dtheta + 2.0)

        return sigma_r, sigma_theta

    def validate_coordinate(
        self, coord: BiosphereCoordinateV1Data,
    ) -> dict[str, bool]:
        """Validate coordinate against specification constraints."""
        results = {
            'r_non_negative': coord.r >= 0,
            'r_finite': math.isfinite(coord.r),
            'theta_in_range': 0 <= coord.theta < 360,
            'kappa_matches_spec': abs(coord.kappa - KAPPA_V1_0) < 1e-6,
            'version_v1_0': coord.specification_version == "1.0",
        }
        results['valid'] = all(results.values())
        return results


# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def extract_coordinate_v1(
    embedding: Union[Tensor, np.ndarray],
    reference_file: Optional[Union[str, Path]] = None,
) -> BiosphereCoordinateV1Data:
    """
    Convenience function for one-off coordinate extraction.

    Args:
        embedding: (d,) Poincaré ball embedding vector
        reference_file: Optional path to reference realization

    Returns:
        BiosphereCoordinateV1Data
    """
    extractor = BiosphereCoordinateV1(reference_file=reference_file)
    return extractor.extract(embedding)
