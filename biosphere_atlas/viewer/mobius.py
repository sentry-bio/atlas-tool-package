"""
Mobius transformations on the 2D Poincare disk.
=================================================

Navigation in the Poincare disk is implemented via Mobius automorphisms
of the disk.  These are isometries of hyperbolic space — they preserve
geodesic distances, angles, and all geometric structure.

The core operation is ``translate_to_origin(p)``: the Mobius automorphism
that moves a point p to the disk center.  This IS the "zoom" function —
the neighborhood of p expands to fill the disk while the rest of the
tree of life compresses toward the boundary (but remains visible).

All transforms are represented as complex 2x2 matrices in SU(1,1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from torch import Tensor

from biosphere_atlas.core.hyperbolic import KAPPA_DEFAULT, ball_radius


# -- Complex number helpers (real Tensor operations) ---------------------------

def _complex_mul(a: Tensor, b: Tensor) -> Tensor:
    """Multiply two complex numbers stored as (2,) tensors [real, imag]."""
    return torch.tensor([
        a[0] * b[0] - a[1] * b[1],
        a[0] * b[1] + a[1] * b[0],
    ])


def _complex_conj(a: Tensor) -> Tensor:
    """Complex conjugate of a (2,) tensor."""
    return torch.tensor([a[0], -a[1]])


def _complex_div(a: Tensor, b: Tensor) -> Tensor:
    """Divide complex a by complex b."""
    denom = b[0] ** 2 + b[1] ** 2
    return torch.tensor([
        (a[0] * b[0] + a[1] * b[1]) / denom,
        (a[1] * b[0] - a[0] * b[1]) / denom,
    ])


def _complex_abs(a: Tensor) -> float:
    """Absolute value of complex number."""
    return math.sqrt(a[0].item() ** 2 + a[1].item() ** 2)


# -- Mobius2D class ------------------------------------------------------------

@dataclass
class Mobius2D:
    """A Mobius automorphism of the Poincare disk.

    Represented as a map on complex numbers:

        f(z) = (a*z + b) / (conj(b)*z + conj(a))

    where a, b are complex numbers satisfying |a|^2 - |b|^2 = 1 (SU(1,1)).

    Points in the disk are treated as complex numbers: z = x + iy.
    The disk has radius R = 1/sqrt(kappa), so all operations are scaled accordingly.
    """

    a_re: float = 1.0
    a_im: float = 0.0
    b_re: float = 0.0
    b_im: float = 0.0
    kappa: float = KAPPA_DEFAULT

    @property
    def a(self) -> Tensor:
        return torch.tensor([self.a_re, self.a_im])

    @property
    def b(self) -> Tensor:
        return torch.tensor([self.b_re, self.b_im])

    @staticmethod
    def identity(kappa: float = KAPPA_DEFAULT) -> Mobius2D:
        """Return the identity transformation."""
        return Mobius2D(a_re=1.0, a_im=0.0, b_re=0.0, b_im=0.0, kappa=kappa)

    @staticmethod
    def translate_to_origin(
        point: Tensor, kappa: float = KAPPA_DEFAULT
    ) -> Mobius2D:
        """Generate Mobius automorphism that moves ``point`` to the disk origin.

        This is the core navigation operation.  Clicking on a clade in the
        viewer generates this transform, which smoothly brings that region
        to the center while compressing everything else toward the boundary.

        The transform on the unit disk is:

            f(z) = (z - p) / (1 - conj(p)*z)

        We normalize to the ball radius R = 1/sqrt(kappa) by working in
        scaled coordinates.

        Args:
            point: (2,) tensor [x, y] inside the Poincare disk.
            kappa: curvature.

        Returns:
            Mobius2D representing the translation.
        """
        R = ball_radius(kappa)
        # Normalize point to unit disk
        px, py = point[0].item() / R, point[1].item() / R
        p_norm2 = px ** 2 + py ** 2

        if p_norm2 > 0.99:
            # Point too close to boundary — clamp
            scale = 0.99 / math.sqrt(p_norm2)
            px, py = px * scale, py * scale
            p_norm2 = 0.99 ** 2

        # SU(1,1) matrix for translation:
        # a = 1 / sqrt(1 - |p|^2),  b = -p / sqrt(1 - |p|^2)
        inv_scale = 1.0 / math.sqrt(max(1.0 - p_norm2, 1e-10))

        return Mobius2D(
            a_re=inv_scale,
            a_im=0.0,
            b_re=-px * inv_scale,
            b_im=-py * inv_scale,  # b = -p/sqrt(1-|p|^2), where p = px + i*py
            kappa=kappa,
        )

    @staticmethod
    def rotation(angle: float, kappa: float = KAPPA_DEFAULT) -> Mobius2D:
        """Generate rotation around the origin by ``angle`` radians.

        In the Poincare disk, rotation around the origin is just Euclidean
        rotation: f(z) = e^{i*theta} * z.

        In SU(1,1): a = e^{i*theta/2}, b = 0.
        """
        half = angle / 2.0
        return Mobius2D(
            a_re=math.cos(half),
            a_im=math.sin(half),
            b_re=0.0,
            b_im=0.0,
            kappa=kappa,
        )

    def compose(self, other: Mobius2D) -> Mobius2D:
        """Compose: self after other (self(other(z))).

        SU(1,1) matrix multiplication:
            [a1  b1*] . [a2  b2*] = [a1*a2 + b1*conj(b2)  ...]
            [b1  a1*]   [b2  a2*]
        """
        a1, b1 = self.a, self.b
        a2, b2 = other.a, other.b

        new_a = _complex_mul(a1, a2) + _complex_mul(b1, _complex_conj(b2))
        new_b = _complex_mul(a1, b2) + _complex_mul(b1, _complex_conj(a2))

        return Mobius2D(
            a_re=new_a[0].item(), a_im=new_a[1].item(),
            b_re=new_b[0].item(), b_im=new_b[1].item(),
            kappa=self.kappa,
        )

    def inverse(self) -> Mobius2D:
        """Return the inverse transformation.

        For SU(1,1): inverse is conjugate transpose.
            a_inv = conj(a), b_inv = -b
        """
        return Mobius2D(
            a_re=self.a_re, a_im=-self.a_im,
            b_re=-self.b_re, b_im=-self.b_im,
            kappa=self.kappa,
        )

    def interpolate(self, other: Mobius2D, t: float) -> Mobius2D:
        """Smooth interpolation between self and other at parameter t in [0, 1].

        Uses SLERP-like interpolation on the SU(1,1) components.
        At t=0 returns self, at t=1 returns other.

        For smooth animation, we use an ease-in-out cubic curve.
        """
        # Ease-in-out
        s = 3 * t * t - 2 * t * t * t  # smoothstep

        # Linear interpolation in the matrix components
        # (good approximation for nearby transforms; exact for small angles)
        a_re = self.a_re * (1 - s) + other.a_re * s
        a_im = self.a_im * (1 - s) + other.a_im * s
        b_re = self.b_re * (1 - s) + other.b_re * s
        b_im = self.b_im * (1 - s) + other.b_im * s

        # Renormalize to SU(1,1): |a|^2 - |b|^2 = 1
        a_norm2 = a_re ** 2 + a_im ** 2
        b_norm2 = b_re ** 2 + b_im ** 2
        det = a_norm2 - b_norm2
        if det > 1e-10:
            scale = 1.0 / math.sqrt(det)
            a_re *= scale
            a_im *= scale
            b_re *= scale
            b_im *= scale

        return Mobius2D(a_re=a_re, a_im=a_im, b_re=b_re, b_im=b_im, kappa=self.kappa)

    def apply(self, points: Tensor) -> Tensor:
        """Apply this Mobius transform to a batch of 2D points.

        Args:
            points: (N, 2) or (2,) disk coordinates.

        Returns:
            Transformed coordinates, same shape as input.
        """
        squeeze = points.dim() == 1
        if squeeze:
            points = points.unsqueeze(0)

        R = ball_radius(self.kappa)
        # Normalize to unit disk
        z = points / R  # (N, 2) where z[:,0] = real, z[:,1] = imag

        a = self.a  # (2,)
        b = self.b  # (2,)

        # f(z) = (a*z + b) / (conj(b)*z + conj(a))
        # Batch complex multiplication
        z_re, z_im = z[:, 0], z[:, 1]
        a_re, a_im = a[0], a[1]
        b_re, b_im = b[0], b[1]

        # Numerator: a*z + b
        num_re = a_re * z_re - a_im * z_im + b_re
        num_im = a_re * z_im + a_im * z_re + b_im

        # Denominator: conj(b)*z + conj(a)
        den_re = b_re * z_re + b_im * z_im + a_re
        den_im = b_re * z_im - b_im * z_re - a_im

        # Complex division
        denom = den_re ** 2 + den_im ** 2
        denom = denom.clamp_min(1e-15)
        out_re = (num_re * den_re + num_im * den_im) / denom
        out_im = (num_im * den_re - num_re * den_im) / denom

        result = torch.stack([out_re, out_im], dim=-1) * R  # Scale back

        if squeeze:
            result = result.squeeze(0)
        return result

    def to_dict(self) -> dict:
        """Serialize for JSON embedding."""
        return {
            "a_re": self.a_re, "a_im": self.a_im,
            "b_re": self.b_re, "b_im": self.b_im,
            "kappa": self.kappa,
        }


# -- Navigation state ----------------------------------------------------------

@dataclass
class NavigationState:
    """Tracks current and target Mobius transforms for smooth animation."""

    current: Mobius2D = field(default_factory=Mobius2D.identity)
    target: Mobius2D = field(default_factory=Mobius2D.identity)
    animation_t: float = 1.0  # 1.0 = animation complete
    animation_speed: float = 3.0  # t per second

    @property
    def is_animating(self) -> bool:
        return self.animation_t < 1.0

    def navigate_to(self, point: Tensor, kappa: float = KAPPA_DEFAULT) -> None:
        """Start smooth navigation to center the given point."""
        translation = Mobius2D.translate_to_origin(point, kappa)
        self.target = translation.compose(self.current)
        self.animation_t = 0.0

    def rotate(self, angle: float, kappa: float = KAPPA_DEFAULT) -> None:
        """Apply rotation to current transform."""
        rot = Mobius2D.rotation(angle, kappa)
        self.current = rot.compose(self.current)
        self.target = self.current

    def reset(self) -> None:
        """Reset to identity (full-disk view)."""
        self.target = Mobius2D.identity(self.current.kappa)
        self.animation_t = 0.0

    def step(self, dt: float) -> Mobius2D:
        """Advance animation by dt seconds.  Returns the active transform."""
        if self.is_animating:
            self.animation_t = min(1.0, self.animation_t + dt * self.animation_speed)
            return self.current.interpolate(self.target, self.animation_t)
        return self.current

    def finish_animation(self) -> None:
        """Snap to target."""
        self.current = self.target
        self.animation_t = 1.0
