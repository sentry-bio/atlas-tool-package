"""
Curvature adapter for handling the kappa phase transition.

During training, the BiosphereAtlas exhibits three geometric phases:

  Phase A (Unsupervised MAGs): kappa -> 1.0  (functional clustering)
  Phase B (Supervised RefSeq):  kappa stays ~1.0
  Phase C (Joint training):     kappa transitions 1.0 -> 1.2475 (phylogenetic tree)

The HPLG classifier must handle this transition gracefully:
- During transition, prototypes freeze to teacher bank (prevent drift)
- Acceptance thresholds become conservative
- Fallback thresholds become generous (prefer graceful abstention)
- Student prototype updates pause until kappa stabilizes

The CurvatureAdapter monitors the learned curvature and triggers
appropriate responses when a phase transition is detected.
"""

from dataclasses import dataclass
from typing import Optional, List

from biosphere_atlas.core.hyperbolic import KAPPA_FUNCTIONAL, KAPPA_PHYLOGENETIC


@dataclass
class CurvatureState:
    """Current state of the curvature parameter."""
    kappa: float
    phase: str              # "functional", "transition", "phylogenetic"
    is_stable: bool         # True if kappa has been stable for N steps
    steps_in_phase: int     # How many steps we've been in current phase
    transition_progress: float  # 0.0 = start of transition, 1.0 = complete


class CurvatureAdapter:
    """
    Monitors kappa and adapts HPLG behavior during phase transitions.

    The adapter maintains a running window of kappa values and detects:
    1. Onset of transition (kappa starts moving from 1.0 toward 1.2475)
    2. Active transition (kappa between 1.0 and 1.2475)
    3. Stabilization (kappa settles at target)

    During transition, it provides scaling factors for:
    - Threshold expansion (more conservative acceptance)
    - Prototype update dampening (slower adaptation)
    - Reanchoring strength increase (tighter teacher constraint)
    """

    def __init__(
        self,
        kappa_functional: float = KAPPA_FUNCTIONAL,
        kappa_phylogenetic: float = KAPPA_PHYLOGENETIC,
        stability_window: int = 50,
        stability_threshold: float = 0.005,
    ):
        self.kappa_functional = kappa_functional
        self.kappa_phylogenetic = kappa_phylogenetic
        self.stability_window = stability_window
        self.stability_threshold = stability_threshold

        self._history: List[float] = []
        self._state = CurvatureState(
            kappa=kappa_functional,
            phase="functional",
            is_stable=True,
            steps_in_phase=0,
            transition_progress=0.0,
        )

    def update(self, kappa: float) -> CurvatureState:
        """
        Update with new kappa observation and return current state.

        Args:
            kappa: Current curvature value from the model

        Returns:
            Updated CurvatureState with phase and stability info
        """
        self._history.append(kappa)

        # Determine phase
        kappa_range = self.kappa_phylogenetic - self.kappa_functional
        if kappa_range < 1e-6:
            progress = 1.0
        else:
            progress = (kappa - self.kappa_functional) / kappa_range
        progress = max(0.0, min(1.0, progress))

        if progress < 0.05:
            phase = "functional"
        elif progress > 0.95:
            phase = "phylogenetic"
        else:
            phase = "transition"

        # Check stability
        if len(self._history) >= self.stability_window:
            recent = self._history[-self.stability_window:]
            kappa_std = (
                sum((k - sum(recent) / len(recent)) ** 2 for k in recent) / len(recent)
            ) ** 0.5
            is_stable = kappa_std < self.stability_threshold
        else:
            is_stable = len(self._history) < 10  # assume stable at start

        # Track steps in phase
        if phase != self._state.phase:
            steps_in_phase = 0
        else:
            steps_in_phase = self._state.steps_in_phase + 1

        self._state = CurvatureState(
            kappa=kappa,
            phase=phase,
            is_stable=is_stable,
            steps_in_phase=steps_in_phase,
            transition_progress=progress,
        )

        return self._state

    @property
    def state(self) -> CurvatureState:
        """Current curvature state."""
        return self._state

    def threshold_scale(self) -> float:
        """
        Scaling factor for accept/fallback thresholds during transition.

        During transition: thresholds expand (more conservative).
        Returns multiplier > 1.0 during transition, 1.0 when stable.
        """
        if self._state.phase == "transition" and not self._state.is_stable:
            # Maximum expansion at midpoint of transition
            midpoint_distance = abs(self._state.transition_progress - 0.5) * 2
            return 1.0 + 0.5 * (1.0 - midpoint_distance)  # up to 1.5x
        return 1.0

    def momentum_scale(self) -> float:
        """
        Scaling factor for prototype update momentum during transition.

        During transition: momentum increases (slower adaptation).
        Returns value in [1.0, ~1.0] that pushes momentum toward 1.0.
        """
        if self._state.phase == "transition" and not self._state.is_stable:
            # Dampen updates: push momentum closer to 1.0
            return 0.5  # reduce effective update rate by 50%
        return 1.0

    def reanchor_scale(self) -> float:
        """
        Scaling factor for reanchoring strength during transition.

        During transition: stronger reanchoring (tighter teacher constraint).
        """
        if self._state.phase == "transition" and not self._state.is_stable:
            return 2.0  # double reanchoring strength
        return 1.0
