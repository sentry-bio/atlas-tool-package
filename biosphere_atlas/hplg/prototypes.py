"""
Dual-bank prototype system for HPLG classification.

Core architectural departure from vanilla prototype networks: maintains TWO
parallel prototype banks per taxon:

  Teacher Bank: Frozen exemplar centroids providing stable reference points.
                These anchor the geometry and prevent catastrophic drift.
  Student Bank: EMA-updated prototypes that adapt to the data distribution
                while constrained by teacher guidance (reanchoring).

This dual-bank design was discovered to be necessary during Phase A/B training
where unsupervised MAGs stabilized at kappa~1.0 before transitioning to
kappa~1.2475. Without teacher anchoring, the transition destabilized prototypes.

Prototype updates are manifold-safe: all movements use geodesic interpolation
rather than Euclidean averaging, preserving the hyperbolic geometry.
"""

import torch
from typing import Dict, Optional
from dataclasses import dataclass

from biosphere_atlas.core.hyperbolic import (
    KAPPA_DEFAULT,
    poincare_distance,
    geodesic_interpolation,
    karcher_mean,
    _clamp_to_ball,
)
from biosphere_atlas.hplg.taxonomy import Rank, DEFAULT_MOMENTUM, DEFAULT_MIN_OBS


@dataclass
class PrototypeState:
    """State for a single taxon prototype."""
    embedding: torch.Tensor     # Current position in Poincare ball
    observation_count: int = 0  # Number of sequences assigned to this prototype
    rank: Rank = Rank.SPECIES
    taxon_id: str = ""


class DualBankPrototypes:
    """
    Dual-bank prototype system with teacher-student architecture.

    The teacher bank provides geometric stability. The student bank
    adapts to incoming data via exponential moving average (EMA)
    updates performed on the manifold.

    Update rules:
    - Momentum per rank (species: 0.99, genus: 0.97, ...)
    - Minimum observation thresholds before trust
    - Parent-child order preservation
    - Distance-based outlier rejection
    - Periodic reanchoring toward teacher (strength 0.1-0.2)
    """

    def __init__(
        self,
        embedding_dim: int,
        kappa: float = KAPPA_DEFAULT,
        momentum: Optional[Dict[Rank, float]] = None,
        min_obs: Optional[Dict[Rank, int]] = None,
        reanchor_strength: float = 0.1,
        outlier_threshold: float = 3.0,
    ):
        self.embedding_dim = embedding_dim
        self.kappa = kappa
        self.momentum = momentum or DEFAULT_MOMENTUM
        self.min_obs = min_obs or DEFAULT_MIN_OBS
        self.reanchor_strength = reanchor_strength
        self.outlier_threshold = outlier_threshold

        # Teacher bank: frozen reference prototypes
        self._teacher: Dict[str, PrototypeState] = {}
        # Student bank: adaptable prototypes
        self._student: Dict[str, PrototypeState] = {}

    def register_prototype(
        self,
        taxon_id: str,
        rank: Rank,
        embedding: torch.Tensor,
    ):
        """
        Register a new prototype from exemplar embedding(s).

        If embedding is 2D (batch of exemplars), computes Karcher mean.
        """
        if embedding.dim() == 2:
            centroid, _ = karcher_mean(embedding, kappa=self.kappa)
        else:
            centroid = embedding.clone()

        centroid = _clamp_to_ball(centroid.unsqueeze(0), self.kappa).squeeze(0)

        teacher_state = PrototypeState(
            embedding=centroid.clone(),
            observation_count=0,
            rank=rank,
            taxon_id=taxon_id,
        )
        student_state = PrototypeState(
            embedding=centroid.clone(),
            observation_count=0,
            rank=rank,
            taxon_id=taxon_id,
        )

        self._teacher[taxon_id] = teacher_state
        self._student[taxon_id] = student_state

    def get_embedding(self, taxon_id: str, bank: str = "student") -> Optional[torch.Tensor]:
        """Get prototype embedding from specified bank."""
        store = self._student if bank == "student" else self._teacher
        state = store.get(taxon_id)
        return state.embedding if state else None

    def get_all_embeddings(
        self,
        taxon_ids: list,
        bank: str = "student",
    ) -> torch.Tensor:
        """Get embeddings for a list of taxon IDs as a batched tensor."""
        store = self._student if bank == "student" else self._teacher
        embeddings = []
        for tid in taxon_ids:
            state = store.get(tid)
            if state is not None:
                embeddings.append(state.embedding)
            else:
                embeddings.append(torch.zeros(self.embedding_dim))
        return torch.stack(embeddings)

    def update(
        self,
        taxon_id: str,
        new_embedding: torch.Tensor,
        force: bool = False,
    ) -> bool:
        """
        EMA update of student prototype using geodesic interpolation.

        Returns True if the update was applied, False if rejected (outlier
        or insufficient trust).
        """
        state = self._student.get(taxon_id)
        if state is None:
            return False

        # Outlier rejection: if new embedding is too far from current prototype
        if not force:
            dist = poincare_distance(
                new_embedding.unsqueeze(0),
                state.embedding.unsqueeze(0),
                self.kappa,
            ).item()

            # Compute expected radius for this taxon
            teacher_state = self._teacher.get(taxon_id)
            if teacher_state and teacher_state.observation_count > 0:
                teacher_dist = poincare_distance(
                    state.embedding.unsqueeze(0),
                    teacher_state.embedding.unsqueeze(0),
                    self.kappa,
                ).item()
                max_allowed = max(teacher_dist * self.outlier_threshold, 0.5)
                if dist > max_allowed:
                    return False  # Reject outlier

        # Get rank-appropriate momentum
        mom = self.momentum.get(state.rank, 0.95)

        # Geodesic interpolation: move (1 - momentum) fraction toward new embedding
        # This is manifold-safe (unlike Euclidean weighted average)
        t = 1.0 - mom  # interpolation fraction
        new_proto = geodesic_interpolation(
            state.embedding, new_embedding, t, self.kappa
        )

        state.embedding = _clamp_to_ball(new_proto.unsqueeze(0), self.kappa).squeeze(0)
        state.observation_count += 1

        return True

    def reanchor(self):
        """
        Pull student prototypes back toward teacher bank.

        Called periodically (e.g., every N batches) to prevent drift.
        Uses geodesic interpolation with reanchor_strength.
        """
        for taxon_id in self._student:
            if taxon_id not in self._teacher:
                continue

            student = self._student[taxon_id]
            teacher = self._teacher[taxon_id]

            reanchored = geodesic_interpolation(
                student.embedding,
                teacher.embedding,
                self.reanchor_strength,
                self.kappa,
            )
            student.embedding = _clamp_to_ball(
                reanchored.unsqueeze(0), self.kappa
            ).squeeze(0)

    def freeze_student_to_teacher(self):
        """
        Snapshot current student bank as new teacher bank.

        Called at the end of a training phase to create new anchors.
        """
        for taxon_id, student_state in self._student.items():
            if taxon_id in self._teacher:
                self._teacher[taxon_id].embedding = student_state.embedding.clone()
                self._teacher[taxon_id].observation_count = student_state.observation_count

    def is_reliable(self, taxon_id: str) -> bool:
        """Check if a prototype has enough observations to be trusted."""
        state = self._student.get(taxon_id)
        if state is None:
            return False
        min_required = self.min_obs.get(state.rank, 10)
        return state.observation_count >= min_required

    @property
    def num_prototypes(self) -> int:
        """Total number of registered prototypes."""
        return len(self._student)

    def taxon_ids(self) -> list:
        """All registered taxon IDs."""
        return list(self._student.keys())

    def state_dict(self) -> dict:
        """Serialize prototype state for checkpointing."""
        return {
            "teacher": {
                tid: {
                    "embedding": s.embedding.cpu(),
                    "count": s.observation_count,
                    "rank": s.rank.value,
                }
                for tid, s in self._teacher.items()
            },
            "student": {
                tid: {
                    "embedding": s.embedding.cpu(),
                    "count": s.observation_count,
                    "rank": s.rank.value,
                }
                for tid, s in self._student.items()
            },
            "kappa": self.kappa,
            "embedding_dim": self.embedding_dim,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "DualBankPrototypes":
        """Load prototype state from checkpoint."""
        bank = cls(
            embedding_dim=state["embedding_dim"],
            kappa=state["kappa"],
        )
        for tid, s in state["teacher"].items():
            bank._teacher[tid] = PrototypeState(
                embedding=s["embedding"],
                observation_count=s["count"],
                rank=Rank(s["rank"]),
                taxon_id=tid,
            )
        for tid, s in state["student"].items():
            bank._student[tid] = PrototypeState(
                embedding=s["embedding"],
                observation_count=s["count"],
                rank=Rank(s["rank"]),
                taxon_id=tid,
            )
        return bank
