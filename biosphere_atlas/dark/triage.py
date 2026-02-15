"""Three-way triage for new genomes: redundant / novel-certain / novel-uncertain.

This is the decision function that drives the crystallization pipeline.
For each new genome embedding, it determines:

  Redundant:        d_geo < epsilon               -> already represented
  Novel-Certain:    d_geo >= epsilon, sigma low    -> new leaf, place geometrically
  Novel-Uncertain:  d_geo >= epsilon, sigma high   -> dark matter, needs training

The thresholds are calibrated from the reference field statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import torch
from torch import Tensor

from biosphere_atlas.core.atlas import Atlas
from biosphere_atlas.core.hyperbolic import poincare_distance
from biosphere_atlas.core.io import read_fasta
from biosphere_atlas.dark.field import UncertaintyField
from biosphere_atlas.place.reference import Rank, ReferenceDB


class TriageCategory(Enum):
    REDUNDANT = "redundant"
    NOVEL_CERTAIN = "novel_certain"
    NOVEL_UNCERTAIN = "novel_uncertain"


@dataclass
class TriageResult:
    sequence_id: str
    category: TriageCategory
    d_geo: float
    sigma_local: float
    nearest_taxon_id: str
    epsilon: float
    sigma_threshold: float

    def to_dict(self) -> dict:
        return {
            "sequence_id": self.sequence_id,
            "category": self.category.value,
            "d_geo": self.d_geo,
            "sigma_local": self.sigma_local,
            "nearest_taxon_id": self.nearest_taxon_id,
            "epsilon": self.epsilon,
            "sigma_threshold": self.sigma_threshold,
        }


def triage_embedding(
    embedding: Tensor,
    field: UncertaintyField,
    epsilon: Optional[float] = None,
    sigma_threshold: Optional[float] = None,
    sequence_id: str = "",
) -> TriageResult:
    """Triage a single embedding against the uncertainty field.

    Args:
        embedding: (D,) Poincare ball embedding.
        field: pre-computed UncertaintyField.
        epsilon: redundancy distance threshold (default: field median sigma).
        sigma_threshold: dark matter sigma threshold (default: field q95).
        sequence_id: identifier for reporting.

    Returns:
        TriageResult with category and metrics.
    """
    stats = field.stats()
    embedding = embedding.detach().to(field._emb.device).float()
    if epsilon is None:
        epsilon = stats.median_sigma
    if sigma_threshold is None:
        sigma_threshold = stats.q95_sigma

    # Nearest prototype distance
    q = embedding.unsqueeze(0).expand_as(field._emb)
    d = poincare_distance(q, field._emb, kappa=field.kappa)
    nearest_idx = int(d.argmin().item())
    d_geo = float(d[nearest_idx].item())
    nearest_id = field._ids[nearest_idx]

    # Local sigma at this point
    sigma_local = field.sigma(embedding)

    # Three-way decision
    if d_geo < epsilon:
        cat = TriageCategory.REDUNDANT
    elif sigma_local <= sigma_threshold:
        cat = TriageCategory.NOVEL_CERTAIN
    else:
        cat = TriageCategory.NOVEL_UNCERTAIN

    return TriageResult(
        sequence_id=sequence_id,
        category=cat,
        d_geo=d_geo,
        sigma_local=sigma_local,
        nearest_taxon_id=nearest_id,
        epsilon=epsilon,
        sigma_threshold=sigma_threshold,
    )


def triage_genomes(
    input_fasta: str,
    reference: Union[str, ReferenceDB],
    model_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    rank: Rank = Rank.FAMILY,
    k: int = 5,
    epsilon: Optional[float] = None,
    sigma_threshold: Optional[float] = None,
    device: str = "cpu",
    max_tokens: int = 512,
) -> List[TriageResult]:
    """Triage a FASTA file of new genomes.

    Args:
        input_fasta: path to FASTA.
        reference: ReferenceDB or path to one.
        model_path: Atlas checkpoint (None = k-mer fallback).
        tokenizer_path: BPE vocab for V13 checkpoints.
        rank: prototype rank level.
        k: k-nearest for sigma estimation.
        epsilon: redundancy threshold.
        sigma_threshold: dark matter threshold.
        device: cpu/cuda.
        max_tokens: max tokenized length.

    Returns:
        List of TriageResult, one per sequence.
    """
    ref_db = ReferenceDB.load(reference) if isinstance(reference, (str, Path)) else reference

    field = UncertaintyField(ref_db, rank=rank, k=k, kappa=ref_db.kappa)

    atlas = Atlas.from_checkpoint(
        path=model_path,
        tokenizer_path=tokenizer_path,
        device=device,
        max_tokens=max_tokens,
    ) if model_path else Atlas(device=device, embedding_dim=max(129, ref_db.embedding_dim))

    results: List[TriageResult] = []
    for sid, seq in read_fasta(input_fasta):
        emb = atlas.encode(seq)
        results.append(
            triage_embedding(
                emb,
                field=field,
                epsilon=epsilon,
                sigma_threshold=sigma_threshold,
                sequence_id=sid,
            )
        )
    return results

