"""
Chimera detection via hyperbolic geometric anomalies.

Core insight: A genuine biological sequence occupies a single coherent
region in hyperbolic space. A chimeric sequence — assembled from fragments
of different organisms — is pulled toward multiple phylogenetic neighborhoods.
This manifests as:

1. High hyperbolic variance around the Karcher mean of sub-sequence embeddings
2. Bimodal structure in the tangent space projection (the sequence "wants to be"
   in two places at once)

The combined chimera score S_chim = alpha * V_H + beta * delta * balance
where V_H is hyperbolic variance, delta is the separation between clusters
in tangent space, and balance measures how evenly the sub-sequences split
between the two modes.

This approach is fundamentally different from reference-based methods:
- UCHIME/ChimeraSlayer: "Does this look like a patchwork of known sequences?"
- atlas-chimera: "Does this sequence occupy a geometrically coherent position?"

The geometric approach works on completely novel organisms with no reference.
It can also detect reassortment events (as validated by the Influenza A
n_eff = 2.2 finding in Fenn & Fenn 2025).
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

from biosphere_atlas.core.hyperbolic import KAPPA_DEFAULT, karcher_mean, log_map, poincare_distance, dist_from_origin


@dataclass
class ChimeraScore:
    """
    Chimera detection result for a single sequence.

    Attributes:
        score: Combined chimera score (0 = clean, higher = more chimeric)
        variance: Hyperbolic variance of sub-sequence embeddings
        bimodality: Bimodality coefficient (0 = unimodal, 1 = perfectly bimodal)
        separation: Distance between cluster centroids in tangent space
        balance: Balance of sub-sequences between clusters (0.5 = even split)
        is_chimera: Boolean call at given threshold
        confidence: Confidence in the call (based on score margin from threshold)
        breakpoint_idx: Estimated chimera breakpoint (index in sub-sequence array)
    """
    score: float
    variance: float
    bimodality: float
    separation: float
    balance: float
    is_chimera: bool
    confidence: float
    breakpoint_idx: Optional[int] = None


def _compute_bimodality(
    tangent_vectors: torch.Tensor,
) -> Tuple[float, float, float, Optional[int]]:
    """
    Detect bimodal structure in tangent space via k-means (k=2).

    A chimeric sequence will have sub-sequence embeddings that cluster
    into two groups in the tangent space at the Karcher mean.
    A genuine sequence will show unimodal, roughly isotropic scatter.

    Returns:
        separation: Distance between cluster centroids
        balance: Min cluster size / max cluster size (0-0.5, 0.5 = even)
        bimodality: Combined bimodality score
        breakpoint: Index where the sequence transitions between clusters
    """
    n = tangent_vectors.shape[0]
    if n < 4:
        return 0.0, 0.0, 0.0, None

    # Simple k-means with k=2 in tangent space
    # Initialize with the two most distant points
    dists = torch.cdist(tangent_vectors, tangent_vectors)
    i, j = divmod(dists.argmax().item(), n)
    centroids = torch.stack([tangent_vectors[i], tangent_vectors[j]])

    labels = torch.zeros(n, dtype=torch.long, device=tangent_vectors.device)

    for _ in range(20):  # k-means iterations
        # Assign
        d0 = (tangent_vectors - centroids[0]).pow(2).sum(dim=-1)
        d1 = (tangent_vectors - centroids[1]).pow(2).sum(dim=-1)
        labels = (d1 < d0).long()

        # Update
        mask0 = labels == 0
        mask1 = labels == 1
        if mask0.sum() == 0 or mask1.sum() == 0:
            return 0.0, 0.0, 0.0, None

        centroids[0] = tangent_vectors[mask0].mean(dim=0)
        centroids[1] = tangent_vectors[mask1].mean(dim=0)

    # Compute metrics
    separation = (centroids[0] - centroids[1]).norm().item()

    count0 = mask0.sum().item()
    count1 = mask1.sum().item()
    balance = min(count0, count1) / max(count0, count1)

    # Within-cluster variance (normalized by between-cluster distance)
    var0 = (tangent_vectors[mask0] - centroids[0]).pow(2).sum(dim=-1).mean().item()
    var1 = (tangent_vectors[mask1] - centroids[1]).pow(2).sum(dim=-1).mean().item()
    within_var = (var0 + var1) / 2
    between_var = (separation ** 2) / 4

    # Bimodality: high when separation is large relative to within-cluster spread
    if within_var < 1e-10:
        bimodality = 0.0
    else:
        bimodality = min(1.0, between_var / (within_var + between_var))

    # Estimate breakpoint: find the transition index in the ordered sub-sequences
    # This assumes sub-sequences are in positional order along the original sequence
    label_changes = (labels[1:] != labels[:-1]).nonzero(as_tuple=True)[0]
    if len(label_changes) > 0:
        # The primary breakpoint is the position with the most sustained label change
        # For a simple chimera (A|B junction), there should be one dominant transition
        breakpoint = label_changes[len(label_changes) // 2].item()
    else:
        breakpoint = None

    return separation, balance, bimodality, breakpoint


def score_chimera(
    sub_embeddings: torch.Tensor,
    kappa: float = KAPPA_DEFAULT,
    alpha: float = 1.0,
    beta: float = 1.0,
    threshold: float = 0.5,
) -> ChimeraScore:
    """
    Compute chimera score for a single sequence from its sub-sequence embeddings.

    The scoring pipeline:
    1. Compute Karcher mean of all sub-sequence embeddings
    2. Measure hyperbolic variance (V_H) — high for chimeras
    3. Project sub-embeddings to tangent space at Karcher mean
    4. Detect bimodality via k-means(k=2) — chimeras split into two clusters
    5. Combine: S = alpha * V_H + beta * separation * balance

    Args:
        sub_embeddings: Embeddings of sub-sequences, shape (n_subs, dim)
            These should be in positional order along the original sequence.
        kappa: Curvature parameter (default: 1.247 for multi-domain life)
        alpha: Weight for hyperbolic variance component
        beta: Weight for bimodality component
        threshold: Score threshold for binary chimera call

    Returns:
        ChimeraScore with all detection metrics
    """
    n_subs = sub_embeddings.shape[0]

    if n_subs < 2:
        return ChimeraScore(
            score=0.0, variance=0.0, bimodality=0.0,
            separation=0.0, balance=0.0, is_chimera=False,
            confidence=1.0, breakpoint_idx=None,
        )

    # Step 1-2: Karcher mean and hyperbolic variance
    mean, variance = karcher_mean(sub_embeddings, kappa=kappa)

    # Step 3: Project to tangent space at Karcher mean
    mean_expanded = mean.unsqueeze(0).expand_as(sub_embeddings)
    tangent_vectors = log_map(sub_embeddings, mean_expanded, kappa)

    # Step 4: Bimodality detection
    separation, balance, bimodality, breakpoint = _compute_bimodality(tangent_vectors)

    # Step 5: Combined score
    # Normalize variance by expected value for genuine sequences
    # (empirically calibrated; will be refined with real data)
    normalized_variance = variance  # raw for now; calibration in v0.2

    score = alpha * normalized_variance + beta * separation * balance

    # Confidence: distance from threshold, scaled
    margin = abs(score - threshold)
    confidence = min(1.0, margin / threshold) if threshold > 0 else 1.0

    return ChimeraScore(
        score=score,
        variance=variance,
        bimodality=bimodality,
        separation=separation,
        balance=balance,
        is_chimera=score > threshold,
        confidence=confidence,
        breakpoint_idx=breakpoint,
    )


def score_batch(
    batch_sub_embeddings: List[torch.Tensor],
    kappa: float = KAPPA_DEFAULT,
    alpha: float = 1.0,
    beta: float = 1.0,
    threshold: float = 0.5,
) -> List[ChimeraScore]:
    """
    Score chimera likelihood for a batch of sequences.

    Args:
        batch_sub_embeddings: List of sub-sequence embedding tensors,
            each shape (n_subs_i, dim)
        kappa: Curvature parameter
        alpha: Variance weight
        beta: Bimodality weight
        threshold: Binary call threshold

    Returns:
        List of ChimeraScore objects, one per input sequence
    """
    return [
        score_chimera(subs, kappa=kappa, alpha=alpha, beta=beta, threshold=threshold)
        for subs in batch_sub_embeddings
    ]
