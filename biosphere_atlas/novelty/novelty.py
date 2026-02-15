from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from biosphere_atlas.core.atlas import Atlas
from biosphere_atlas.core.hyperbolic import KAPPA_DEFAULT, poincare_distance
from biosphere_atlas.core.io import read_fasta
from biosphere_atlas.place.reference import Rank, ReferenceDB


@dataclass
class NoveltyResult:
    sequence_id: str
    novelty_score: float
    is_novel: bool
    threshold: float
    nearest_taxon_id: str
    nearest_distance: float
    topk_taxa: List[str]
    topk_distances: List[float]

    def to_dict(self) -> dict:
        return {
            "sequence_id": self.sequence_id,
            "novelty_score": self.novelty_score,
            "is_novel": self.is_novel,
            "threshold": self.threshold,
            "nearest_taxon_id": self.nearest_taxon_id,
            "nearest_distance": self.nearest_distance,
            "topk_taxa": self.topk_taxa,
            "topk_distances": self.topk_distances,
        }


def estimate_threshold_from_reference(
    ref_db: ReferenceDB,
    rank: Rank = Rank.FAMILY,
    quantile: float = 0.99,
    kappa: Optional[float] = None,
) -> float:
    """Estimate novelty threshold from leave-one-out nearest-neighbor distances."""
    ids, emb = ref_db.get_prototypes_at_rank(rank)
    if emb.numel() == 0 or emb.shape[0] < 2:
        return 0.0

    c = float(kappa if kappa is not None else ref_db.kappa)
    dmins = []
    for i in range(emb.shape[0]):
        q = emb[i].unsqueeze(0).expand_as(emb)
        d = poincare_distance(q, emb, kappa=c)
        d[i] = float("inf")
        dmins.append(float(torch.min(d).item()))
    return float(np.quantile(np.array(dmins), quantile))


def score_embedding_novelty(
    embedding: Tensor,
    ref_db: ReferenceDB,
    rank: Rank = Rank.FAMILY,
    threshold: Optional[float] = None,
    kappa: Optional[float] = None,
    top_k: int = 5,
    sequence_id: str = "",
) -> NoveltyResult:
    ids, ref_emb = ref_db.get_prototypes_at_rank(rank)
    if ref_emb.numel() == 0:
        raise ValueError(f"No prototypes found at rank={rank.name}")

    c = float(kappa if kappa is not None else ref_db.kappa)
    emb = embedding.detach().cpu().float()
    q = emb.unsqueeze(0).expand_as(ref_emb)
    d = poincare_distance(q, ref_emb, kappa=c)

    k = min(max(1, top_k), d.shape[0])
    topd, topi = torch.topk(d, k=k, largest=False)

    nearest_distance = float(topd[0].item())
    novelty_score = nearest_distance
    thr = float(threshold if threshold is not None else 0.0)
    is_novel = novelty_score > thr if threshold is not None else False

    topk_taxa = [ids[int(i.item())] for i in topi]
    topk_distances = [float(x.item()) for x in topd]

    return NoveltyResult(
        sequence_id=sequence_id,
        novelty_score=novelty_score,
        is_novel=is_novel,
        threshold=thr,
        nearest_taxon_id=topk_taxa[0],
        nearest_distance=nearest_distance,
        topk_taxa=topk_taxa,
        topk_distances=topk_distances,
    )


def detect_novel_sequences(
    input_fasta: str,
    reference: str | ReferenceDB,
    model_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    rank: Rank = Rank.FAMILY,
    threshold: Optional[float] = None,
    auto_threshold_quantile: float = 0.99,
    device: str = "cpu",
    max_tokens: int = 512,
    top_k: int = 5,
) -> List[NoveltyResult]:
    ref_db = ReferenceDB.load(reference) if isinstance(reference, (str, Path)) else reference

    if threshold is None:
        threshold = estimate_threshold_from_reference(
            ref_db=ref_db,
            rank=rank,
            quantile=auto_threshold_quantile,
            kappa=ref_db.kappa,
        )

    atlas = Atlas.from_checkpoint(
        path=model_path,
        tokenizer_path=tokenizer_path,
        device=device,
        max_tokens=max_tokens,
    ) if model_path else Atlas(device=device, embedding_dim=max(129, ref_db.embedding_dim))

    out: List[NoveltyResult] = []
    for sid, seq in read_fasta(input_fasta):
        emb = atlas.encode(seq)
        out.append(
            score_embedding_novelty(
                emb,
                ref_db=ref_db,
                rank=rank,
                threshold=threshold,
                kappa=atlas.kappa if model_path else ref_db.kappa,
                top_k=top_k,
                sequence_id=sid,
            )
        )
    return out
