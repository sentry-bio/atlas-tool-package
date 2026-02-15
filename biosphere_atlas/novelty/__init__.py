"""atlas-novelty: novelty detection using geodesic distance to reference prototypes."""

from biosphere_atlas.novelty.novelty import (
    NoveltyResult,
    detect_novel_sequences,
    estimate_threshold_from_reference,
    score_embedding_novelty,
)

__all__ = [
    "NoveltyResult",
    "score_embedding_novelty",
    "estimate_threshold_from_reference",
    "detect_novel_sequences",
]
