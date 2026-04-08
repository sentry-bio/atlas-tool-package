"""
Main chimera detection pipeline.

This is the primary user-facing API. It orchestrates:
1. FASTA reading
2. Sequence encoding via BiosphereCodec
3. Chimera scoring via hyperbolic geometric anomaly detection
4. BiosphereAtlas coordinate extraction

Usage:
    from biosphere_atlas.chimera import detect_chimeras

    results = detect_chimeras("input.fasta")
    for r in results:
        print(f"{r.sequence_id}: chimera={r.chimera.is_chimera}, "
              f"coordinate=({r.coordinate.r:.3f}, {r.coordinate.theta:.3f})")
"""

import numpy as np
import torch
from dataclasses import dataclass, asdict
from typing import List, Optional, Iterator
from pathlib import Path

from biosphere_atlas.chimera.encoder import BiosphereEncoder
from biosphere_atlas.chimera.chimera import score_chimera, ChimeraScore
from biosphere_atlas.core.hyperbolic import KAPPA_DEFAULT
from biosphere_atlas.core.coordinates import extract_coordinate, BiosphereCoordinate
from biosphere_atlas.chimera.io import read_fasta


def _to_tensor(x) -> torch.Tensor:
    """Convert numpy array or tensor to torch tensor."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return x.float()


@dataclass
class ChimeraResult:
    """
    Complete result for a single sequence.

    Contains both chimera detection metrics and BiosphereAtlas coordinates.
    This is the fundamental output unit — every sequence that passes through
    atlas-chimera gets both a chimera call and a coordinate.
    """
    sequence_id: str
    length: int
    chimera: ChimeraScore
    coordinate: BiosphereCoordinate

    def to_dict(self) -> dict:
        """Serialize to flat dictionary for TSV/JSON output."""
        return {
            "sequence_id": self.sequence_id,
            "length": self.length,
            "chimera_score": self.chimera.score,
            "is_chimera": self.chimera.is_chimera,
            "confidence": self.chimera.confidence,
            "variance": self.chimera.variance,
            "bimodality": self.chimera.bimodality,
            "separation": self.chimera.separation,
            "balance": self.chimera.balance,
            "breakpoint": self.chimera.breakpoint_idx,
            "coordinate": self.coordinate.to_dict(),
        }


def detect_chimeras(
    input_path: str,
    model_path: Optional[str] = None,
    kappa: float = KAPPA_DEFAULT,
    threshold: float = 0.5,
    window_size: int = 1000,
    stride: int = 500,
    device: str = "cpu",
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    verbose: bool = False,
) -> List[ChimeraResult]:
    """
    Detect chimeric sequences in a FASTA file.

    This is the main entry point for atlas-chimera. For each sequence:
    1. Split into overlapping sub-sequences (windows)
    2. Embed each window in the Poincare ball via BiosphereCodec
    3. Detect chimeras via hyperbolic variance and tangent-space bimodality
    4. Extract BiosphereAtlas (r, theta) coordinate

    Args:
        input_path: Path to input FASTA file (.fasta, .fa, .fasta.gz)
        model_path: Path to pre-trained BiosphereCodec weights.
            If None, downloads the default model on first use.
        kappa: Curvature parameter. Default 1.247 for multi-domain life.
            Use lower values (~1.2) for recently-emerged populations,
            higher (~1.6) for deep reservoirs. See Fenn & Fenn 2025.
        threshold: Chimera score threshold for binary calls.
        window_size: Sub-sequence window size in nucleotides.
        stride: Window stride (overlap = window_size - stride).
        device: "cpu" or "cuda" for GPU acceleration.
        batch_size: Number of sequences to process at once.
        verbose: Print progress to stderr.

    Returns:
        List of ChimeraResult objects, one per input sequence.

    Example:
        >>> results = detect_chimeras("metagenome.fasta")
        >>> chimeric = [r for r in results if r.chimera.is_chimera]
        >>> print(f"Found {len(chimeric)}/{len(results)} chimeras")
        >>> for r in results:
        ...     print(f"{r.sequence_id}: ({r.coordinate.r:.3f}, {r.coordinate.theta:.3f})")
    """
    # Initialize encoder
    encoder = BiosphereEncoder(
        model_path=model_path,
        api_url=api_url,
        api_key=api_key,
        device=device,
        kappa=kappa,
        window_size=window_size,
        stride=stride,
    )

    results = []
    sequences = list(read_fasta(input_path))

    if verbose:
        from tqdm import tqdm
        iterator = tqdm(sequences, desc="Detecting chimeras", unit="seq")
    else:
        iterator = sequences

    for header, sequence in iterator:
        sub_emb, full_emb = encoder.encode_subsequences(sequence)

        chimera_score = score_chimera(
            _to_tensor(sub_emb), kappa=kappa, threshold=threshold,
        )
        coordinate = extract_coordinate(_to_tensor(full_emb), kappa=kappa)

        results.append(ChimeraResult(
            sequence_id=header, length=len(sequence),
            chimera=chimera_score, coordinate=coordinate,
        ))

    return results


def detect_chimeras_streaming(
    input_path: str,
    model_path: Optional[str] = None,
    kappa: float = KAPPA_DEFAULT,
    threshold: float = 0.5,
    window_size: int = 1000,
    stride: int = 500,
    device: str = "cpu",
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Iterator[ChimeraResult]:
    """
    Streaming version of detect_chimeras for large files.

    Yields results one at a time instead of collecting all in memory.
    Use this for files larger than available RAM.

    Yields:
        ChimeraResult objects, one per input sequence
    """
    encoder = BiosphereEncoder(
        model_path=model_path,
        api_url=api_url,
        api_key=api_key,
        device=device,
        kappa=kappa,
        window_size=window_size,
        stride=stride,
    )

    for header, sequence in read_fasta(input_path):
        sub_emb, full_emb = encoder.encode_subsequences(sequence)

        chimera_score = score_chimera(
            _to_tensor(sub_emb), kappa=kappa, threshold=threshold,
        )
        coordinate = extract_coordinate(_to_tensor(full_emb), kappa=kappa)

        yield ChimeraResult(
            sequence_id=header, length=len(sequence),
            chimera=chimera_score, coordinate=coordinate,
        )
