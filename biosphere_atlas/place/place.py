"""
Main placement pipeline.
=========================

Orchestrates: encode → index → place → calibrate → output.

This is the primary API entry point for atlas-place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Generator, List, Optional, Union

import torch
from torch import Tensor

from biosphere_atlas.place.calibrator import PlacementCalibrator, compute_nonconformity
from biosphere_atlas.place.encoder import BiosphereEncoder
from biosphere_atlas.core.hyperbolic import KAPPA_DEFAULT
from biosphere_atlas.place.io import read_fasta, write_results_jplace, write_results_json, write_results_tsv
from biosphere_atlas.place.placer import PlacementEngine, PlacementResult
from biosphere_atlas.place.reference import Rank, ReferenceDB


# ── High-level API ───────────────────────────────────────────────────────────

def place_sequences(
    fasta_path: Union[str, Path],
    reference: ReferenceDB,
    encoder: Optional[BiosphereEncoder] = None,
    calibrator: Optional[PlacementCalibrator] = None,
    output_path: Optional[Union[str, Path]] = None,
    output_format: str = "tsv",
    top_k: int = 5,
    mode: str = "flat",
    kappa: float = KAPPA_DEFAULT,
    device: str = "cpu",
    verbose: bool = False,
) -> List[PlacementResult]:
    """
    Place sequences from a FASTA file against a reference database.

    This is the primary entry point for atlas-place.

    Args:
        fasta_path: path to input FASTA.
        reference: populated ReferenceDB.
        encoder: BiosphereEncoder (or None for default k-mer encoder).
        calibrator: PlacementCalibrator (or None for uncalibrated).
        output_path: optional output file path.
        output_format: 'tsv', 'json', or 'jplace'.
        top_k: number of candidate placements per query.
        mode: 'flat' or 'hierarchical'.
        kappa: curvature constant.
        device: 'cpu' or 'cuda'.
        verbose: print progress.

    Returns:
        List of PlacementResult.
    """
    # Initialize encoder if not provided
    if encoder is None:
        encoder = BiosphereEncoder(
            embedding_dim=reference.embedding_dim,
            kappa=kappa,
            device=device,
        )

    # Initialize placement engine
    engine = PlacementEngine(reference, kappa=kappa, top_k=top_k)

    # Initialize calibrator if not provided
    if calibrator is None:
        calibrator = PlacementCalibrator()

    # Read and process sequences
    results: List[PlacementResult] = []
    count = 0

    for header, sequence in read_fasta(fasta_path):
        if len(sequence) < 10:
            continue

        # Encode
        embedding = encoder.encode(sequence)

        # Place
        result = engine.place(embedding, sequence_id=header, mode=mode)

        # Calibrate
        calibrator.calibrate_placement(result)

        results.append(result)
        count += 1

        if verbose and count % 100 == 0:
            print(f"  Placed {count} sequences...")

    if verbose:
        print(f"  Total: {count} sequences placed.")

    # Write output
    if output_path is not None:
        output_path = Path(output_path)
        if output_format == "json":
            write_results_json(results, output_path)
        elif output_format == "jplace":
            write_results_jplace(results, output_path)
        else:
            write_results_tsv(results, output_path)

        if verbose:
            print(f"  Results written to {output_path}")

    return results


def place_sequences_streaming(
    fasta_path: Union[str, Path],
    reference: ReferenceDB,
    encoder: Optional[BiosphereEncoder] = None,
    calibrator: Optional[PlacementCalibrator] = None,
    top_k: int = 5,
    mode: str = "flat",
    kappa: float = KAPPA_DEFAULT,
    device: str = "cpu",
) -> Generator[PlacementResult, None, None]:
    """
    Streaming variant — yields PlacementResult one at a time.

    Memory-efficient for large FASTA files.
    """
    if encoder is None:
        encoder = BiosphereEncoder(
            embedding_dim=reference.embedding_dim,
            kappa=kappa,
            device=device,
        )

    engine = PlacementEngine(reference, kappa=kappa, top_k=top_k)

    if calibrator is None:
        calibrator = PlacementCalibrator()

    for header, sequence in read_fasta(fasta_path):
        if len(sequence) < 10:
            continue

        embedding = encoder.encode(sequence)
        result = engine.place(embedding, sequence_id=header, mode=mode)
        calibrator.calibrate_placement(result)
        yield result


# ── Batch embedding API (for pre-encoded data) ──────────────────────────────

def place_embeddings(
    embeddings: Tensor,
    reference: ReferenceDB,
    sequence_ids: Optional[List[str]] = None,
    calibrator: Optional[PlacementCalibrator] = None,
    top_k: int = 5,
    mode: str = "flat",
    kappa: float = KAPPA_DEFAULT,
) -> List[PlacementResult]:
    """
    Place pre-computed embeddings against a reference database.

    Use this when embeddings are already available (e.g. from V11 encoder).

    Args:
        embeddings: (B, D) Poincaré ball embeddings.
        reference: populated ReferenceDB.
        sequence_ids: optional identifiers.
        calibrator: PlacementCalibrator (or None).
        top_k: number of candidates.
        mode: 'flat' or 'hierarchical'.
        kappa: curvature.

    Returns:
        List of PlacementResult.
    """
    engine = PlacementEngine(reference, kappa=kappa, top_k=top_k)

    if calibrator is None:
        calibrator = PlacementCalibrator()

    results = engine.place_batch(embeddings, sequence_ids=sequence_ids, mode=mode)
    calibrator.calibrate_batch(results)

    return results


# ── Summary statistics ───────────────────────────────────────────────────────

def placement_summary(results: List[PlacementResult]) -> Dict:
    """
    Compute summary statistics for a set of placements.
    """
    if not results:
        return {"total": 0}

    total = len(results)
    zones = {"accept": 0, "escalate": 0, "fallback": 0}
    distances = []
    margins = []
    confidences = []

    for r in results:
        if r.zone:
            zones[r.zone] = zones.get(r.zone, 0) + 1
        distances.append(r.best_distance)
        if r.margin < float("inf"):
            margins.append(r.margin)
        if r.confidence is not None:
            confidences.append(r.confidence)

    import statistics

    summary = {
        "total": total,
        "zones": zones,
        "accept_rate": zones["accept"] / total if total > 0 else 0,
        "distance_mean": statistics.mean(distances) if distances else 0,
        "distance_median": statistics.median(distances) if distances else 0,
    }

    if margins:
        summary["margin_mean"] = statistics.mean(margins)
        summary["margin_median"] = statistics.median(margins)

    if confidences:
        summary["confidence_mean"] = statistics.mean(confidences)
        summary["confidence_median"] = statistics.median(confidences)

    return summary
