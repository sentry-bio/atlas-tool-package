"""
Command-line interface for atlas-chimera.

Usage:
    # Basic chimera detection
    atlas-chimera input.fasta -o results.tsv

    # With GPU acceleration
    atlas-chimera input.fasta -o results.tsv --device cuda

    # Adjust curvature for viral datasets
    atlas-chimera viral_metagenome.fasta --kappa 1.35 -o results.tsv

    # JSON output for pipeline integration
    atlas-chimera input.fasta --format jsonl -o results.jsonl

    # Stream large files without loading all into memory
    atlas-chimera large_metagenome.fasta --stream -o results.tsv

    # Coordinates only (skip chimera detection, just place sequences)
    atlas-chimera input.fasta --coordinates-only -o coordinates.tsv
"""

import argparse
import sys
import time
from pathlib import Path

from biosphere_atlas.chimera.detect import detect_chimeras, detect_chimeras_streaming
from biosphere_atlas.core.io import write_results_tsv, write_results_json
from biosphere_atlas.core.hyperbolic import KAPPA_DEFAULT


def main():
    parser = argparse.ArgumentParser(
        prog="atlas-chimera",
        description=(
            "Geometry-based chimera detection using hyperbolic embeddings.\n\n"
            "Detects chimeric sequences by identifying geometric anomalies in\n"
            "the BiosphereAtlas coordinate system. Every sequence receives both\n"
            "a chimera score and a universal (r, theta) coordinate.\n\n"
            "Reference: Fenn & Fenn (2025). Evolution as Active Geometry."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input",
        help="Input FASTA file (.fasta, .fa, .fasta.gz)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--format",
        choices=["tsv", "jsonl"],
        default="tsv",
        help="Output format (default: tsv)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to BiosphereCodec model weights",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=KAPPA_DEFAULT,
        help=(
            f"Curvature parameter (default: {KAPPA_DEFAULT}). "
            "Use ~1.2 for recent outbreaks, ~1.6 for deep reservoirs."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Chimera score threshold for binary calls (default: 0.5)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1000,
        help="Sub-sequence window size in nucleotides (default: 1000)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=500,
        help="Window stride in nucleotides (default: 500)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Compute device (default: cpu)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream results for large files (lower memory usage)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show progress bar",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="atlas-chimera 0.1.0",
    )

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Run detection
    t0 = time.time()

    if args.stream:
        results = []
        for result in detect_chimeras_streaming(
            str(input_path),
            model_path=args.model,
            kappa=args.kappa,
            threshold=args.threshold,
            window_size=args.window_size,
            stride=args.stride,
            device=args.device,
        ):
            results.append(result.to_dict())
    else:
        raw_results = detect_chimeras(
            str(input_path),
            model_path=args.model,
            kappa=args.kappa,
            threshold=args.threshold,
            window_size=args.window_size,
            stride=args.stride,
            device=args.device,
            verbose=args.verbose,
        )
        results = [r.to_dict() for r in raw_results]

    elapsed = time.time() - t0

    # Write output
    if args.format == "tsv":
        write_results_tsv(results, args.output)
    else:
        write_results_json(results, args.output)

    # Summary to stderr
    n_total = len(results)
    n_chimeric = sum(1 for r in results if r["is_chimera"])
    print(
        f"\natlas-chimera: {n_total} sequences processed in {elapsed:.1f}s "
        f"({n_chimeric} chimeras detected, {n_chimeric/max(n_total,1)*100:.1f}%)",
        file=sys.stderr,
    )
    print(
        f"All sequences assigned BiosphereAtlas coordinates (kappa={args.kappa})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
