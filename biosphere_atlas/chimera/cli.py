"""
Command-line interface for atlas-chimera.

Usage:
    # Basic chimera detection (uses BiosphereAtlas API by default)
    atlas-chimera input.fasta -o results.tsv

    # With local model (offline, no API calls)
    atlas-chimera input.fasta --local --model /path/to/checkpoint.pt -o results.tsv

    # Custom API endpoint
    atlas-chimera input.fasta --api-url https://api.biosphereatlas.com --api-key YOUR_KEY

    # Adjust curvature for viral datasets
    atlas-chimera viral_metagenome.fasta --kappa 1.35 -o results.tsv

    # JSON output for pipeline integration
    atlas-chimera input.fasta --format jsonl -o results.jsonl

    # Stream large files without loading all into memory
    atlas-chimera large_metagenome.fasta --stream -o results.tsv
"""

import argparse
import sys
import time
from pathlib import Path

from biosphere_atlas.chimera.detect import detect_chimeras, detect_chimeras_streaming
from biosphere_atlas.chimera.io import write_results_tsv, write_results_json
from biosphere_atlas.core.hyperbolic import KAPPA_DEFAULT


def main():
    parser = argparse.ArgumentParser(
        prog="atlas-chimera",
        description=(
            "Geometry-based chimera detection using hyperbolic embeddings.\n\n"
            "Detects chimeric sequences by identifying geometric anomalies in\n"
            "the Poincare ball coordinate system. A genuine sequence occupies a\n"
            "coherent region; a chimera is pulled toward multiple phylogenetic\n"
            "neighborhoods, producing high variance and bimodal tangent-space\n"
            "structure.\n\n"
            "By default, uses the BiosphereAtlas API for embeddings (no model\n"
            "download required). Use --local for offline inference.\n\n"
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

    # Encoder mode
    mode_group = parser.add_argument_group("encoder mode")
    mode_group.add_argument(
        "--api-url",
        default=None,
        help="BiosphereAtlas API URL (default: https://api.biosphereatlas.com)",
    )
    mode_group.add_argument(
        "--api-key",
        default=None,
        help="API key (or set BIOSPHERE_API_KEY env var)",
    )
    mode_group.add_argument(
        "--local",
        action="store_true",
        help="Use local model instead of API (requires --model)",
    )
    mode_group.add_argument(
        "--model",
        default=None,
        help="Path to local model checkpoint (requires --local)",
    )

    # Detection parameters
    parser.add_argument(
        "--kappa",
        type=float,
        default=KAPPA_DEFAULT,
        help=f"Curvature parameter (default: {KAPPA_DEFAULT})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Chimera score threshold (default: 0.5)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1000,
        help="Sub-sequence window size in bp (default: 1000)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=500,
        help="Window stride in bp (default: 500, i.e. 50%% overlap)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Compute device for local mode (default: cpu)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream results for large files (lower memory)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show progress bar",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="atlas-chimera 0.2.0",
    )

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Determine model path
    model_path = args.model if args.local else None
    if args.local and not args.model:
        print("Error: --local requires --model <path>", file=sys.stderr)
        sys.exit(1)

    # Run detection
    t0 = time.time()
    common_kwargs = dict(
        model_path=model_path,
        kappa=args.kappa,
        threshold=args.threshold,
        window_size=args.window_size,
        stride=args.stride,
        device=args.device,
        api_url=args.api_url,
        api_key=args.api_key,
    )

    if args.stream:
        results = [
            r.to_dict()
            for r in detect_chimeras_streaming(str(input_path), **common_kwargs)
        ]
    else:
        raw_results = detect_chimeras(
            str(input_path), verbose=args.verbose, **common_kwargs,
        )
        results = [r.to_dict() for r in raw_results]

    elapsed = time.time() - t0

    # Write output
    if args.format == "tsv":
        write_results_tsv(results, args.output)
    else:
        write_results_json(results, args.output)

    # Summary
    n_total = len(results)
    n_chimeric = sum(1 for r in results if r["is_chimera"])
    pct = n_chimeric / max(n_total, 1) * 100
    mode = "local" if args.local else "API"
    print(
        f"\natlas-chimera ({mode}): {n_total} sequences in {elapsed:.1f}s "
        f"| {n_chimeric} chimeras ({pct:.1f}%) | kappa={args.kappa}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
