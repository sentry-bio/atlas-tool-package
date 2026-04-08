"""
Command-line interface for atlas-place.
========================================

Usage:
    atlas-place input.fasta -r reference.db -o placements.tsv
    atlas-place input.fasta -r reference.db --format jplace -o placements.jplace
    atlas-place info -r reference.db
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atlas-place",
        description=(
            "Phylogenetic placement via BiosphereAtlas hyperbolic coordinates. "
            "Drop-in replacement for pplacer / GTDB-Tk classify."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── place (default) ──────────────────────────────────────────────────
    place_parser = subparsers.add_parser(
        "place",
        help="Place sequences against a reference database.",
    )
    place_parser.add_argument(
        "input",
        type=str,
        help="Input FASTA file.",
    )
    place_parser.add_argument(
        "-r", "--reference",
        type=str,
        required=True,
        help="Path to reference database (.pkl or .json).",
    )
    place_parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file path (default: stdout as TSV).",
    )
    place_parser.add_argument(
        "--format",
        choices=["tsv", "json", "jplace"],
        default="tsv",
        help="Output format (default: tsv).",
    )
    place_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of candidate placements per query (default: 5).",
    )
    place_parser.add_argument(
        "--mode",
        choices=["flat", "hierarchical"],
        default="flat",
        help="Placement mode (default: flat).",
    )
    place_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to BiosphereCodec/V13 checkpoint (default: k-mer encoder).",
    )
    place_parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to BPE vocab JSON (required for V13 checkpoint mode if autodetect fails).",
    )
    place_parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokenized length for BPE encoder mode (default: 512).",
    )
    place_parser.add_argument(
        "--kappa",
        type=float,
        default=1.247,
        help="Curvature constant (default: 1.247).",
    )
    place_parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Compute device (default: cpu).",
    )
    place_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress.",
    )

    # ── info ─────────────────────────────────────────────────────────────
    info_parser = subparsers.add_parser(
        "info",
        help="Display information about a reference database.",
    )
    info_parser.add_argument(
        "-r", "--reference",
        type=str,
        required=True,
        help="Path to reference database.",
    )

    # ── build-ref ────────────────────────────────────────────────────────
    build_ref_parser = subparsers.add_parser(
        "build-ref",
        help="Build reference database from a manifest + V13 checkpoint.",
    )
    build_ref_parser.add_argument("--manifest", type=str, required=True, help="Training manifest CSV path.")
    build_ref_parser.add_argument("--output", type=str, required=True, help="Output reference (.pkl or .json).")
    build_ref_parser.add_argument("--model", type=str, required=True, help="V13 checkpoint path.")
    build_ref_parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to BPE vocab JSON (required if autodetect fails).",
    )
    build_ref_parser.add_argument("--split", type=str, default="train", help="Manifest split to use (default: train).")
    build_ref_parser.add_argument(
        "--rank",
        choices=["family", "genus", "species"],
        default="family",
        help="Leaf rank for prototype construction (default: family).",
    )
    build_ref_parser.add_argument("--batch-size", type=int, default=16)
    build_ref_parser.add_argument("--max-samples", type=int, default=0)
    build_ref_parser.add_argument("--max-tokens", type=int, default=512)
    build_ref_parser.add_argument("--kappa", type=float, default=1.247)
    build_ref_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    build_ref_parser.add_argument("--seed", type=int, default=42)
    build_ref_parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable manifest row shuffling before max-samples truncation.",
    )

    return parser


def cmd_place(args: argparse.Namespace) -> None:
    """Execute the place command."""
    from biosphere_atlas.place.calibrator import PlacementCalibrator
    from biosphere_atlas.place.encoder import BiosphereEncoder
    from biosphere_atlas.place.place import place_sequences, placement_summary
    from biosphere_atlas.place.reference import ReferenceDB

    if args.verbose:
        print(f"atlas-place v0.1.0")
        print(f"  Input:     {args.input}")
        print(f"  Reference: {args.reference}")
        print(f"  Mode:      {args.mode}")
        print(f"  κ:         {args.kappa}")
        print()

    t0 = time.time()

    # Load reference
    if args.verbose:
        print("Loading reference database...")
    reference = ReferenceDB.load(args.reference)
    if args.verbose:
        summary = reference.summary()
        print(f"  Prototypes: {reference.size}")
        for rank, count in summary.items():
            print(f"    {rank}: {count}")
        print()

    # Initialize encoder
    encoder = BiosphereEncoder(
        model_path=args.model,
        embedding_dim=reference.embedding_dim,
        kappa=args.kappa,
        device=args.device,
        tokenizer_path=args.tokenizer,
        max_tokens=args.max_tokens,
    )

    # Initialize calibrator
    calibrator = PlacementCalibrator()

    # Run placement
    if args.verbose:
        print("Placing sequences...")
    results = place_sequences(
        fasta_path=args.input,
        reference=reference,
        encoder=encoder,
        calibrator=calibrator,
        output_path=args.output,
        output_format=args.format,
        top_k=args.top_k,
        mode=args.mode,
        kappa=args.kappa,
        device=args.device,
        verbose=args.verbose,
    )

    # Print to stdout if no output file
    if args.output is None:
        from biosphere_atlas.place.io import TSV_COLUMNS

        print("\t".join(TSV_COLUMNS))
        for r in results:
            d = r.to_dict()
            print("\t".join(str(d.get(col, "")) for col in TSV_COLUMNS))

    # Summary
    elapsed = time.time() - t0
    if args.verbose:
        print()
        stats = placement_summary(results)
        print(f"Summary:")
        print(f"  Total placed:  {stats['total']}")
        if "zones" in stats:
            z = stats["zones"]
            print(f"  Accept:        {z.get('accept', 0)} ({stats.get('accept_rate', 0):.1%})")
            print(f"  Escalate:      {z.get('escalate', 0)}")
            print(f"  Fallback:      {z.get('fallback', 0)}")
        print(f"  Mean distance: {stats.get('distance_mean', 0):.4f}")
        if "confidence_mean" in stats:
            print(f"  Mean confidence: {stats['confidence_mean']:.4f}")
        print(f"  Elapsed:       {elapsed:.1f}s")


def cmd_info(args: argparse.Namespace) -> None:
    """Display reference database info."""
    from biosphere_atlas.place.reference import ReferenceDB

    reference = ReferenceDB.load(args.reference)
    print(f"atlas-place reference database")
    print(f"  Path:          {args.reference}")
    print(f"  κ:             {reference.kappa}")
    print(f"  Embedding dim: {reference.embedding_dim}")
    print(f"  Total protos:  {reference.size}")
    print()
    summary = reference.summary()
    for rank, count in summary.items():
        print(f"  {rank:>10s}: {count}")


def cmd_build_ref(args: argparse.Namespace) -> None:
    """Build a reference DB from tokenized manifest rows and a V13 checkpoint."""
    from biosphere_atlas.place.build_reference import build_reference_from_manifest

    stats = build_reference_from_manifest(
        manifest_path=args.manifest,
        output_path=args.output,
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        split=args.split,
        rank=args.rank,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
        device=args.device,
        kappa=args.kappa,
        shuffle_rows=not args.no_shuffle,
        seed=args.seed,
    )
    print("atlas-place reference build complete")
    print(f"  output: {args.output}")
    for k, v in stats.items():
        if isinstance(v, float) and abs(v - round(v)) < 1e-9:
            print(f"  {k}: {int(v)}")
        else:
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        # Default: treat first positional arg as FASTA input for 'place'
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            sys.argv.insert(1, "place")
            args = parser.parse_args()
        else:
            parser.print_help()
            sys.exit(1)

    if args.command == "place":
        cmd_place(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "build-ref":
        cmd_build_ref(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
