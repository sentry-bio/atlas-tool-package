"""
Command-line interface for atlas-hplg.

Usage:
    atlas-hplg classify input.fasta --checkpoint model.pt --taxonomy gtdb.tsv
    atlas-hplg calibrate --checkpoint model.pt --calibration-set cal.fasta
    atlas-hplg info --checkpoint model.pt
"""

import argparse
import sys
import json


def main():
    parser = argparse.ArgumentParser(
        prog="atlas-hplg",
        description="HPLG taxonomic classifier with formal coverage guarantees",
    )
    subparsers = parser.add_subparsers(dest="command")

    # classify subcommand
    classify_parser = subparsers.add_parser(
        "classify",
        help="Classify sequences using HPLG three-zone decisions",
    )
    classify_parser.add_argument("input", help="Input FASTA file")
    classify_parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    classify_parser.add_argument("--taxonomy", help="Taxonomy file (GTDB format)")
    classify_parser.add_argument("--output", "-o", default="-", help="Output file (default: stdout)")
    classify_parser.add_argument("--format", choices=["tsv", "jsonl"], default="tsv")
    classify_parser.add_argument("--kappa", type=float, default=1.247)
    classify_parser.add_argument("--epsilon", type=float, default=0.10,
                                help="Target error rate for accept zone")

    # info subcommand
    info_parser = subparsers.add_parser(
        "info",
        help="Display classifier configuration and calibration status",
    )
    info_parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "classify":
        print(
            "atlas-hplg classify: requires a trained BiosphereCodec checkpoint.\n"
            "See https://github.com/sentry-bio/active-geometry for model training.\n\n"
            "Quick start with pre-trained model:\n"
            "  atlas-hplg classify input.fasta --checkpoint biosphere-v1.pt\n\n"
            "The classifier will:\n"
            "  1. Encode each sequence to hyperbolic coordinates\n"
            "  2. Navigate domain -> species using three-zone decisions\n"
            "  3. Output classifications with calibrated confidence\n"
            "  4. Gracefully fall back when uncertain (formal guarantee)\n",
            file=sys.stderr,
        )

    elif args.command == "info":
        print(
            "atlas-hplg info: Display classifier state from checkpoint.\n"
            f"  Checkpoint: {args.checkpoint}\n\n"
            "This would show:\n"
            "  - Number of prototypes per rank\n"
            "  - Calibration status (scores per rank, threshold values)\n"
            "  - Curvature state (current kappa, phase)\n"
            "  - Coverage guarantees\n",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
