from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from biosphere_atlas.novelty.novelty import detect_novel_sequences
from biosphere_atlas.place.reference import Rank


def _parse_rank(s: str) -> Rank:
    s = s.strip().lower()
    mapping = {
        "domain": Rank.DOMAIN,
        "phylum": Rank.PHYLUM,
        "class": Rank.CLASS,
        "order": Rank.ORDER,
        "family": Rank.FAMILY,
        "genus": Rank.GENUS,
        "species": Rank.SPECIES,
    }
    if s not in mapping:
        raise ValueError(f"Unknown rank: {s}")
    return mapping[s]


def _write_results(results, output: str | None, fmt: str) -> None:
    rows = [r.to_dict() for r in results]
    if fmt == "jsonl":
        lines = "\n".join(json.dumps(r) for r in rows) + "\n"
    else:
        cols = [
            "sequence_id",
            "novelty_score",
            "is_novel",
            "threshold",
            "nearest_taxon_id",
            "nearest_distance",
        ]
        header = "\t".join(cols)
        body = []
        for r in rows:
            body.append(
                "\t".join(
                    [
                        str(r["sequence_id"]),
                        f"{float(r['novelty_score']):.6f}",
                        str(bool(r["is_novel"])),
                        f"{float(r['threshold']):.6f}",
                        str(r["nearest_taxon_id"]),
                        f"{float(r['nearest_distance']):.6f}",
                    ]
                )
            )
        lines = header + "\n" + "\n".join(body) + ("\n" if body else "")

    if output:
        Path(output).write_text(lines)
    else:
        sys.stdout.write(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="atlas-novelty",
        description="Novelty detection via geodesic distance to reference prototypes",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    detect = sub.add_parser("detect", help="Detect novelty for FASTA sequences")
    detect.add_argument("--input", required=True, help="Input FASTA path")
    detect.add_argument("--reference", required=True, help="ReferenceDB path (.pkl/.json)")
    detect.add_argument("--model", default=None, help="Atlas checkpoint path (optional)")
    detect.add_argument("--tokenizer", default=None, help="Tokenizer path for V13 checkpoints")
    detect.add_argument("--rank", default="family", help="Rank for novelty scoring")
    detect.add_argument("--threshold", type=float, default=None, help="Manual novelty threshold")
    detect.add_argument(
        "--auto-quantile",
        type=float,
        default=0.99,
        help="Quantile for auto-threshold when --threshold is omitted",
    )
    detect.add_argument("--top-k", type=int, default=5, help="Top-k nearest prototypes to include")
    detect.add_argument("--device", default="cpu", help="cpu/cuda")
    detect.add_argument("--max-tokens", type=int, default=512)
    detect.add_argument("--format", choices=["tsv", "jsonl"], default="tsv")
    detect.add_argument("-o", "--output", default=None)

    args = parser.parse_args(argv)
    if args.command == "detect":
        rank = _parse_rank(args.rank)
        results = detect_novel_sequences(
            input_fasta=args.input,
            reference=args.reference,
            model_path=args.model,
            tokenizer_path=args.tokenizer,
            rank=rank,
            threshold=args.threshold,
            auto_threshold_quantile=args.auto_quantile,
            device=args.device,
            max_tokens=args.max_tokens,
            top_k=args.top_k,
        )
        _write_results(results, args.output, args.format)


if __name__ == "__main__":
    main()
