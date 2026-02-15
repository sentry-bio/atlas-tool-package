"""CLI for atlas-dark: dark matter mapping and genome triage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="atlas-dark",
        description="Dark matter biology mapper and genome triage.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- map -------------------------------------------------------------------
    map_p = sub.add_parser("map", help="Build dark matter map from a reference DB.")
    map_p.add_argument("--reference", required=True, help="ReferenceDB path (.pkl/.json)")
    map_p.add_argument("--rank", default="family")
    map_p.add_argument("--k", type=int, default=5, help="k for sigma estimation")
    map_p.add_argument("--dark-quantile", type=float, default=0.95)
    map_p.add_argument("-o", "--output", default=None, help="Output JSON path")

    # -- triage ----------------------------------------------------------------
    tri_p = sub.add_parser("triage", help="Triage FASTA genomes into redundant/novel/dark.")
    tri_p.add_argument("--input", required=True, help="Input FASTA")
    tri_p.add_argument("--reference", required=True, help="ReferenceDB path")
    tri_p.add_argument("--model", default=None, help="Atlas checkpoint")
    tri_p.add_argument("--tokenizer", default=None)
    tri_p.add_argument("--rank", default="family")
    tri_p.add_argument("--k", type=int, default=5)
    tri_p.add_argument("--epsilon", type=float, default=None)
    tri_p.add_argument("--sigma-threshold", type=float, default=None)
    tri_p.add_argument("--device", default="cpu")
    tri_p.add_argument("--max-tokens", type=int, default=512)
    tri_p.add_argument("--format", choices=["tsv", "jsonl"], default="tsv")
    tri_p.add_argument("-o", "--output", default=None)

    args = parser.parse_args(argv)

    if args.command == "map":
        _cmd_map(args)
    elif args.command == "triage":
        _cmd_triage(args)


def _parse_rank(s):
    from biosphere_atlas.place.reference import Rank
    return {
        "domain": Rank.DOMAIN, "phylum": Rank.PHYLUM, "class": Rank.CLASS,
        "order": Rank.ORDER, "family": Rank.FAMILY, "genus": Rank.GENUS,
        "species": Rank.SPECIES,
    }[s.strip().lower()]


def _cmd_map(args):
    from biosphere_atlas.place.reference import ReferenceDB
    from biosphere_atlas.dark.field import UncertaintyField
    from biosphere_atlas.dark.dark import DarkMatterMap

    ref = ReferenceDB.load(args.reference)
    rank = _parse_rank(args.rank)
    field = UncertaintyField(ref, rank=rank, k=args.k, kappa=ref.kappa)
    dm = DarkMatterMap.from_field(field, dark_quantile=args.dark_quantile)

    out = json.dumps(dm.to_dict(), indent=2)
    if args.output:
        Path(args.output).write_text(out)
    else:
        sys.stdout.write(out + "\n")

    print(f"Dark matter map: {dm.summary['n_dark']} dark / "
          f"{dm.summary['n_charted']} charted / "
          f"{dm.summary['n_total']} total "
          f"(threshold={dm.dark_threshold:.4f})", file=sys.stderr)


def _cmd_triage(args):
    from biosphere_atlas.dark.triage import triage_genomes

    rank = _parse_rank(args.rank)
    results = triage_genomes(
        input_fasta=args.input,
        reference=args.reference,
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        rank=rank,
        k=args.k,
        epsilon=args.epsilon,
        sigma_threshold=args.sigma_threshold,
        device=args.device,
        max_tokens=args.max_tokens,
    )

    rows = [r.to_dict() for r in results]
    if args.format == "jsonl":
        lines = "\n".join(json.dumps(r) for r in rows) + "\n"
    else:
        cols = ["sequence_id", "category", "d_geo", "sigma_local",
                "nearest_taxon_id", "epsilon", "sigma_threshold"]
        header = "\t".join(cols)
        body = []
        for r in rows:
            body.append("\t".join([
                str(r["sequence_id"]),
                str(r["category"]),
                f"{r['d_geo']:.6f}",
                f"{r['sigma_local']:.6f}",
                str(r["nearest_taxon_id"]),
                f"{r['epsilon']:.6f}",
                f"{r['sigma_threshold']:.6f}",
            ]))
        lines = header + "\n" + "\n".join(body) + ("\n" if body else "")

    if args.output:
        Path(args.output).write_text(lines)
    else:
        sys.stdout.write(lines)

    # Summary to stderr
    from collections import Counter
    cats = Counter(r["category"] for r in rows)
    print(f"Triage: {cats.get('redundant',0)} redundant, "
          f"{cats.get('novel_certain',0)} novel-certain, "
          f"{cats.get('novel_uncertain',0)} novel-uncertain "
          f"(n={len(results)})", file=sys.stderr)


if __name__ == "__main__":
    main()

