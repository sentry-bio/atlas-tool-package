"""
Command-line interface for atlas-tree.
=======================================

Usage::

    atlas-tree build --embeddings coords.pt --taxa taxa.txt -o tree.nwk
    atlas-tree check --tree tree.json --embeddings coords.pt
    atlas-tree info --tree tree.json

Subcommands
-----------
build   Build a phylogenetic tree from Poincare ball embeddings.
check   Check quartet consistency of a tree against coordinates.
info    Print summary information about a tree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="atlas-tree",
        description="Phylogenetic tree construction from BiosphereAtlas hyperbolic coordinates.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- build -----------------------------------------------------------------
    build_p = sub.add_parser("build", help="Build tree from embeddings.")
    build_p.add_argument(
        "--embeddings", required=True, type=str,
        help="Path to embeddings file (.pt tensor or .npy).",
    )
    build_p.add_argument(
        "--taxa", required=True, type=str,
        help="Path to taxa list (one taxon per line).",
    )
    build_p.add_argument(
        "-o", "--output", required=True, type=str,
        help="Output file path (.nwk, .json, or .svg).",
    )
    build_p.add_argument(
        "--kappa", type=float, default=1.247,
        help="Curvature constant (default: 1.247).",
    )
    build_p.add_argument(
        "--format", choices=["newick", "json", "svg"], default=None,
        help="Output format (auto-detected from extension).",
    )
    build_p.add_argument(
        "--midpoint-root", action="store_true",
        help="Root tree at midpoint of longest path.",
    )

    # -- check -----------------------------------------------------------------
    check_p = sub.add_parser("check", help="Check quartet consistency.")
    check_p.add_argument(
        "--tree", required=True, type=str,
        help="Path to tree JSON file.",
    )
    check_p.add_argument(
        "--embeddings", required=True, type=str,
        help="Path to embeddings file (.pt tensor).",
    )
    check_p.add_argument(
        "--max-quartets", type=int, default=10000,
        help="Maximum number of quartets to sample.",
    )
    check_p.add_argument("--kappa", type=float, default=1.247)
    check_p.add_argument("--seed", type=int, default=42)

    # -- info ------------------------------------------------------------------
    info_p = sub.add_parser("info", help="Print tree summary.")
    info_p.add_argument("--tree", required=True, type=str, help="Path to tree JSON file.")

    args = parser.parse_args(argv)

    if args.command == "build":
        _cmd_build(args)
    elif args.command == "check":
        _cmd_check(args)
    elif args.command == "info":
        _cmd_info(args)


def _cmd_build(args) -> None:
    from .nj import neighbor_joining, root_at_midpoint
    from .export import write_newick, write_json, write_svg

    # Load embeddings
    emb_path = Path(args.embeddings)
    if emb_path.suffix == ".pt":
        embeddings = torch.load(str(emb_path), map_location="cpu", weights_only=True)
    elif emb_path.suffix == ".npy":
        import numpy as np
        embeddings = torch.from_numpy(np.load(str(emb_path))).float()
    else:
        print(f"Unsupported embedding format: {emb_path.suffix}", file=sys.stderr)
        sys.exit(1)

    # Load taxa
    with open(args.taxa) as f:
        taxa = [line.strip() for line in f if line.strip()]

    if len(taxa) != embeddings.size(0):
        print(
            f"Mismatch: {len(taxa)} taxa vs {embeddings.size(0)} embeddings.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Building tree from {len(taxa)} taxa (dim={embeddings.size(1)}, kappa={args.kappa})...")

    tree = neighbor_joining(embeddings, taxa, kappa=args.kappa)

    if args.midpoint_root:
        root_at_midpoint(tree)

    # Detect format
    out_path = Path(args.output)
    fmt = args.format
    if fmt is None:
        ext_map = {".nwk": "newick", ".newick": "newick", ".json": "json", ".svg": "svg"}
        fmt = ext_map.get(out_path.suffix.lower(), "newick")

    if fmt == "newick":
        write_newick(tree, str(out_path))
    elif fmt == "json":
        write_json(tree, str(out_path))
    elif fmt == "svg":
        write_svg(tree, str(out_path))

    print(f"Wrote {fmt} to {out_path}")
    print(tree.summary())


def _cmd_check(args) -> None:
    from .quartet import check_quartet_consistency

    # Load tree
    with open(args.tree) as f:
        tree_data = json.load(f)

    # Reconstruct tree
    from .tree_struct import PhyloTree
    tree = PhyloTree(kappa=tree_data.get("kappa", args.kappa))
    node_map = {}
    for nd in tree_data["nodes"]:
        nid = nd["node_id"]
        if nd.get("taxon_id"):
            emb = torch.tensor(nd["embedding"]) if nd.get("embedding") else None
            tree.add_leaf(nd["taxon_id"], emb)
        else:
            emb = torch.tensor(nd["embedding"]) if nd.get("embedding") else None
            tree.add_internal(embedding=emb)
        node_map[nid] = nid

    for ed in tree_data["edges"]:
        tree.add_edge(ed["source"], ed["target"], ed["length"], ed.get("support", 1.0))

    # Load embeddings
    embeddings = torch.load(args.embeddings, map_location="cpu", weights_only=True)
    leaf_ids = tree.leaf_ids()
    emb_map = {}
    for i, lid in enumerate(leaf_ids):
        if i < embeddings.size(0):
            emb_map[lid] = embeddings[i]

    report = check_quartet_consistency(
        tree, emb_map, kappa=args.kappa,
        max_quartets=args.max_quartets, seed=args.seed,
    )
    print(report.summary())


def _cmd_info(args) -> None:
    with open(args.tree) as f:
        tree_data = json.load(f)

    print(f"Leaves: {tree_data.get('n_leaves', '?')}")
    print(f"Internal nodes: {tree_data.get('n_internal', '?')}")
    print(f"Total branch length: {tree_data.get('total_branch_length', '?'):.4f}")
    print(f"Kappa: {tree_data.get('kappa', '?')}")
    print(f"Edges: {len(tree_data.get('edges', []))}")


if __name__ == "__main__":
    main()
