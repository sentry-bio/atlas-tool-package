"""
Command-line interface for atlas-viewer.
=========================================

Usage::

    atlas-viewer render --tree tree.json -o viewer.html
    atlas-viewer render --embeddings coords.pt --taxa taxa.txt -o viewer.html
    atlas-viewer info --tree tree.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="atlas-viewer",
        description="Interactive Poincare disk viewer for BiosphereAtlas.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- render ----------------------------------------------------------------
    render_p = sub.add_parser("render", help="Generate HTML viewer.")
    render_p.add_argument("--tree", type=str, default=None,
                          help="Path to tree.json from atlas-tree.")
    render_p.add_argument("--embeddings", type=str, default=None,
                          help="Path to embeddings (.pt or .npy).")
    render_p.add_argument("--taxa", type=str, default=None,
                          help="Path to taxa list (one per line).")
    render_p.add_argument("--lineages", type=str, default=None,
                          help="Path to lineages (semicolon-separated, one per line).")
    render_p.add_argument("-o", "--output", required=True, type=str,
                          help="Output HTML file path.")
    render_p.add_argument("--title", type=str, default="BiosphereAtlas Viewer",
                          help="Page title.")
    render_p.add_argument("--kappa", type=float, default=1.247,
                          help="Curvature constant (default: 1.247).")
    render_p.add_argument("--canvas-size", type=int, default=900,
                          help="Canvas width/height in pixels.")
    render_p.add_argument("--color-by", choices=["rank", "depth", "none"],
                          default="rank", help="Coloring scheme.")

    # -- info ------------------------------------------------------------------
    info_p = sub.add_parser("info", help="Display data statistics.")
    info_p.add_argument("--tree", type=str, default=None)
    info_p.add_argument("--embeddings", type=str, default=None)
    info_p.add_argument("--taxa", type=str, default=None)
    info_p.add_argument("--kappa", type=float, default=1.247)

    args = parser.parse_args(argv)

    if args.command == "render":
        _cmd_render(args)
    elif args.command == "info":
        _cmd_info(args)


def _cmd_render(args) -> None:
    from .data import from_tree, from_embeddings
    from .render import generate_viewer_html

    # Load data
    if args.tree:
        data = from_tree(args.tree, kappa=args.kappa, title=args.title)
    elif args.embeddings and args.taxa:
        data = from_embeddings(
            args.embeddings, args.taxa,
            kappa=args.kappa, lineages=args.lineages,
            title=args.title,
        )
    else:
        print("Provide --tree or (--embeddings + --taxa).", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {data.n_organisms} organisms, {len(data.edges)} edges")
    if data.projection_variance:
        pct = sum(data.projection_variance) * 100
        print(f"PCA variance explained: {pct:.1f}%")

    generate_viewer_html(
        data, args.output,
        title=args.title,
        canvas_size=args.canvas_size,
        color_by=args.color_by,
    )
    print(f"Viewer written to {args.output}")


def _cmd_info(args) -> None:
    from .data import from_tree, from_embeddings

    if args.tree:
        data = from_tree(args.tree, kappa=args.kappa)
    elif args.embeddings and args.taxa:
        data = from_embeddings(args.embeddings, args.taxa, kappa=args.kappa)
    else:
        print("Provide --tree or (--embeddings + --taxa).", file=sys.stderr)
        sys.exit(1)

    print(f"Organisms: {data.n_organisms}")
    print(f"Edges: {len(data.edges)}")
    print(f"Kappa: {data.kappa}")
    print(f"Rank bands: {len(data.rank_bands)}")
    if data.projection_variance:
        for i, v in enumerate(data.projection_variance):
            print(f"  PC{i+1}: {v*100:.1f}%")


if __name__ == "__main__":
    main()
