"""
Command-line interface for atlas-viewer.
=========================================

Usage::

    # 2D interactive Poincaré disk
    atlas-viewer render --tree tree.json -o viewer.html

    # 3D dark-mode Poincaré ball (requires plotly CDN)
    atlas-viewer render --mode 3d --tree tree.json -o viewer_3d.html

    # From raw embeddings
    atlas-viewer render --embeddings coords.pt --taxa taxa.txt -o viewer.html

    # Data statistics
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
    render_p.add_argument("--color-by", choices=["rank", "depth", "domain", "none"],
                          default="rank", help="Coloring scheme.")
    render_p.add_argument("--mode", choices=["2d", "3d"], default="2d",
                          help="Viewer mode: 2d (Poincaré disk) or 3d (Poincaré ball).")

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
    from .render3d import generate_3d_viewer_html

    mode = getattr(args, "mode", "2d")
    color_by = getattr(args, "color_by", "rank")

    # Load data
    if args.tree:
        data = from_tree(args.tree, kappa=args.kappa, title=args.title)
    elif args.embeddings and args.taxa:
        data = from_embeddings(
            args.embeddings, args.taxa,
            kappa=args.kappa, lineages=getattr(args, "lineages", None),
            title=args.title,
        )
    else:
        print("Provide --tree or (--embeddings + --taxa).", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {data.n_organisms} organisms, {len(data.edges)} edges")

    if mode == "3d":
        if data.coords_3d is None:
            print("ERROR: 3D coordinates not available for this data source.", file=sys.stderr)
            sys.exit(1)
        if data.projection_variance_3d:
            pct = sum(data.projection_variance_3d) * 100
            print(f"3D PCA variance explained: {pct:.1f}%")
        generate_3d_viewer_html(
            data, args.output,
            title=args.title,
            color_by=color_by if color_by != "rank" else "domain",
        )
        print(f"3D viewer written to {args.output}")
    else:
        if data.projection_variance:
            pct = sum(data.projection_variance) * 100
            print(f"PCA variance explained: {pct:.1f}%")
        generate_viewer_html(
            data, args.output,
            title=args.title,
            canvas_size=getattr(args, "canvas_size", 900),
            color_by=color_by,
        )
        print(f"2D viewer written to {args.output}")


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
