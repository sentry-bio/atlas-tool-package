"""
Tree export: Newick, SVG, JSON.
================================

Export a ``PhyloTree`` to standard phylogenetics formats and
a basic radial SVG visualization.
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Set, TextIO, Tuple

from .tree_struct import PhyloTree, TreeNode


# -- Newick --------------------------------------------------------------------

def to_newick(tree: PhyloTree, root_id: Optional[int] = None) -> str:
    """Export tree in Newick format.

    If *root_id* is given, the tree is traversed from that root.
    Otherwise the root is auto-detected (node with no parent), or
    the first internal node is used.

    Returns a Newick string like ``((A:0.1,B:0.2):0.05,C:0.3);``
    """
    if root_id is None:
        root_id = tree.find_root()
    if root_id is None:
        # No explicit root — pick the first internal node
        internals = tree.internal_nodes()
        if internals:
            root_id = internals[0].node_id
        else:
            # All leaves — just list them
            leaves = tree.leaves()
            if not leaves:
                return "();"
            parts = [_leaf_label(n) for n in leaves]
            return "(" + ",".join(parts) + ");"

    adj = tree._adjacency()
    return _newick_subtree(tree, root_id, visited=set(), adj=adj) + ";"


def _newick_subtree(tree: PhyloTree, nid: int, visited: Set[int], adj: Dict) -> str:
    """Recursively build Newick string using adjacency structure."""
    visited.add(nid)
    node = tree.get_node(nid)
    label = node.taxon_id if node.taxon_id else ""

    # Always use adjacency for traversal (authoritative for NJ trees)
    subtrees = [nb for nb, _ in adj.get(nid, []) if nb not in visited]

    if not subtrees:
        # Leaf
        return label

    parts = []
    for cid in subtrees:
        child_str = _newick_subtree(tree, cid, visited, adj)
        edge = tree.get_edge(nid, cid)
        if edge is not None:
            parts.append(f"{child_str}:{edge.length:.6f}")
        else:
            parts.append(child_str)

    return f"({','.join(parts)}){label}"


def _leaf_label(node: TreeNode) -> str:
    return node.taxon_id if node.taxon_id else f"node_{node.node_id}"


# -- JSON export ---------------------------------------------------------------

def to_json(tree: PhyloTree, indent: int = 2) -> str:
    """Export tree as a JSON string."""
    return json.dumps(tree.to_dict(), indent=indent)


def write_json(tree: PhyloTree, path: str) -> None:
    """Write tree to a JSON file."""
    with open(path, "w") as f:
        f.write(to_json(tree))


def write_newick(tree: PhyloTree, path: str, root_id: Optional[int] = None) -> None:
    """Write tree in Newick format to a file."""
    with open(path, "w") as f:
        f.write(to_newick(tree, root_id))
        f.write("\n")


# -- SVG radial tree -----------------------------------------------------------

def to_svg(
    tree: PhyloTree,
    root_id: Optional[int] = None,
    width: int = 800,
    height: int = 800,
    leaf_font_size: int = 10,
    branch_color: str = "#336699",
    leaf_color: str = "#222222",
    show_support: bool = False,
) -> str:
    """Render a radial SVG tree visualization.

    The tree is laid out radially: leaves are placed at equal angles
    around a circle, and internal nodes are positioned at radial depths
    proportional to their distance from the root.

    Args:
        tree: the tree to render.
        root_id: root node ID (auto-detected if None).
        width, height: SVG dimensions.
        leaf_font_size: font size for leaf labels.
        branch_color: stroke color for branches.
        leaf_color: text color for leaf labels.
        show_support: if True, annotate edges with support values.

    Returns:
        SVG string.
    """
    if tree.n_nodes == 0:
        return _empty_svg(width, height)

    if root_id is None:
        root_id = tree.find_root()
    if root_id is None:
        return _empty_svg(width, height)

    # Assign angular positions to leaves
    leaf_order = _get_leaf_order(tree, root_id)
    if not leaf_order:
        return _empty_svg(width, height)

    n_leaves = len(leaf_order)
    angle_step = 2.0 * math.pi / max(n_leaves, 1)
    leaf_angles: Dict[int, float] = {}
    for i, lid in enumerate(leaf_order):
        leaf_angles[lid] = i * angle_step

    # Compute node depths (distance from root)
    depths: Dict[int, float] = {}
    _compute_depths(tree, root_id, 0.0, depths, set())

    max_depth = max(depths.values()) if depths else 1.0
    if max_depth < 1e-10:
        max_depth = 1.0

    # Compute angular position for internal nodes (mean of children)
    angles: Dict[int, float] = dict(leaf_angles)
    _compute_angles(tree, root_id, angles, set())

    # Map to SVG coordinates
    cx, cy = width / 2.0, height / 2.0
    radius = min(width, height) * 0.38

    def to_xy(nid: int) -> Tuple[float, float]:
        r = (depths.get(nid, 0.0) / max_depth) * radius
        a = angles.get(nid, 0.0)
        return cx + r * math.cos(a), cy + r * math.sin(a)

    # Build SVG
    lines: List[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
                 f'width="{width}" height="{height}" '
                 f'viewBox="0 0 {width} {height}">')
    lines.append(f'<rect width="100%" height="100%" fill="white"/>')

    # Edges
    visited_edges: Set[Tuple[int, int]] = set()
    for edge in tree.edges():
        key = (min(edge.source, edge.target), max(edge.source, edge.target))
        if key in visited_edges:
            continue
        visited_edges.add(key)
        x1, y1 = to_xy(edge.source)
        x2, y2 = to_xy(edge.target)
        lines.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" '
            f'x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{branch_color}" stroke-width="1.5" '
            f'stroke-linecap="round"/>'
        )
        if show_support and edge.support < 1.0:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            lines.append(
                f'<text x="{mx:.1f}" y="{my:.1f}" font-size="8" '
                f'fill="#999" text-anchor="middle">{edge.support:.2f}</text>'
            )

    # Leaf labels
    for lid in leaf_order:
        node = tree.get_node(lid)
        x, y = to_xy(lid)
        a = angles.get(lid, 0.0)
        deg = math.degrees(a)
        anchor = "start" if -90 < deg < 90 or deg > 270 else "end"
        rot = deg if -90 < deg < 90 else deg + 180
        label = node.taxon_id or f"node_{lid}"
        lines.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="{branch_color}"/>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{leaf_font_size}" '
            f'fill="{leaf_color}" text-anchor="{anchor}" '
            f'transform="rotate({rot:.1f},{x:.1f},{y:.1f})" '
            f'dx="5" dy="3">{label}</text>'
        )

    # Internal node dots
    for node in tree.internal_nodes():
        x, y = to_xy(node.node_id)
        lines.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2" fill="{branch_color}" opacity="0.5"/>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


def write_svg(tree: PhyloTree, path: str, **kwargs) -> None:
    """Write radial SVG tree to a file."""
    with open(path, "w") as f:
        f.write(to_svg(tree, **kwargs))


# -- SVG helpers ---------------------------------------------------------------

def _get_leaf_order(tree: PhyloTree, root_id: int) -> List[int]:
    """DFS leaf ordering from root, using adjacency structure."""
    order: List[int] = []
    visited: Set[int] = set()
    adj = tree._adjacency()

    def dfs(nid: int) -> None:
        if nid in visited:
            return
        visited.add(nid)
        node = tree.get_node(nid)
        neighbours = [nb for nb, _ in adj.get(nid, []) if nb not in visited]
        if not neighbours:
            # Leaf (or no unvisited neighbours)
            if node.is_leaf:
                order.append(nid)
            return
        for nb in neighbours:
            dfs(nb)

    dfs(root_id)
    return order


def _compute_depths(
    tree: PhyloTree, nid: int, depth: float,
    depths: Dict[int, float], visited: Set[int]
) -> None:
    if nid in visited:
        return
    visited.add(nid)
    depths[nid] = depth
    adj = tree._adjacency()
    for nb, w in adj.get(nid, []):
        if nb not in visited:
            _compute_depths(tree, nb, depth + w, depths, visited)


def _compute_angles(
    tree: PhyloTree, nid: int,
    angles: Dict[int, float], visited: Set[int]
) -> float:
    """Recursively compute angles; returns the angle for nid."""
    if nid in visited:
        return angles.get(nid, 0.0)
    visited.add(nid)

    if nid in angles:
        return angles[nid]

    adj = tree._adjacency()
    child_angles: List[float] = []
    for nb, _ in adj.get(nid, []):
        if nb not in visited:
            a = _compute_angles(tree, nb, angles, visited)
            child_angles.append(a)

    if child_angles:
        # Circular mean
        sx = sum(math.cos(a) for a in child_angles)
        sy = sum(math.sin(a) for a in child_angles)
        angles[nid] = math.atan2(sy, sx)
    else:
        angles[nid] = 0.0

    return angles[nid]


def _empty_svg(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}">'
        f'<rect width="100%" height="100%" fill="white"/>'
        f'<text x="{width//2}" y="{height//2}" text-anchor="middle" '
        f'fill="#999">Empty tree</text></svg>'
    )
