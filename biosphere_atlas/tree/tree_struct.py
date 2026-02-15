"""
Phylogenetic tree data structures.
===================================

``TreeNode`` and ``PhyloTree`` represent unrooted or rooted phylogenies
built from Poincare ball coordinates.  Edge lengths are geodesic
distances in the underlying hyperbolic manifold.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Set, Tuple

import torch
from torch import Tensor

from biosphere_atlas.core.hyperbolic import KAPPA_DEFAULT, poincare_distance


# -- Node ----------------------------------------------------------------------

@dataclass
class TreeNode:
    """A node in a phylogenetic tree.

    Leaf nodes have ``taxon_id`` set and an embedding in the Poincare ball.
    Internal nodes may have embeddings (Steiner points) or may be virtual.
    """

    node_id: int
    taxon_id: Optional[str] = None
    embedding: Optional[Tensor] = None
    parent: Optional[int] = None
    children: List[int] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def __repr__(self) -> str:
        label = self.taxon_id or f"node_{self.node_id}"
        kind = "leaf" if self.is_leaf else "internal"
        return f"TreeNode({label}, {kind}, children={len(self.children)})"


# -- Edge ----------------------------------------------------------------------

@dataclass
class TreeEdge:
    """A weighted edge in the phylogenetic tree."""

    source: int
    target: int
    length: float
    """Geodesic distance between source and target embeddings."""
    support: float = 1.0
    """Bootstrap or quartet support, in [0, 1]."""


# -- Tree ----------------------------------------------------------------------

class PhyloTree:
    """An unrooted (or rooted) phylogenetic tree with geodesic edge lengths.

    Nodes are stored by integer ID.  Leaves carry taxon labels and
    Poincare ball embeddings.  Internal nodes may carry Steiner-point
    embeddings after refinement.

    Parameters
    ----------
    kappa : float
        Curvature of the Poincare ball (default 1.247).
    """

    def __init__(self, kappa: float = KAPPA_DEFAULT) -> None:
        self.kappa = kappa
        self._nodes: Dict[int, TreeNode] = {}
        self._edges: Dict[Tuple[int, int], TreeEdge] = {}
        self._next_id = 0

    # -- Construction ----------------------------------------------------------

    def add_leaf(
        self, taxon_id: str, embedding: Tensor, parent: Optional[int] = None
    ) -> int:
        """Add a leaf node with a taxon label and embedding.  Returns node ID."""
        nid = self._alloc_id()
        node = TreeNode(
            node_id=nid, taxon_id=taxon_id, embedding=embedding, parent=parent
        )
        self._nodes[nid] = node
        if parent is not None and parent in self._nodes:
            self._nodes[parent].children.append(nid)
        return nid

    def add_internal(
        self,
        children: Optional[List[int]] = None,
        embedding: Optional[Tensor] = None,
        parent: Optional[int] = None,
    ) -> int:
        """Add an internal node.  Returns node ID."""
        nid = self._alloc_id()
        node = TreeNode(
            node_id=nid, embedding=embedding, parent=parent,
            children=list(children) if children else [],
        )
        self._nodes[nid] = node
        # Wire children's parent pointers
        for cid in node.children:
            if cid in self._nodes:
                self._nodes[cid].parent = nid
        if parent is not None and parent in self._nodes:
            self._nodes[parent].children.append(nid)
        return nid

    def add_edge(self, src: int, tgt: int, length: float, support: float = 1.0) -> None:
        """Record a weighted edge between two nodes."""
        edge = TreeEdge(source=src, target=tgt, length=length, support=support)
        key = (min(src, tgt), max(src, tgt))
        self._edges[key] = edge

    def connect(self, parent_id: int, child_id: int) -> None:
        """Wire parent-child relationship and compute geodesic edge length."""
        p = self._nodes[parent_id]
        c = self._nodes[child_id]
        c.parent = parent_id
        if child_id not in p.children:
            p.children.append(child_id)
        if p.embedding is not None and c.embedding is not None:
            d = poincare_distance(p.embedding, c.embedding, self.kappa).item()
            self.add_edge(parent_id, child_id, d)

    # -- Accessors -------------------------------------------------------------

    @property
    def n_leaves(self) -> int:
        return sum(1 for n in self._nodes.values() if n.is_leaf)

    @property
    def n_internal(self) -> int:
        return sum(1 for n in self._nodes.values() if not n.is_leaf)

    @property
    def n_nodes(self) -> int:
        return len(self._nodes)

    @property
    def n_edges(self) -> int:
        return len(self._edges)

    def get_node(self, node_id: int) -> TreeNode:
        return self._nodes[node_id]

    def get_edge(self, src: int, tgt: int) -> Optional[TreeEdge]:
        key = (min(src, tgt), max(src, tgt))
        return self._edges.get(key)

    def leaves(self) -> List[TreeNode]:
        return [n for n in self._nodes.values() if n.is_leaf]

    def internal_nodes(self) -> List[TreeNode]:
        return [n for n in self._nodes.values() if not n.is_leaf]

    def leaf_ids(self) -> List[int]:
        return [n.node_id for n in self._nodes.values() if n.is_leaf]

    def edges(self) -> List[TreeEdge]:
        return list(self._edges.values())

    def total_branch_length(self) -> float:
        """Sum of all edge lengths."""
        return sum(e.length for e in self._edges.values())

    # -- Neighbours (for unrooted tree traversal) ------------------------------

    def _adjacency(self) -> Dict[int, List[Tuple[int, float]]]:
        """Build adjacency list from edges."""
        adj: Dict[int, List[Tuple[int, float]]] = {nid: [] for nid in self._nodes}
        for edge in self._edges.values():
            adj[edge.source].append((edge.target, edge.length))
            adj[edge.target].append((edge.source, edge.length))
        return adj

    def path_distance(self, a: int, b: int) -> float:
        """Patristic (tree-path) distance between two nodes via BFS."""
        if a == b:
            return 0.0
        adj = self._adjacency()
        visited: Set[int] = {a}
        queue: List[Tuple[int, float]] = [(a, 0.0)]
        while queue:
            current, dist = queue.pop(0)
            for nb, w in adj.get(current, []):
                if nb == b:
                    return dist + w
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + w))
        return float("inf")

    # -- Root-finding ----------------------------------------------------------

    def find_root(self) -> Optional[int]:
        """Find the root (node with no parent).

        If multiple rootless nodes exist (unrooted tree), prefer the first
        internal node.  Returns None only if the tree is empty.
        """
        if not self._nodes:
            return None
        roots = [nid for nid, n in self._nodes.items() if n.parent is None]
        if len(roots) == 1:
            return roots[0]
        # Prefer internal nodes as root
        internal_roots = [r for r in roots if not self._nodes[r].is_leaf]
        if internal_roots:
            return internal_roots[0]
        # All roots are leaves — pick the first
        return roots[0] if roots else None

    def root_at(self, node_id: int) -> None:
        """Re-root the tree at the given node using adjacency edges.

        Rebuilds parent/children pointers via BFS from the new root,
        using the edge structure (not existing parent pointers).
        """
        adj = self._adjacency()
        # BFS from new root to assign parent/children
        for n in self._nodes.values():
            n.parent = None
            n.children = []

        visited: Set[int] = {node_id}
        queue = [node_id]
        while queue:
            current = queue.pop(0)
            for nb, _ in adj.get(current, []):
                if nb not in visited:
                    visited.add(nb)
                    self._nodes[current].children.append(nb)
                    self._nodes[nb].parent = current
                    queue.append(nb)

    # -- Serialization ---------------------------------------------------------

    def to_dict(self) -> Dict:
        """Serialize to a JSON-compatible dictionary."""
        nodes = []
        for n in self._nodes.values():
            nd = {
                "node_id": n.node_id,
                "taxon_id": n.taxon_id,
                "parent": n.parent,
                "children": n.children,
                "is_leaf": n.is_leaf,
            }
            if n.embedding is not None:
                nd["embedding"] = n.embedding.tolist()
            nodes.append(nd)

        edges = [
            {"source": e.source, "target": e.target,
             "length": e.length, "support": e.support}
            for e in self._edges.values()
        ]

        return {
            "kappa": self.kappa,
            "n_leaves": self.n_leaves,
            "n_internal": self.n_internal,
            "total_branch_length": self.total_branch_length(),
            "nodes": nodes,
            "edges": edges,
        }

    def summary(self) -> str:
        """One-line human-readable summary."""
        return (
            f"PhyloTree(leaves={self.n_leaves}, internal={self.n_internal}, "
            f"edges={self.n_edges}, branch_len={self.total_branch_length():.4f}, "
            f"kappa={self.kappa})"
        )

    # -- Private ---------------------------------------------------------------

    def _alloc_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid
