"""
I/O utilities for atlas-place.
===============================

Supports:
- FASTA reading (streaming)
- TSV output (native atlas-place format)
- JSON output (JSONL, one record per line)
- .jplace output (pplacer-compatible, IETF draft format)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Generator, List, Optional, TextIO, Tuple, Union

from biosphere_atlas.place.placer import PlacementResult


# ── FASTA reader ─────────────────────────────────────────────────────────────

def read_fasta(
    path: Union[str, Path],
) -> Generator[Tuple[str, str], None, None]:
    """
    Yield (header, sequence) pairs from a FASTA file.

    Handles multi-line sequences and strips whitespace.
    """
    path = Path(path)
    header = ""
    seq_parts: List[str] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    yield header, "".join(seq_parts)
                header = line[1:].split()[0]  # First word after >
                seq_parts = []
            else:
                seq_parts.append(line)

    if header:
        yield header, "".join(seq_parts)


# ── TSV output ───────────────────────────────────────────────────────────────

TSV_COLUMNS = [
    "sequence_id",
    "classification",
    "rank",
    "lineage",
    "distance",
    "margin",
    "zone",
    "confidence",
    "prediction_set_size",
    "atlas_r",
    "atlas_theta",
    "n_candidates",
]


def write_results_tsv(
    results: List[PlacementResult],
    path: Union[str, Path],
) -> None:
    """Write placement results to TSV."""
    path = Path(path)
    with open(path, "w") as f:
        f.write("\t".join(TSV_COLUMNS) + "\n")
        for r in results:
            d = r.to_dict()
            row = "\t".join(str(d.get(col, "")) for col in TSV_COLUMNS)
            f.write(row + "\n")


def write_results_tsv_stream(
    result: PlacementResult,
    handle: TextIO,
    write_header: bool = False,
) -> None:
    """Write a single result to an open TSV handle."""
    if write_header:
        handle.write("\t".join(TSV_COLUMNS) + "\n")
    d = result.to_dict()
    row = "\t".join(str(d.get(col, "")) for col in TSV_COLUMNS)
    handle.write(row + "\n")
    handle.flush()


# ── JSON output ──────────────────────────────────────────────────────────────

def write_results_json(
    results: List[PlacementResult],
    path: Union[str, Path],
) -> None:
    """Write placement results as JSONL (one JSON object per line)."""
    path = Path(path)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")


# ── .jplace output (pplacer-compatible) ──────────────────────────────────────

def write_results_jplace(
    results: List[PlacementResult],
    path: Union[str, Path],
    tree: str = "()",
    metadata: Optional[Dict] = None,
) -> None:
    """
    Write placement results in .jplace format.

    The .jplace format (Matsen et al. 2012) is used by pplacer, gappa, and iTOL.
    Structure:

        {
            "version": 3,
            "tree": "<newick>",
            "fields": ["edge_num", "likelihood", "like_weight_ratio",
                        "distal_length", "pendant_length"],
            "placements": [
                {
                    "n": ["<name>"],
                    "p": [[edge, lk, lwr, dl, pl], ...]
                },
                ...
            ],
            "metadata": {...}
        }

    Since atlas-place uses coordinate-based placement (not tree edges),
    we encode the placement distance as pendant_length and confidence
    as like_weight_ratio.  The edge_num maps to the nearest prototype index.
    """
    fields = [
        "edge_num",
        "likelihood",
        "like_weight_ratio",
        "distal_length",
        "pendant_length",
    ]

    placements = []
    for r in results:
        p_entries = []
        for i, cand in enumerate(r.candidates):
            # Map atlas-place fields to jplace fields
            edge_num = hash(cand.taxon_id) % 100000  # Pseudo edge number
            likelihood = -cand.distance  # Higher = better
            lwr = r.confidence if r.confidence is not None and i == 0 else 0.0
            distal_length = 0.0  # Not applicable for coordinate placement
            pendant_length = cand.distance

            p_entries.append([edge_num, likelihood, lwr, distal_length, pendant_length])

        placements.append({
            "n": [r.sequence_id],
            "p": p_entries,
        })

    jplace = {
        "version": 3,
        "tree": tree,
        "fields": fields,
        "placements": placements,
        "metadata": metadata or {
            "invocation": "atlas-place v0.1.0",
            "note": "Coordinate-based placement via BiosphereAtlas hyperbolic geometry",
        },
    }

    path = Path(path)
    with open(path, "w") as f:
        json.dump(jplace, f, indent=2)
