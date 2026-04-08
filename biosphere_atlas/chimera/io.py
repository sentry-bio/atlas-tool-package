"""
I/O utilities for atlas-chimera.

Handles FASTA reading and results output in multiple formats.
Designed for seamless integration into existing metagenomics pipelines.
"""

import sys
from pathlib import Path
from dataclasses import asdict
from typing import List, TextIO, Optional, Iterator, Tuple

from biosphere_atlas.chimera.chimera import ChimeraScore
from biosphere_atlas.core.coordinates import BiosphereCoordinate


def read_fasta(path: str) -> Iterator[Tuple[str, str]]:
    """
    Read sequences from a FASTA file.

    Yields (header, sequence) tuples. Handles multi-line sequences
    and strips whitespace. Supports both .fasta and .fa extensions,
    and gzipped files (.fasta.gz, .fa.gz).

    Args:
        path: Path to FASTA file

    Yields:
        (header, sequence) tuples
    """
    filepath = Path(path)

    if filepath.suffix == ".gz":
        import gzip
        opener = gzip.open(filepath, "rt")
    else:
        opener = open(filepath, "r")

    with opener as f:
        header = None
        sequence_parts = []

        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # Yield previous sequence if exists
                if header is not None:
                    yield header, "".join(sequence_parts)

                header = line[1:].strip()
                sequence_parts = []
            else:
                sequence_parts.append(line.upper())

        # Yield final sequence
        if header is not None:
            yield header, "".join(sequence_parts)


def write_results_tsv(
    results: List[dict],
    output: Optional[str] = None,
):
    """
    Write chimera detection results as TSV.

    Output format:
        sequence_id  chimera_score  is_chimera  variance  bimodality
        separation  balance  breakpoint  r  theta  theta_deg

    The last three columns (r, theta, theta_deg) are the BiosphereAtlas
    coordinates — the Trojan horse that gets the coordinate system
    into every pipeline that uses this tool.

    Args:
        results: List of result dictionaries
        output: Output file path (None = stdout)
    """
    if output:
        f = open(output, "w")
    else:
        f = sys.stdout

    try:
        # Header
        columns = [
            "sequence_id",
            "length",
            "chimera_score",
            "is_chimera",
            "confidence",
            "variance",
            "bimodality",
            "separation",
            "balance",
            "breakpoint",
            "atlas_r",
            "atlas_theta",
            "atlas_theta_deg",
            "kappa",
        ]
        f.write("\t".join(columns) + "\n")

        # Rows
        for r in results:
            row = [
                r["sequence_id"],
                str(r["length"]),
                f"{r['chimera_score']:.6f}",
                str(r["is_chimera"]),
                f"{r['confidence']:.4f}",
                f"{r['variance']:.6f}",
                f"{r['bimodality']:.4f}",
                f"{r['separation']:.4f}",
                f"{r['balance']:.4f}",
                str(r.get("breakpoint", "NA")),
                f"{r['coordinate']['r']:.6f}",
                f"{r['coordinate']['theta']:.6f}",
                f"{r['coordinate']['theta_degrees']:.2f}",
                str(r['coordinate']['kappa']),
            ]
            f.write("\t".join(row) + "\n")
    finally:
        if output and f is not sys.stdout:
            f.close()


def write_results_json(
    results: List[dict],
    output: Optional[str] = None,
):
    """Write results as JSON (one object per line, JSONL format)."""
    import json

    if output:
        f = open(output, "w")
    else:
        f = sys.stdout

    try:
        for r in results:
            f.write(json.dumps(r) + "\n")
    finally:
        if output and f is not sys.stdout:
            f.close()
