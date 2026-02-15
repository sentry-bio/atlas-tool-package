"""
Reference DB builder for atlas-place.

Builds a prototype reference directly from a training manifest and V13 checkpoint,
using pre-tokenized genomes for speed and deterministic geometry.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor

from biosphere_atlas.core.atlas import Atlas
from biosphere_atlas.core.hyperbolic import KAPPA_DEFAULT
from biosphere_atlas.place.reference import Rank, ReferenceDB


RANK_NAME_TO_ENUM: Dict[str, Rank] = {
    "domain": Rank.DOMAIN,
    "phylum": Rank.PHYLUM,
    "class": Rank.CLASS,
    "order": Rank.ORDER,
    "family": Rank.FAMILY,
    "genus": Rank.GENUS,
    "species": Rank.SPECIES,
}


def _clean_label(x: str) -> str:
    return str(x or "UNK").strip().replace(" ", "_")


def _lineage_from_row(row: Dict[str, str]) -> Tuple[str, ...]:
    """Create a 7-level lineage tuple with domain-aware IDs to avoid collisions."""
    domain = _clean_label(row.get("domain", "Unknown"))
    phylum = _clean_label(row.get("phylum", "UNK"))
    class_ = _clean_label(row.get("class", "UNK"))
    order = _clean_label(row.get("order", "UNK"))
    family = _clean_label(row.get("family", "UNK"))
    genus = _clean_label(row.get("genus", "UNK"))
    species = _clean_label(row.get("species", "UNK"))
    # Domain scoping keeps same-named lower-rank labels in different domains distinct.
    return (
        f"d__{domain}",
        f"p__{domain}|{phylum}",
        f"c__{domain}|{class_}",
        f"o__{domain}|{order}",
        f"f__{domain}|{family}",
        f"g__{domain}|{genus}",
        f"s__{domain}|{species}",
    )


def _load_manifest_rows(
    manifest_path: str,
    split: str,
    max_samples: int,
    shuffle_rows: bool,
    seed: int,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("domain") == "Viruses":
                continue
            if split and row.get("split") != split:
                continue
            tok_path = row.get("tokenized_path", "")
            if not tok_path or not Path(tok_path).exists():
                continue
            if not row.get("family"):
                continue
            rows.append(row)
    if shuffle_rows:
        random.Random(seed).shuffle(rows)
    if max_samples > 0:
        rows = rows[:max_samples]
    return rows


def _encode_rows_tokenized(
    rows: List[Dict[str, str]],
    encoder: Atlas,
    batch_size: int = 16,
    max_tokens: int = 512,
) -> Tensor:
    out: List[Tensor] = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        arrs = []
        for row in chunk:
            x = np.load(row["tokenized_path"])
            if x.ndim != 1:
                x = x.reshape(-1)
            x = x[:max_tokens]
            if x.shape[0] < max_tokens:
                x = np.pad(x, (0, max_tokens - x.shape[0]), mode="constant", constant_values=0)
            arrs.append(torch.from_numpy(x).long())
        tokens = torch.stack(arrs, dim=0)
        emb = encoder.encode_token_batch(tokens).detach().cpu()
        out.append(emb)
    return torch.cat(out, dim=0) if out else torch.zeros(0, 129)


def build_reference_from_manifest(
    manifest_path: str,
    output_path: str,
    model_path: str,
    tokenizer_path: str | None = None,
    split: str = "train",
    rank: str = "family",
    batch_size: int = 16,
    max_samples: int = 0,
    max_tokens: int = 512,
    device: str = "cpu",
    kappa: float = KAPPA_DEFAULT,
    shuffle_rows: bool = True,
    seed: int = 42,
) -> Dict[str, float]:
    rows = _load_manifest_rows(
        manifest_path,
        split=split,
        max_samples=max_samples,
        shuffle_rows=shuffle_rows,
        seed=seed,
    )
    if not rows:
        raise ValueError("No usable rows found in manifest for requested split.")

    encoder = Atlas(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        device=device,
        kappa=kappa,
        max_tokens=max_tokens,
    )
    embeddings = _encode_rows_tokenized(rows, encoder, batch_size=batch_size, max_tokens=max_tokens)
    if embeddings.shape[0] != len(rows):
        raise RuntimeError("Embedding count does not match row count.")

    leaf_rank = RANK_NAME_TO_ENUM[rank]
    lineages = [_lineage_from_row(r) for r in rows]
    # Use live kappa from model if available.
    live_kappa = kappa
    if hasattr(encoder, "_model") and hasattr(encoder._model, "manifold"):
        live_kappa = float(encoder._model.manifold.k.item())

    db = ReferenceDB.from_lineages(
        lineages=lineages,
        embeddings=embeddings,
        kappa=live_kappa,
        leaf_rank=leaf_rank,
    )
    db.save(output_path)

    summary = db.summary()
    return {
        "n_rows": float(len(rows)),
        "embedding_dim": float(db.embedding_dim),
        "kappa": float(db.kappa),
        "n_prototypes_total": float(db.size),
        "n_family": float(summary.get("family", 0)),
        "n_genus": float(summary.get("genus", 0)),
        "n_species": float(summary.get("species", 0)),
    }


