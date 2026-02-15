"""
Tests for the main placement pipeline.
"""

import math
import tempfile
from pathlib import Path

import pytest
import torch

from biosphere_atlas.place.calibrator import PlacementCalibrator
from biosphere_atlas.place.encoder import BiosphereEncoder, KmerEncoder
from biosphere_atlas.place.hyperbolic import KAPPA_DEFAULT, _clamp_to_ball
from biosphere_atlas.place.place import place_embeddings, place_sequences, placement_summary
from biosphere_atlas.place.reference import Rank, ReferenceDB


KAPPA = KAPPA_DEFAULT
DIM = 64  # Match KmerEncoder default


def _ball_point(dim: int = DIM, scale: float = 0.5) -> torch.Tensor:
    p = torch.randn(dim) * scale / math.sqrt(KAPPA)
    return _clamp_to_ball(p, KAPPA)


def _make_reference(n_species: int = 5) -> ReferenceDB:
    """Build a reference DB with k-mer encoded prototypes."""
    encoder = KmerEncoder(k=4, embedding_dim=DIM, kappa=KAPPA)
    lineage_templates = [
        ("d__Bacteria", "p__Proteobacteria", "c__Gamma",
         "o__Enterobacterales", "f__Enterobacteriaceae",
         "g__Escherichia", "s__Species_{idx}"),
    ]
    lineages = []
    embeddings = []
    for i in range(n_species):
        lineage = tuple(
            t.format(idx=i) for t in lineage_templates[0]
        )
        lineages.append(lineage)
        # Generate a pseudo-sequence for this species
        seq = "ATCG" * 100 + "A" * i  # Slightly different per species
        embeddings.append(encoder.encode(seq))

    return ReferenceDB.from_lineages(
        lineages, torch.stack(embeddings), kappa=KAPPA
    )


def _make_fasta(tmp_path: Path, n_seqs: int = 3) -> Path:
    """Create a temporary FASTA file."""
    fasta = tmp_path / "test.fasta"
    with open(fasta, "w") as f:
        for i in range(n_seqs):
            f.write(f">seq_{i}\n")
            # Generate random DNA
            import random
            random.seed(i)
            seq = "".join(random.choice("ACGT") for _ in range(500))
            f.write(seq + "\n")
    return fasta


class TestPlaceEmbeddings:
    def test_basic(self):
        ref = _make_reference()
        queries = torch.stack([_ball_point() for _ in range(4)])
        results = place_embeddings(queries, ref, top_k=3, kappa=KAPPA)

        assert len(results) == 4
        for r in results:
            assert r.zone is not None
            assert len(r.candidates) <= 3

    def test_with_ids(self):
        ref = _make_reference()
        queries = torch.stack([_ball_point() for _ in range(2)])
        ids = ["query_A", "query_B"]
        results = place_embeddings(queries, ref, sequence_ids=ids)

        assert results[0].sequence_id == "query_A"
        assert results[1].sequence_id == "query_B"


class TestPlaceSequences:
    def test_end_to_end(self, tmp_path):
        ref = _make_reference()
        fasta = _make_fasta(tmp_path, n_seqs=3)
        output = tmp_path / "results.tsv"

        results = place_sequences(
            fasta_path=fasta,
            reference=ref,
            output_path=output,
            output_format="tsv",
            kappa=KAPPA,
        )

        assert len(results) == 3
        assert output.exists()

        # Read back and verify
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 4  # header + 3 results
        assert "sequence_id" in lines[0]

    def test_json_output(self, tmp_path):
        ref = _make_reference()
        fasta = _make_fasta(tmp_path, n_seqs=2)
        output = tmp_path / "results.jsonl"

        place_sequences(
            fasta_path=fasta,
            reference=ref,
            output_path=output,
            output_format="json",
            kappa=KAPPA,
        )

        assert output.exists()
        import json
        lines = output.read_text().strip().split("\n")
        for line in lines:
            obj = json.loads(line)
            assert "sequence_id" in obj

    def test_jplace_output(self, tmp_path):
        ref = _make_reference()
        fasta = _make_fasta(tmp_path, n_seqs=2)
        output = tmp_path / "placements.jplace"

        place_sequences(
            fasta_path=fasta,
            reference=ref,
            output_path=output,
            output_format="jplace",
            kappa=KAPPA,
        )

        assert output.exists()
        import json
        with open(output) as f:
            jplace = json.load(f)
        assert jplace["version"] == 3
        assert "placements" in jplace
        assert len(jplace["placements"]) == 2


class TestPlacementSummary:
    def test_basic_summary(self):
        ref = _make_reference()
        queries = torch.stack([_ball_point() for _ in range(10)])
        results = place_embeddings(queries, ref)

        summary = placement_summary(results)
        assert summary["total"] == 10
        assert "zones" in summary
        assert "distance_mean" in summary

    def test_empty_summary(self):
        summary = placement_summary([])
        assert summary["total"] == 0


class TestEncoder:
    def test_kmer_encoder(self):
        enc = KmerEncoder(k=4, embedding_dim=DIM, kappa=KAPPA)
        emb = enc.encode("ATCGATCGATCGATCG" * 10)
        assert emb.shape == (DIM,)
        assert emb.norm().item() < 1.0 / math.sqrt(KAPPA)

    def test_kmer_batch(self):
        enc = KmerEncoder(k=4, embedding_dim=DIM, kappa=KAPPA)
        seqs = ["ATCG" * 50, "GCTA" * 50, "AAAA" * 50]
        embs = enc.encode_batch(seqs)
        assert embs.shape == (3, DIM)

    def test_different_sequences_different_embeddings(self):
        enc = KmerEncoder(k=4, embedding_dim=DIM, kappa=KAPPA)
        e1 = enc.encode("ATCG" * 100)
        e2 = enc.encode("GGCC" * 100)
        # Different sequences should produce different embeddings
        assert not torch.allclose(e1, e2, atol=1e-3)

    def test_biosphere_encoder_kmer_mode(self):
        enc = BiosphereEncoder(embedding_dim=DIM, kappa=KAPPA)
        assert enc.mode == "kmer"
        emb = enc.encode("ATCGATCG" * 50)
        assert emb.shape == (DIM,)
