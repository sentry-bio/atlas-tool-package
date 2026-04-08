"""
BiosphereCodec encoder wrapper.
================================

Wraps the BiosphereCodec / BiosphereAtlas encoder to map raw DNA sequences
to Poincaré ball embeddings.  When no trained model is available, falls
back to a lightweight k-mer frequency encoder for development and testing.

Same pattern as atlas-chimera's encoder.py.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from biosphere_atlas.core.hyperbolic import KAPPA_DEFAULT
from biosphere_atlas.core.hyperbolic import _clamp_to_ball


# ── K-mer frequency encoder (development proxy) ─────────────────────────────

class KmerEncoder:
    """
    Lightweight k-mer frequency encoder for development.

    Maps DNA sequences to a fixed-dimensional Poincaré ball embedding
    via k-mer frequencies → L2-normalized → scaled to ball interior.

    This is a placeholder.  In production, BiosphereCodec provides
    the trained encoder with full hierarchical pooling.
    """

    NUCLEOTIDE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}

    def __init__(
        self,
        k: int = 4,
        embedding_dim: int = 64,
        kappa: float = KAPPA_DEFAULT,
    ):
        self.k = k
        self.embedding_dim = embedding_dim
        self.kappa = kappa
        self.vocab_size = 4 ** k

        # Deterministic projection matrix (seeded for reproducibility)
        gen = torch.Generator().manual_seed(42)
        self._proj = torch.randn(self.vocab_size, embedding_dim, generator=gen)
        self._proj = self._proj / self._proj.norm(dim=0, keepdim=True)

    def _kmer_frequencies(self, sequence: str) -> Tensor:
        """Compute normalized k-mer frequency vector."""
        freq = torch.zeros(self.vocab_size)
        seq = sequence.upper().replace("N", "A")  # Simple N handling

        for i in range(len(seq) - self.k + 1):
            kmer = seq[i : i + self.k]
            idx = 0
            valid = True
            for ch in kmer:
                if ch not in self.NUCLEOTIDE_MAP:
                    valid = False
                    break
                idx = idx * 4 + self.NUCLEOTIDE_MAP[ch]
            if valid:
                freq[idx] += 1

        total = freq.sum()
        if total > 0:
            freq = freq / total
        return freq

    def encode(self, sequence: str) -> Tensor:
        """
        Encode a single DNA sequence to a Poincaré ball embedding.

        Args:
            sequence: DNA string (A/C/G/T/N).

        Returns:
            (D,) tensor inside the Poincaré ball.
        """
        freq = self._kmer_frequencies(sequence)
        # Project to embedding dimension
        emb = freq @ self._proj
        # Normalize and scale to ball interior (0.7 of max radius)
        emb = F.normalize(emb, dim=-1) * 0.7 / (self.kappa ** 0.5)
        return _clamp_to_ball(emb, self.kappa)

    def encode_batch(self, sequences: List[str]) -> Tensor:
        """
        Encode a batch of sequences.

        Args:
            sequences: list of DNA strings.

        Returns:
            (B, D) tensor of embeddings.
        """
        return torch.stack([self.encode(seq) for seq in sequences])


# ── BPE tokenizer (V10/V13-compatible) ───────────────────────────────────────

class SimpleBPETokenizer:
    """
    Greedy longest-match BPE tokenizer compatible with V10/V13 preprocessing.

    Supports vocab files stored as:
      - token->id dict
      - list[str] of tokens
      - {"vocab": [...]}
    """

    def __init__(self, vocab_path: str):
        with open(vocab_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            self.vocab = {t: i for i, t in enumerate(data)}
        elif isinstance(data, dict):
            if "vocab" in data and isinstance(data["vocab"], list):
                self.vocab = {t: i for i, t in enumerate(data["vocab"])}
            else:
                self.vocab = data
        else:
            raise ValueError(f"Unsupported BPE vocab format: {type(data)}")

        self.unk_id = int(self.vocab.get("[UNK]", self.vocab.get("<UNK>", 1)))
        self.pad_id = int(self.vocab.get("[PAD]", self.vocab.get("<PAD>", 0)))
        self.max_token_len = max((len(t) for t in self.vocab.keys()), default=1)

    def tokenize(self, sequence: str, max_length: int = 512) -> List[int]:
        sequence = sequence.upper().replace("N", "A")
        tokens: List[int] = []
        i = 0
        n = len(sequence)

        while i < n and len(tokens) < max_length:
            matched = False
            max_len = min(self.max_token_len, n - i)
            for l in range(max_len, 0, -1):
                sub = sequence[i : i + l]
                if sub in self.vocab:
                    tokens.append(int(self.vocab[sub]))
                    i += l
                    matched = True
                    break
            if not matched:
                tokens.append(self.unk_id)
                i += 1

        if len(tokens) < max_length:
            tokens.extend([self.pad_id] * (max_length - len(tokens)))
        else:
            tokens = tokens[:max_length]
        return tokens


# ── BiosphereCodec wrapper ───────────────────────────────────────────────────

class BiosphereEncoder:
    """
    Wrapper for the BiosphereCodec encoder.

    Accepts either:
    - A trained BiosphereCodec / BiosphereAtlas model
    - A path to a saved model checkpoint
    - None (falls back to KmerEncoder for development)
    """

    def __init__(
        self,
        model: Optional[object] = None,
        model_path: Optional[str] = None,
        embedding_dim: int = 64,
        kappa: float = KAPPA_DEFAULT,
        device: str = "cpu",
        tokenizer_path: Optional[str] = None,
        max_tokens: int = 512,
    ):
        self.kappa = kappa
        self.device = torch.device(device)
        self.embedding_dim = embedding_dim
        self.max_tokens = max_tokens
        self._codec_mode = "none"
        self._tokenizer = None

        if model is not None:
            self._model = model
            self._mode = "codec"
            self._codec_mode = "preloaded"
        elif model_path is not None:
            self._model = self._load_model(model_path)
            self._mode = "codec"
            if self._codec_mode == "v13_encoder":
                # Allow explicit override, else try common server/local locations.
                vocab_candidates = [
                    tokenizer_path,
                    "/zfs_raid/SentryBio/working/inference_data/bpe_vocab.json",
                    "/zfs_raid/SentryBio/working/bpe_vocab_8192.json",
                    str(Path(model_path).resolve().parent / "bpe_vocab.json"),
                ]
                vocab_path = next((p for p in vocab_candidates if p and os.path.exists(p)), None)
                if vocab_path is None:
                    raise FileNotFoundError(
                        "V13 checkpoint detected but no BPE vocab found. "
                        "Pass tokenizer_path explicitly."
                    )
                self._tokenizer = SimpleBPETokenizer(vocab_path)
        else:
            self._model = KmerEncoder(
                k=4, embedding_dim=embedding_dim, kappa=kappa
            )
            self._mode = "kmer"

    def _load_model(self, path: str) -> object:
        """Load either a direct codec object or a V13 checkpoint state dict."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if hasattr(checkpoint, "encode"):
            self._codec_mode = "preloaded"
            return checkpoint

        # V13/V12-style checkpoint: {'model_state_dict': ...}
        state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if isinstance(state, dict) and any(k.startswith("encoder.") for k in state.keys()):
            self._codec_mode = "v13_encoder"
            return self._load_v13_encoder_from_state(state)

        raise ValueError(
            f"Cannot load model from {path}: unsupported checkpoint format for atlas-place encoder."
        )

    def _load_v13_encoder_from_state(self, state: dict) -> object:
        """Instantiate BiosphereAtlasMaxEntropy and load encoder.* weights from V13 checkpoint."""
        repo_root = Path(__file__).resolve().parents[3]
        candidates = [
            "/zfs_raid/SentryBio/5k_test_genomes",
            str(repo_root),
        ]
        for c in candidates:
            if c not in sys.path:
                sys.path.insert(0, c)

        from Biosphere_codec.BiosphereAtlasMaxEntropy import BiosphereAtlasMaxEntropy  # type: ignore

        # Infer dims from checkpoint
        embed_w = state.get("encoder.encoder.embed.weight")
        if embed_w is None:
            raise ValueError("V13 checkpoint missing encoder embedding weights")
        vocab_size = int(embed_w.shape[0])
        to_latent_w = state.get("encoder.encoder.to_latent.3.weight")
        latent_dim = int(to_latent_w.shape[0]) if to_latent_w is not None else 128

        model = BiosphereAtlasMaxEntropy(vocab_size=vocab_size, latent_dim=latent_dim).to(self.device)

        enc_state = {}
        for k, v in state.items():
            if k.startswith("encoder."):
                nk = k[len("encoder.") :]
                if "curvature_history" in nk:
                    continue
                enc_state[nk] = v
        model.load_state_dict(enc_state, strict=False)
        model.eval()
        return model

    @property
    def mode(self) -> str:
        return self._mode

    def encode(self, sequence: str) -> Tensor:
        """Encode a single DNA sequence."""
        if self._mode == "kmer":
            return self._model.encode(sequence)

        # Generic object with .encode API
        if self._codec_mode in {"preloaded", "none"} and hasattr(self._model, "encode"):
            with torch.no_grad():
                embedding = self._model.encode(sequence)
                if isinstance(embedding, dict):
                    embedding = embedding.get("z", embedding.get("mu"))
                return _clamp_to_ball(embedding.squeeze(0), self.kappa)

        # V13 encoder checkpoint path
        if self._codec_mode == "v13_encoder":
            if self._tokenizer is None:
                raise RuntimeError("V13 mode requires tokenizer, but tokenizer is missing")
            token_ids = self._tokenizer.tokenize(sequence, max_length=self.max_tokens)
            tokens = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            with torch.no_grad():
                out = self._model({"tokens": tokens})
                embedding = out.get("embeddings", out.get("z"))
                c = float(self._model.manifold.k.item()) if hasattr(self._model, "manifold") else self.kappa
                return _clamp_to_ball(embedding.squeeze(0).float(), c)

        with torch.no_grad():
            embedding = self._model.encode(sequence)
            if isinstance(embedding, dict):
                embedding = embedding.get("z", embedding.get("mu"))
            return _clamp_to_ball(embedding.squeeze(0), self.kappa)

    def encode_batch(self, sequences: List[str]) -> Tensor:
        """Encode a batch of sequences."""
        if self._mode == "kmer":
            return self._model.encode_batch(sequences)

        if self._codec_mode == "v13_encoder":
            if self._tokenizer is None:
                raise RuntimeError("V13 mode requires tokenizer, but tokenizer is missing")
            batch_ids = [self._tokenizer.tokenize(s, max_length=self.max_tokens) for s in sequences]
            tokens = torch.tensor(batch_ids, dtype=torch.long, device=self.device)
            with torch.no_grad():
                out = self._model({"tokens": tokens})
                embeddings = out.get("embeddings", out.get("z")).float()
                c = float(self._model.manifold.k.item()) if hasattr(self._model, "manifold") else self.kappa
                return _clamp_to_ball(embeddings, c)

        with torch.no_grad():
            embeddings = torch.stack([self.encode(seq) for seq in sequences])
            return embeddings

    def encode_token_batch(self, tokens: Tensor) -> Tensor:
        """
        Encode a pre-tokenized batch of token IDs.

        Args:
            tokens: (B, L) long tensor of token IDs.

        Returns:
            (B, D) Poincare-ball embeddings.
        """
        if self._mode == "kmer":
            raise RuntimeError("Token-batch encoding is unavailable in k-mer mode.")

        tokens = tokens.to(self.device)

        # V13 direct path
        if self._codec_mode == "v13_encoder":
            with torch.no_grad():
                out = self._model({"tokens": tokens})
                embeddings = out.get("embeddings", out.get("z")).float()
                c = float(self._model.manifold.k.item()) if hasattr(self._model, "manifold") else self.kappa
                return _clamp_to_ball(embeddings, c)

        # Generic codec fallback (if model exposes a token-batch forward API)
        with torch.no_grad():
            if callable(self._model):
                out = self._model({"tokens": tokens})
                if isinstance(out, dict):
                    emb = out.get("embeddings", out.get("z", out.get("mu")))
                else:
                    emb = out
                return _clamp_to_ball(emb.float(), self.kappa)
            raise RuntimeError("Loaded codec does not support token-batch encoding.")
