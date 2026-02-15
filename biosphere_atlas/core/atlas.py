"""Unified checkpoint loader and encoder interface for biosphere-atlas tools."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from biosphere_atlas.core.coordinates import BiosphereCoordinate, extract_coordinate
from biosphere_atlas.core.hyperbolic import KAPPA_DEFAULT, _clamp_to_ball, karcher_mean
from biosphere_atlas.core.tokenizer import SimpleBPETokenizer


class _KmerEncoder:
    """Lightweight fallback encoder for development and tests."""

    NUCLEOTIDE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}

    def __init__(self, k: int = 4, embedding_dim: int = 128, kappa: float = KAPPA_DEFAULT):
        self.k = k
        self.embedding_dim = embedding_dim
        self.kappa = kappa
        self.vocab_size = 4 ** k
        gen = torch.Generator().manual_seed(42)
        self._proj = torch.randn(self.vocab_size, embedding_dim, generator=gen)
        self._proj = self._proj / self._proj.norm(dim=0, keepdim=True)

    def _kmer_frequencies(self, sequence: str) -> Tensor:
        freq = torch.zeros(self.vocab_size)
        seq = sequence.upper().replace("N", "A")
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
        freq = self._kmer_frequencies(sequence)
        emb = freq @ self._proj
        emb = F.normalize(emb, dim=-1) * 0.7 / (self.kappa ** 0.5)
        return _clamp_to_ball(emb, self.kappa)

    def encode_batch(self, sequences: List[str]) -> Tensor:
        return torch.stack([self.encode(seq) for seq in sequences])


class Atlas:
    """The shared Atlas object that all tools consume."""

    DEFAULT_WINDOW_SIZE = 1000
    DEFAULT_STRIDE = 500
    MIN_SEQUENCE_LENGTH = 200

    def __init__(
        self,
        model: Optional[object] = None,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        embedding_dim: int = 129,
        kappa: float = KAPPA_DEFAULT,
        device: str = "cpu",
        max_tokens: int = 512,
        window_size: int = DEFAULT_WINDOW_SIZE,
        stride: int = DEFAULT_STRIDE,
    ):
        self.device = torch.device(device)
        self._fallback_kappa = float(kappa)
        self.embedding_dim = embedding_dim
        self.max_tokens = max_tokens
        self.window_size = window_size
        self.stride = stride
        self._codec_mode = "none"
        self._tokenizer: Optional[SimpleBPETokenizer] = None

        if model is not None:
            self._model = model
            self._mode = "codec"
            self._codec_mode = "preloaded"
        elif model_path is not None:
            self._model = self._load_model(model_path)
            self._mode = "codec"
            if self._codec_mode == "v13_encoder":
                vocab_candidates = [
                    tokenizer_path,
                    "/zfs_raid/SentryBio/working/inference_data/bpe_vocab.json",
                    "/zfs_raid/SentryBio/working/bpe_vocab_8192.json",
                    str(Path(model_path).resolve().parent / "bpe_vocab.json"),
                ]
                vocab_path = next((p for p in vocab_candidates if p and os.path.exists(p)), None)
                if vocab_path is None:
                    raise FileNotFoundError("V13 checkpoint detected but no BPE vocab found.")
                self._tokenizer = SimpleBPETokenizer(vocab_path)
        else:
            self._model = _KmerEncoder(k=4, embedding_dim=embedding_dim, kappa=kappa)
            self._mode = "kmer"

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        tokenizer_path: Optional[str] = None,
        device: str = "cpu",
        max_tokens: int = 512,
        window_size: int = DEFAULT_WINDOW_SIZE,
        stride: int = DEFAULT_STRIDE,
    ) -> "Atlas":
        return cls(
            model_path=path,
            tokenizer_path=tokenizer_path,
            device=device,
            max_tokens=max_tokens,
            window_size=window_size,
            stride=stride,
        )

    @property
    def kappa(self) -> float:
        if hasattr(self._model, "manifold") and hasattr(self._model.manifold, "k"):
            return float(self._model.manifold.k.item())
        return self._fallback_kappa

    def _load_model(self, path: str) -> object:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if hasattr(checkpoint, "encode"):
            self._codec_mode = "preloaded"
            return checkpoint
        state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if isinstance(state, dict) and any(k.startswith("encoder.") for k in state.keys()):
            self._codec_mode = "v13_encoder"
            return self._load_v13_encoder_from_state(state)
        raise ValueError(f"Unsupported checkpoint format: {path}")

    def _load_v13_encoder_from_state(self, state: dict) -> object:
        repo_root = Path(__file__).resolve().parents[3]
        for c in ["/zfs_raid/SentryBio/5k_test_genomes", str(repo_root)]:
            if c not in sys.path:
                sys.path.insert(0, c)
        from Biosphere_codec.BiosphereAtlasMaxEntropy import BiosphereAtlasMaxEntropy  # type: ignore

        embed_w = state.get("encoder.encoder.embed.weight")
        if embed_w is None:
            raise ValueError("Checkpoint missing encoder embedding weights")
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

    def encode(self, sequence: str) -> Tensor:
        if self._mode == "kmer":
            return self._model.encode(sequence)
        if self._codec_mode == "v13_encoder":
            if self._tokenizer is None:
                raise RuntimeError("V13 mode requires tokenizer.")
            token_ids = self._tokenizer.tokenize(sequence, max_length=self.max_tokens)
            tokens = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            with torch.no_grad():
                out = self._model({"tokens": tokens})
                emb = out.get("embeddings", out.get("z")).float().squeeze(0)
                return _clamp_to_ball(emb, self.kappa)

        with torch.no_grad():
            embedding = self._model.encode(sequence)
            if isinstance(embedding, dict):
                embedding = embedding.get("z", embedding.get("mu"))
            return _clamp_to_ball(embedding.squeeze(0), self.kappa)

    def encode_batch(self, sequences: List[str]) -> Tensor:
        if self._mode == "kmer":
            return self._model.encode_batch(sequences)
        if self._codec_mode == "v13_encoder":
            if self._tokenizer is None:
                raise RuntimeError("V13 mode requires tokenizer.")
            batch_ids = [self._tokenizer.tokenize(s, max_length=self.max_tokens) for s in sequences]
            tokens = torch.tensor(batch_ids, dtype=torch.long, device=self.device)
            return self.encode_tokens(tokens)
        with torch.no_grad():
            return torch.stack([self.encode(seq) for seq in sequences])

    def encode_tokens(self, tokens: Tensor) -> Tensor:
        if self._mode == "kmer":
            raise RuntimeError("Token-batch encoding unavailable in k-mer mode.")
        tokens = tokens.to(self.device)
        if self._codec_mode == "v13_encoder":
            with torch.no_grad():
                out = self._model({"tokens": tokens})
                emb = out.get("embeddings", out.get("z")).float()
                return _clamp_to_ball(emb, self.kappa)
        if callable(self._model):
            with torch.no_grad():
                out = self._model({"tokens": tokens})
                if isinstance(out, dict):
                    emb = out.get("embeddings", out.get("z", out.get("mu")))
                else:
                    emb = out
                return _clamp_to_ball(emb.float(), self.kappa)
        raise RuntimeError("Loaded codec does not support token-batch encoding.")

    def encode_subsequences(self, sequence: str) -> Tuple[Tensor, Tensor]:
        seq_len = len(sequence)
        if seq_len < self.MIN_SEQUENCE_LENGTH:
            emb = self.encode(sequence)
            return emb.unsqueeze(0), emb

        windows = []
        for start in range(0, seq_len - self.window_size + 1, self.stride):
            windows.append(sequence[start : start + self.window_size])
        if len(windows) == 0 or (seq_len - self.window_size) % self.stride != 0:
            windows.append(sequence[-self.window_size:] if seq_len >= self.window_size else sequence)

        sub_embeddings = torch.stack([self.encode(w) for w in windows])
        full_embedding, _ = karcher_mean(sub_embeddings, kappa=self.kappa)
        return sub_embeddings, full_embedding

    def coordinate(self, embedding: Tensor) -> BiosphereCoordinate:
        return extract_coordinate(embedding, kappa=self.kappa)

