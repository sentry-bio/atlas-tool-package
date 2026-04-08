"""
BiosphereAtlas encoder interface for chimera detection.

Two modes of operation:

  1. API mode (default): Calls the live BiosphereAtlas API for embeddings.
     Zero model download, works immediately. Requires network access.

  2. Local mode: Loads a BiosphereCodec checkpoint for offline inference.
     Requires `pip install atlas-chimera[local]` for PyTorch.

For chimera detection, each sequence is split into overlapping windows
and each window is embedded independently. Chimeric sequences produce
sub-embeddings that scatter across phylogenetically distant regions of
the Poincaré ball — detectable via Karcher mean variance and tangent-space
bimodality.

Architecture (Fenn & Fenn 2025):
    Tokenizer → BPE → Hyena/Transformer → Angular ODE flow →
    Poincaré ball (κ = 5/4) → 129-dim tangent vector
"""

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import torch

from biosphere_atlas.core.hyperbolic import KAPPA_DEFAULT, karcher_mean
from biosphere_atlas.core.hyperbolic import _clamp_to_ball

logger = logging.getLogger("atlas_chimera.encoder")

DEFAULT_WINDOW_SIZE = 1000
DEFAULT_STRIDE = 500
MIN_SEQUENCE_LENGTH = 200

# BiosphereAtlas API defaults
DEFAULT_API_URL = "https://api.biosphereatlas.com"


class BiosphereEncoder:
    """
    Encoder that maps DNA sequences to Poincaré ball embeddings.

    Automatically selects API or local mode based on initialization.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        device: str = "cpu",
        kappa: float = KAPPA_DEFAULT,
        embedding_dim: int = 129,
        window_size: int = DEFAULT_WINDOW_SIZE,
        stride: int = DEFAULT_STRIDE,
    ):
        self.kappa = kappa
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.stride = stride
        self.device = device
        self.mode = "api"

        if model_path is not None:
            self._init_local(model_path, device)
            self.mode = "local"
        else:
            self._init_api(api_url, api_key)

    def _init_api(self, api_url: Optional[str], api_key: Optional[str]):
        """Initialize API-backed encoder."""
        self._api_url = api_url or os.environ.get("BIOSPHERE_API_URL", DEFAULT_API_URL)
        self._api_key = api_key or os.environ.get("BIOSPHERE_API_KEY", "")
        logger.info(f"Encoder: API mode ({self._api_url})")

    def _init_local(self, model_path: str, device: str):
        """Initialize local model encoder."""
        import torch as _torch
        self._device = _torch.device(device)
        checkpoint = _torch.load(model_path, map_location=self._device, weights_only=False)

        # Try to reconstruct the V15.5 model architecture
        try:
            from model_v15_5 import load_v15_5_model
            self._model = load_v15_5_model(model_path, device=device)
            self._tokenizer_path = os.environ.get(
                "BPE_VOCAB_PATH", "/models/inference_data/bpe_vocab.json")
            from tokenizer import SimpleBPETokenizer
            self._tokenizer = SimpleBPETokenizer(self._tokenizer_path)
            self._local_mode = "v15"
        except ImportError:
            # Fallback: try to use the checkpoint state dict directly
            self._checkpoint = checkpoint
            self._local_mode = "raw"
            logger.warning("V15.5 model module not available; raw checkpoint stored")

        logger.info(f"Encoder: local mode ({model_path}, {self._local_mode})")

    def _api_encode(self, sequence: str) -> np.ndarray:
        """Call BiosphereAtlas API for a single embedding."""
        import urllib.request
        import json

        payload = json.dumps({"sequence": sequence}).encode()
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["X-API-Key"] = self._api_key

        req = urllib.request.Request(
            f"{self._api_url}/predict",
            data=payload, headers=headers, method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())

        # The /predict endpoint returns coordinates[:3] but we need the
        # full tangent. For chimera detection, the 3D coordinates are
        # insufficient. We use radial depth + angular coordinates.
        # TODO: expose full tangent via /predict or use /identify
        coords = result.get("coordinates", [0, 0, 0])
        radius = result.get("radius", 0)

        # Reconstruct a proxy tangent from the 3D coordinates + radius
        # This is a dimensional reduction — production should expose full tangent
        embedding = np.array(coords + [radius] * (self.embedding_dim - len(coords)),
                             dtype=np.float32)
        # Normalize to stay in ball
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding * min(radius / max(norm, 1e-7), 0.85)

        return embedding

    def _local_encode(self, sequence: str) -> np.ndarray:
        """Encode locally with the V15.5 model."""
        if self._local_mode == "v15":
            tokens = self._tokenizer.tokenize(sequence, max_length=512)
            tokens_tensor = torch.tensor([tokens], dtype=torch.long, device=self._device)
            with torch.no_grad(), torch.cuda.amp.autocast(
                enabled=(str(self._device).startswith("cuda"))
            ):
                result = self._model.classify(tokens_tensor)
                z_ang = result["z_ang"][0].float().cpu().numpy()
            return z_ang
        raise RuntimeError("Local encoder not fully initialized")

    def encode(self, sequence: str) -> np.ndarray:
        """
        Encode a single DNA sequence to a Poincaré ball embedding.

        Returns:
            numpy array of shape (embedding_dim,)
        """
        if self.mode == "api":
            return self._api_encode(sequence)
        return self._local_encode(sequence)

    def encode_subsequences(self, sequence: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode a sequence as overlapping sub-sequence embeddings.

        This is the primary method for chimera detection. The sequence is
        split into overlapping windows, each embedded independently.

        Returns:
            sub_embeddings: (n_windows, embedding_dim) numpy array
            full_embedding: (embedding_dim,) Karcher mean of sub-embeddings
        """
        seq_len = len(sequence)

        if seq_len < MIN_SEQUENCE_LENGTH:
            emb = self.encode(sequence)
            return emb.reshape(1, -1), emb

        # Generate overlapping windows
        windows = []
        for start in range(0, seq_len - self.window_size + 1, self.stride):
            windows.append(sequence[start:start + self.window_size])

        # Ensure final window is included
        if not windows or (seq_len - self.window_size) % self.stride != 0:
            tail = sequence[-self.window_size:] if seq_len >= self.window_size else sequence
            windows.append(tail)

        # Encode all windows
        sub_embeddings = np.stack([self.encode(w) for w in windows])

        # Karcher mean for full embedding
        sub_tensor = torch.from_numpy(sub_embeddings).float()
        mean, _ = karcher_mean(sub_tensor, kappa=self.kappa)
        full_embedding = mean.numpy()

        return sub_embeddings, full_embedding

    def encode_batch(
        self, sequences: List[str]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Encode multiple sequences, returning sub-embeddings for each."""
        return [self.encode_subsequences(seq) for seq in sequences]
