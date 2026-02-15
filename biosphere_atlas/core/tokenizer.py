"""Tokenizer utilities shared by all biosphere-atlas tools."""

from __future__ import annotations

import json
from typing import List


class SimpleBPETokenizer:
    """
    Greedy longest-match BPE tokenizer compatible with V10/V13 preprocessing.
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

