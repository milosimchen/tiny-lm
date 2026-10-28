"""Tokenizers mapping text to integer ids and back.

* CharTokenizer: one token per character. Small vocabulary (65 for
  Shakespeare) and no training cost.
* BPETokenizer: byte-level byte-pair encoding, as used by GPT-2. Starting from
  raw bytes, it repeatedly merges the most frequent adjacent pair into a new
  token, so common chunks become single ids and sequences get shorter.

Both are dependency-free so the tokenizer can be inspected and tested in
isolation from the model.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


class CharTokenizer:
    kind = "char"

    def __init__(self, chars: list[str]):
        self.chars = chars
        self.stoi = {c: i for i, c in enumerate(chars)}

    @classmethod
    def train(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.chars[i] for i in ids)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "chars": self.chars}


class BPETokenizer:
    """Byte-level BPE. Ids 0-255 are raw bytes; each merge adds one new id."""

    kind = "bpe"

    def __init__(self, merges: list[tuple[int, int]]):
        self.merges = merges
        self.ranks = {pair: 256 + i for i, pair in enumerate(merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in self.ranks.items():
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]

    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    @classmethod
    def train(cls, text: str, vocab_size: int = 512, verbose: bool = False) -> "BPETokenizer":
        assert vocab_size >= 256, "vocab_size must be at least 256 (the raw bytes)"
        ids = list(text.encode("utf-8"))
        merges: list[tuple[int, int]] = []
        for step in range(vocab_size - 256):
            counts = Counter(zip(ids, ids[1:]))
            if not counts:
                break
            pair, freq = counts.most_common(1)[0]
            if freq < 2:  # nothing left worth merging
                break
            new_id = 256 + step
            ids = cls._merge(ids, pair, new_id)
            merges.append(pair)
            if verbose and step % 50 == 0:
                print(f"merge {step}: {pair} -> {new_id} (seen {freq}x)")
        return cls(merges)

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            # Apply the earliest-learned merge present, exactly as in training.
            pairs = set(zip(ids, ids[1:]))
            pair = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            ids = self._merge(ids, pair, self.ranks[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        return {"kind": self.kind, "merges": [list(p) for p in self.merges]}


def save_tokenizer(tok, path: str | Path) -> None:
    Path(path).write_text(json.dumps(tok.to_dict()))


def load_tokenizer(path: str | Path):
    d = json.loads(Path(path).read_text())
    if d["kind"] == "char":
        return CharTokenizer(d["chars"])
    if d["kind"] == "bpe":
        return BPETokenizer([tuple(p) for p in d["merges"]])
    raise ValueError(f"unknown tokenizer kind: {d['kind']}")
