"""Turn a text file into training batches."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_text(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run `python scripts/download_data.py` first, "
            "or point --data at any .txt file you like."
        )
    return p.read_text(encoding="utf-8")


def train_val_split(ids: list[int], val_frac: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Hold out the last val_frac of the data to measure generalization."""
    arr = np.array(ids, dtype=np.int64)
    n = int(len(arr) * (1 - val_frac))
    return arr[:n], arr[n:]


def get_batch(data: np.ndarray, block_size: int, batch_size: int,
              rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random windows of the data. y is x shifted one token to the right:
    at every position, the target is simply the next token."""
    if len(data) <= block_size:
        raise ValueError(f"data has {len(data)} tokens; need more than block_size={block_size}")
    starts = rng.integers(0, len(data) - block_size, size=batch_size)
    x = np.stack([data[s: s + block_size] for s in starts])
    y = np.stack([data[s + 1: s + 1 + block_size] for s in starts])
    return x, y
