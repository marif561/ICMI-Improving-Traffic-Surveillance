"""Rolling prediction: average class probabilities over the last K frames to stop label flickering."""
from __future__ import annotations

from collections import deque
from typing import Iterable, List

import numpy as np


class RollingPredictor:
    """Keeps the probability vectors of the last ``window`` frames and returns their mean."""

    def __init__(self, window: int = 32):
        if window < 1:
            raise ValueError("window must be >= 1")
        self.buffer: deque = deque(maxlen=window)

    def reset(self) -> None:
        self.buffer.clear()

    def update(self, probs: Iterable[float]) -> np.ndarray:
        self.buffer.append(np.asarray(probs, dtype=float))
        return np.mean(self.buffer, axis=0)

    def label(self, probs: Iterable[float]) -> int:
        return int(np.argmax(self.update(probs)))


def smooth_sequence(frame_probs: np.ndarray, window: int) -> List[int]:
    """Smoothed label for every frame of a (T, C) probability array."""
    rp = RollingPredictor(window)
    return [rp.label(p) for p in frame_probs]


def flicker_rate(labels: List[int]) -> float:
    """Fraction of consecutive frames whose label changes (lower is steadier)."""
    if len(labels) < 2:
        return 0.0
    return sum(a != b for a, b in zip(labels, labels[1:])) / (len(labels) - 1)
