"""Minimal synthetic dataset inspired by STLnet air quality examples."""

from __future__ import annotations

from typing import Tuple
import numpy as np

class SyntheticSTLNetDataset:
    """Generate a bounded sinusoidal time-series for STLnet-style demos.

    Each item is a '(time, value)' pair where 'time' is in '[0, 1]' and
    'value' lies roughly in '[0, 1]' with optional Gaussian noise.
    """

    def __init__(self, length: int = 100, noise: float = 0.05) -> None:
        t = np.linspace(0.0, 1.0, num=length, dtype=float)
        clean = 0.5 * (np.sin(2 * np.pi * t) + 1.0)
        self.data = np.stack([t, clean + noise * np.random.randn(length)], axis=1)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Tuple[float, float]:
        t, v = self.data[idx]
        return float(t), float(v)

__all__ = ["SyntheticSTLNetDataset"]
