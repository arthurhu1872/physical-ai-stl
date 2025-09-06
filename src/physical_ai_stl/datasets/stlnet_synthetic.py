from __future__ import annotations

import numpy as np


class SyntheticSTLNetDataset:
    """
    Lightweight synthetic 1‑D signal used by tests.

    Attributes
    ----------
    t : np.ndarray
        Shape (length,) time stamps in [0, 1].
    y : np.ndarray
        Shape (length,) signal values.
    """

    def __init__(self, length: int = 100, noise: float = 0.05) -> None:
        if length < 0:
            raise ValueError("length must be non-negative")

        self.t = np.linspace(0.0, 1.0, num=length, dtype=float)
        # Smooth deterministic base signal (bounded in [-1.5, 1.5]).
        base = np.sin(2.0 * np.pi * self.t) + 0.5 * np.cos(4.0 * np.pi * self.t)

        if noise:
            # Use legacy global RNG so tests can use np.random.get_state()
            # to make construction deterministic across calls.
            self.y = base + noise * np.random.randn(length).astype(float)
        else:
            self.y = base

    def __len__(self) -> int:
        return self.t.shape[0]

    def __getitem__(self, idx: int) -> tuple[float, float]:
        return float(self.t[idx]), float(self.y[idx])
