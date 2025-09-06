from __future__ import annotations

import numpy as np


def linspace_1d(n: int) -> np.ndarray:
    """[0, 1] linspace of length n."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return np.linspace(0.0, 1.0, num=n, dtype=float)


def meshgrid_2d(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    """2‑D meshgrid over [0, 1] × [0, 1]."""
    x = linspace_1d(nx)
    y = linspace_1d(ny)
    return np.meshgrid(x, y, indexing="xy")
