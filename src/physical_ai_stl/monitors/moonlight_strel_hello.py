from __future__ import annotations

import numpy as np


def demo_surface(n: int = 32) -> np.ndarray:
    t = np.linspace(0.0, 1.0, num=n, dtype=float)
    x, y = np.meshgrid(t, t, indexing="xy")
    return np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)
