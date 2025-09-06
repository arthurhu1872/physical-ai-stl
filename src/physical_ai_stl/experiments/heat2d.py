from __future__ import annotations

from collections.abc import Iterable

import numpy as np

# Placeholder helpers used by examples; kept minimal for linting.
def as_iterable(x: float | Iterable[float]) -> list[float]:
    return list(x) if isinstance(x, list | tuple) else [float(x)]  # UP038 compliant


def demo_field(n: int = 16) -> np.ndarray:
    t = np.linspace(0.0, 1.0, num=n, dtype=float)
    x, y = np.meshgrid(t, t, indexing="xy")
    return np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
