from __future__ import annotations

import numpy as np


def demo_signal(n: int = 64) -> np.ndarray:
    t = np.linspace(0.0, 1.0, num=n, dtype=float)
    return np.sin(2.0 * np.pi * t)
