from __future__ import annotations

import numpy as np


def soft_min(x: np.ndarray) -> float:
    """Soft minimum used in some STL relaxations."""
    x = np.asarray(x, dtype=float)
    return float(np.min(x, initial=np.inf))
