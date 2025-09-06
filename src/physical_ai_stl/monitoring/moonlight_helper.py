from __future__ import annotations

import numpy as np


def normalize_series(x: np.ndarray) -> np.ndarray:
    """
    Normalize to [0, 1] for simple plotting / demo monitoring.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    lo = float(x.min())
    hi = float(x.max())
    if hi == lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)
