from __future__ import annotations

import numpy as np


def always_positive(x: np.ndarray) -> bool:
    return bool(np.all(np.asarray(x, dtype=float) >= 0.0))
