from __future__ import annotations


def seed_everything(seed: int = 0) -> None:
    """Seed numpy, random, and torch (if available)."""
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


__all__ = ["seed_everything"]
