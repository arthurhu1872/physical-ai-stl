from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """
    Seed common RNGs to make experiments reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
