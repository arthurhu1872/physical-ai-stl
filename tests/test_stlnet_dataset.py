from __future__ import annotations

import math
import numbers

import pytest

from physical_ai_stl.datasets import SyntheticSTLNetDataset


@pytest.mark.parametrize("length", [1, 5, 17])
@pytest.mark.parametrize("noise", [0.0, 0.25])
def test_synthetic_stlnet_dataset_basic(length: int, noise: float) -> None:
    ds = SyntheticSTLNetDataset(length=length, noise=noise)
    assert len(ds) == length

    # Spot-check only first/last to keep this test O(1) regardless of length.
    for idx in (0, length - 1):
        item = ds[idx]
        assert isinstance(item, tuple) and len(item) == 2, "Each item must be a (t, v) pair."

        t, v = item
        # Accept any real scalar (e.g., Python float or numpy floating).
        assert isinstance(t, numbers.Real), f"time should be real, got {type(t)!r}"
        assert isinstance(v, numbers.Real), f"value should be real, got {type(v)!r}"

        # Ensure downstream code can safely consume plain floats.
        t = float(t)
        v = float(v)

        # Original guarantees: t is normalized, v is a finite real number.
        assert 0.0 <= t <= 1.0, f"time t must be in [0, 1], got {t}"
        assert v == v, "value must not be NaN"  # NaN check (kept from original)
        assert math.isfinite(v), f"value must be finite, got {v}"


def test_synthetic_stlnet_dataset_out_of_range_index() -> None:
    """Indexing past the end should raise IndexError (typical dataset behavior)."""
    ds = SyntheticSTLNetDataset(length=5)
    with pytest.raises(IndexError):
        _ = ds[len(ds)]
