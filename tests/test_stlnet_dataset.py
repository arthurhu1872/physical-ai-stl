"""Fast, robust smoke tests for the Synthetic STLnet demo dataset.

These tests preserve the original intent (length, types, bounds, non-NaN) while
being slightly more permissive (accept any real-number scalar, excluding bool)
and adding a minimal out-of-range & zero-length check. They are intentionally
tiny and O(1) per dataset (spot-check first/last only) to stay fast.
"""

from __future__ import annotations

import math
import numbers

import pytest

from physical_ai_stl.datasets import SyntheticSTLNetDataset


@pytest.mark.parametrize("length", [1, 5])
@pytest.mark.parametrize("noise", [0.0, 0.25])
def test_synthetic_stlnet_dataset_basic(length: int, noise: float) -> None:
    ds = SyntheticSTLNetDataset(length=length, noise=noise)
    assert len(ds) == length

    # Spot-check first/last only (keeps work constant regardless of length).
    for idx in (0, length - 1):
        item = ds[idx]
        assert isinstance(item, tuple) and len(item) == 2, "Each item must be a (t, v) pair."

        t, v = item

        # Accept any real scalar (e.g., Python float or numpy floating), but not bool.
        assert isinstance(t, numbers.Real) and not isinstance(t, bool), f"time should be real, got {type(t)!r}"
        assert isinstance(v, numbers.Real) and not isinstance(v, bool), f"value should be real, got {type(v)!r}"

        # Ensure downstream code can safely consume plain floats.
        t = float(t)
        v = float(v)

        # Guarantees: t is normalized to [0, 1]; v is finite and not NaN.
        assert 0.0 <= t <= 1.0, f"time t must be in [0, 1], got {t}"
        assert v == v, "value must not be NaN"
        assert math.isfinite(v), f"value must be finite, got {v}"


def test_synthetic_stlnet_dataset_out_of_range_index() -> None:
    """Indexing exactly at len(ds) should raise IndexError (sequence semantics)."""
    ds = SyntheticSTLNetDataset(length=5)
    with pytest.raises(IndexError):
        _ = ds[len(ds)]


def test_synthetic_stlnet_dataset_zero_length() -> None:
    """Zero-length datasets report length and raise on any index access."""
    ds = SyntheticSTLNetDataset(length=0)
    assert len(ds) == 0
    with pytest.raises(IndexError):
        _ = ds[0]
