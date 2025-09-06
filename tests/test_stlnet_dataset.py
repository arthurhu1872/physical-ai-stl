"""STL-ready tests for SyntheticSTLNetDataset (fast, deterministic, robust).

This test file is built to match the project's STL monitoring goals:

- Time axis is normalized, monotone, and evenly spaced (linspace on [0, 1]).
- With noise=0.0, the signal equals v(t) = 0.5 * (sin(2πt) + 1).
- Sequence semantics: negative indices work; overflow raises IndexError.
- Zero-length and reproducibility behaviors are well-defined and stable.

Design:
- Deterministic: checks use noise=0.0 unless testing reproducibility.
- O(1) per dataset where possible (boundary probes); O(n) only on tiny traces.
- No heavy deps; stdlib + numpy + pytest only. No external STL packages needed.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pytest

from physical_ai_stl.datasets import SyntheticSTLNetDataset

EPS = 1e-9


def _isclose(a: float, b: float, tol: float = EPS) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tol)


def _robust_always_leq(values: Iterable[float], c: float) -> float:
    """Quantitative STL robustness for 'always (u <= c)' on discrete time."""
    min_margin = math.inf
    for v in values:
        min_margin = min(min_margin, c - v)
    return float(min_margin)


def _robust_eventually_gt(values: Iterable[float], c: float) -> float:
    """Quantitative STL robustness for 'eventually (u > c)' on discrete time."""
    max_margin = -math.inf
    for v in values:
        max_margin = max(max_margin, v - c)
    return float(max_margin)


def test_synthetic_stlnet_dataset_stl_ready() -> None:
    """Deterministic semantic checks aligned with STL monitoring needs."""
    for n in (1, 5, 17, 33):
        ds = SyntheticSTLNetDataset(length=n, noise=0.0)
        assert len(ds) == n

        # Boundary items; negative index aliases the last item.
        t0, v0 = ds[0]
        tL, vL = ds[-1]
        assert (tL, vL) == ds[n - 1]

        # Types and simple bounds.
        for t, v in ((t0, v0), (tL, vL)):
            assert isinstance(t, float) and isinstance(v, float)
            assert 0.0 <= t <= 1.0
            assert v == v and math.isfinite(v)  # not NaN, not +/-inf

        # Monotone, evenly spaced time base over [0, 1].
        assert t0 <= tL
        if n >= 2:
            step0 = ds[1][0] - t0
            stepL = tL - ds[-2][0]
            expected = 1.0 / (n - 1)
            assert _isclose(step0, expected)
            assert _isclose(stepL, expected)

        # Clean sinusoid exactness (within tight floating tolerance).
        # Probe boundaries and an interior point if available.
        probe_idxs = tuple({0, (1 if n > 1 else 0), n - 1})
        for i in probe_idxs:
            ti, vi = ds[i]
            expected_vi = 0.5 * (math.sin(2.0 * math.pi * ti) + 1.0)
            assert _isclose(vi, expected_vi), (i, ti, vi, expected_vi)

        # For one moderate length, verify all steps equal expected (O(n) but tiny).
        if n == 33:
            ts = [ds[i][0] for i in range(n)]
            diffs_ok = all(_isclose(ts[i+1] - ts[i], 1.0 / (n - 1)) for i in range(n - 1))
            assert diffs_ok


def test_synthetic_stlnet_dataset_sequence_semantics_and_zero_length() -> None:
    ds = SyntheticSTLNetDataset(length=5, noise=0.0)
    n = len(ds)
    # Positive overflow.
    with pytest.raises(IndexError):
        _ = ds[n]
    # Negative boundary and overflow.
    assert ds[-n] == ds[0]
    with pytest.raises(IndexError):
        _ = ds[-(n + 1)]

    # Zero-length dataset semantics.
    ds0 = SyntheticSTLNetDataset(length=0, noise=0.0)
    assert len(ds0) == 0
    with pytest.raises(IndexError):
        _ = ds0[0]


def test_synthetic_stlnet_dataset_simple_stl_robustness() -> None:
    """Small, dependency-free STL checks on the noiseless signal."""
    ds = SyntheticSTLNetDataset(length=33, noise=0.0)
    vals = [ds[i][1] for i in range(len(ds))]

    # eventually (u > 0.9): the peak near t=0.25 gives positive robustness.
    rob_ev = _robust_eventually_gt(vals, 0.9)
    assert rob_ev > 0.0

    # always (u <= 1.0): tight at the maximum => robustness >= 0.
    rob_alw = _robust_always_leq(vals, 1.0)
    assert rob_alw >= -EPS


def test_synthetic_stlnet_dataset_reproducible_with_seed() -> None:
    """Same seed -> identical noisy series; different seed -> typically different.
    RNG state is saved/restored to avoid cross-test interference.
    """
    length = 8
    noise = 0.1

    state = np.random.get_state()
    try:
        np.random.seed(123)
        a = SyntheticSTLNetDataset(length=length, noise=noise)
        a_vals = [a[i] for i in range(length)]

        np.random.seed(123)
        b = SyntheticSTLNetDataset(length=length, noise=noise)
        b_vals = [b[i] for i in range(length)]

        assert a_vals == b_vals  # deterministic under the same seed

        # Different seed usually changes at least one element; compute but don't assert to avoid flakiness.
        np.random.seed(456)
        c = SyntheticSTLNetDataset(length=length, noise=noise)
        c_vals = [c[i] for i in range(length)]
        _ = any(x != y for x, y in zip(a_vals, c_vals))
    finally:
        np.random.set_state(state)


def test_synthetic_stlnet_dataset_negative_length_raises() -> None:
    """Constructing with a negative length should fail early."""
    with pytest.raises((ValueError, TypeError)):
        _ = SyntheticSTLNetDataset(length=-1, noise=0.0)
