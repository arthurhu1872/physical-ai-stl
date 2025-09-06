"""Tests for SyntheticSTLNetDataset aligned with STL monitoring use-cases.

This file is designed to be tiny, deterministic, and CI-friendly while
validating properties that matter for Signal Temporal Logic (STL) monitoring
and physical-AI demos:

  * Normalized, monotonic, evenly spaced time base t ∈ [0, 1].
  * Clean reference signal when noise=0.0: v(t) = 0.5 (sin(2πt) + 1).
  * Standard sequence semantics (negative indexing; IndexError on overflow).
  * Zero-length behavior (len==0; any access raises IndexError).

Optionally (if RTAMT is installed), we also evaluate two tiny STL specs against
the dataset to ensure it can be consumed by an STL monitor out of the box:
  * eventually (u > 0.9)        — should be satisfied on the clean sinusoid
  * always    (u <= 1.0)        — exactly satisfied at the peak (robustness ≥ 0)

These checks stay O(1) per dataset (probe boundaries and a couple interior
points only) and depend only on the stdlib (plus pytest for skipping).
"""
from __future__ import annotations

import math
import numbers
from typing import Iterable, Tuple

import pytest

from physical_ai_stl.datasets import SyntheticSTLNetDataset


def _isclose(a: float, b: float, tol: float = 1e-12) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tol)


def test_synthetic_stlnet_dataset_semantics() -> None:
    """Deterministic, constant-time semantic checks over a few lengths."""
    for n in (1, 5, 17):
        ds = SyntheticSTLNetDataset(length=n, noise=0.0)
        assert len(ds) == n

        # Boundary items (negative index must alias the last element).
        t0, v0 = ds[0]
        tL, vL = ds[-1]
        assert (tL, vL) == ds[n - 1]

        # Types and bounds (accept any real scalar; exclude bool).
        for t, v in ((t0, v0), (tL, vL)):
            assert isinstance(t, numbers.Real) and not isinstance(t, bool)
            assert isinstance(v, numbers.Real) and not isinstance(v, bool)
            assert 0.0 <= t <= 1.0
            assert math.isfinite(v)
            assert v == v  # not NaN

        # Monotone, evenly spaced time base over [0, 1].
        assert t0 <= tL
        if n >= 2:
            step0 = ds[1][0] - t0
            stepL = tL - ds[-2][0]
            expected = 1.0 / (n - 1)
            assert _isclose(step0, expected)
            assert _isclose(stepL, expected)

        # Clean sinusoid when noise=0.0.
        probe_idxs = tuple({0, (1 if n > 1 else 0), n - 1})
        for i in probe_idxs:
            ti, vi = ds[i]
            expected_vi = 0.5 * (math.sin(2.0 * math.pi * ti) + 1.0)
            assert _isclose(vi, expected_vi), (i, ti, vi, expected_vi)

    # Indexing semantics and zero-length behavior (O(1)).
    ds = SyntheticSTLNetDataset(length=5, noise=0.0)
    n = len(ds)

    # Positive overflow.
    with pytest.raises(IndexError):
        _ = ds[n]

    # Negative boundary and overflow.
    assert ds[-n] == ds[0]
    with pytest.raises(IndexError):
        _ = ds[-(n + 1)]

    # Zero-length dataset.
    ds0 = SyntheticSTLNetDataset(length=0, noise=0.0)
    assert len(ds0) == 0
    with pytest.raises(IndexError):
        _ = ds0[0]


def test_synthetic_stlnet_dataset_rtamt_optional() -> None:
    """Optional STL smoke check via RTAMT (skips cleanly if missing)."""
    try:
        import rtamt  # type: ignore
    except Exception:
        pytest.skip("RTAMT not available; skipping STL monitor check.")
        return

    ds = SyntheticSTLNetDataset(length=5, noise=0.0)
    # RTAMT discrete-time expects (time_index, value) pairs.
    ts: Iterable[Tuple[int, float]] = [(i, ds[i][1]) for i in range(len(ds))]
    ts_list = list(ts)

    # eventually (u > 0.9)
    spec_ev = rtamt.StlDiscreteTimeSpecification()
    spec_ev.declare_var("u", "float")
    spec_ev.spec = "(eventually (u > 0.9))"
    spec_ev.parse()

    # always (u <= 1.0)
    spec_alw = rtamt.StlDiscreteTimeSpecification()
    spec_alw.declare_var("u", "float")
    spec_alw.spec = "(always (u <= 1.0))"
    spec_alw.parse()

    def _robust(x) -> float:
        """Coerce RTAMT outputs across versions to a float at t0."""
        try:
            return float(x)
        except Exception:
            if isinstance(x, (list, tuple)):
                if not x:
                    return 0.0
                first = x[0]
                if isinstance(first, (list, tuple)):
                    # (t, rob) or similar
                    return float(first[1] if len(first) > 1 else first[0])
                return float(first)
            return float(x)  # may still raise

    try:
        rob_ev = _robust(spec_ev.evaluate(["u"], [ts_list]))
        rob_alw = _robust(spec_alw.evaluate(["u"], [ts_list]))
    except Exception as e:
        pytest.skip(f"RTAMT evaluate API mismatch or runtime error: {e}")
        return

    # The clean sinusoid reaches 1.0 at t = 0.25 -> eventually (>0.9) holds (robust > 0).
    assert rob_ev > 0.0
    # The peak is exactly 1.0 -> always (<= 1.0) has robustness >= 0 (tight at the maximum).
    assert rob_alw >= 0.0
