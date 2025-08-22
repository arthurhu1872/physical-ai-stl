from __future__ import annotations

# Ensure the in-repo package is importable without installing the wheel.
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import math
from typing import Iterable

import numpy as np
import pytest

from physical_ai_stl.datasets import BoundedAtomicSpec, SyntheticSTLNetDataset

# Tight but robust absolute tolerance for exact landmark checks.
EPS = 1e-12


def _isclose(a: float, b: float, tol: float = EPS) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tol)


def _robust_always_leq(values: Iterable[float], c: float) -> float:
    m = math.inf
    for v in values:
        d = c - v
        if d < m:
            m = d
    return float(m)


def _robust_eventually_gt(values: Iterable[float], c: float) -> float:
    m = -math.inf
    for v in values:
        d = v - c
        if d > m:
            m = d
    return float(m)


def test_len_and_tuple_structure() -> None:
    for n in (0, 1, 2, 5):
        ds = SyntheticSTLNetDataset(length=n, noise=0.0)
        assert len(ds) == n
        if n:
            item = ds[0]
            assert isinstance(item, tuple) and len(item) == 2
            t0, v0 = item
            assert isinstance(t0, float) and isinstance(v0, float)


def test_time_grid_endpoints_monotonic_and_uniform() -> None:
    for n in (1, 2, 5, 33):
        ds = SyntheticSTLNetDataset(length=n, noise=0.0)

        if n == 1:
            t0, _ = ds[0]
            assert _isclose(t0, 0.0)
            continue

        t0, _ = ds[0]
        tL, _ = ds[-1]
        assert _isclose(t0, 0.0) and _isclose(tL, 1.0)

        step = 1.0 / (n - 1)
        assert _isclose(ds[1][0] - t0, step)
        assert _isclose(tL - ds[-2][0], step)

        ts = [ds[i][0] for i in range(n)]
        assert all(ts[i + 1] - ts[i] > 0.0 for i in range(n - 1))
        if n == 33:
            # Global uniformity (cheap O(n) check on small grid).
            assert all(_isclose(ts[i + 1] - ts[i], step) for i in range(n - 1))


def test_noiseless_values_at_landmarks_and_bounds() -> None:
    n = 33  # includes 0.25, 0.50, 0.75 exactly (i/(n-1) with n-1=32)
    ds = SyntheticSTLNetDataset(length=n, noise=0.0)

    idxs = [0, 8, 16, 24, 32]
    expected = [0.5, 1.0, 0.5, 0.0, 0.5]
    for i, e in zip(idxs, expected):
        ti, vi = ds[i]
        assert _isclose(vi, e), (i, ti, vi, e)

    vals = [ds[i][1] for i in range(n)]
    assert min(vals) >= -EPS and max(vals) <= 1.0 + EPS


def test_noiseless_full_trace_matches_formula_small() -> None:
    n = 33
    ds = SyntheticSTLNetDataset(length=n, noise=0.0)
    for i in range(n):
        t, v = ds[i]
        clean = 0.5 * (math.sin(2.0 * math.pi * t) + 1.0)
        assert _isclose(v, clean)


def test_length_one_semantics() -> None:
    ds = SyntheticSTLNetDataset(length=1, noise=0.0)
    t, v = ds[0]
    assert _isclose(t, 0.0) and _isclose(v, 0.5)


def test_sequence_indexing_semantics_and_overflow() -> None:
    n = 5
    ds = SyntheticSTLNetDataset(length=n, noise=0.0)

    with pytest.raises(IndexError):
        _ = ds[n]
    assert ds[-1] == ds[n - 1]
    assert ds[-n] == ds[0]
    with pytest.raises(IndexError):
        _ = ds[-(n + 1)]

    ds0 = SyntheticSTLNetDataset(length=0, noise=0.0)
    assert len(ds0) == 0
    with pytest.raises(IndexError):
        _ = ds0[0]


def test_no_nan_or_inf_even_with_noise() -> None:
    n = 33
    np.random.seed(7)
    ds = SyntheticSTLNetDataset(length=n, noise=0.3)
    vals = [ds[i][1] for i in range(n)]
    assert all(np.isfinite(vals))


def test_dependency_free_robustness_checks_on_clean_signal() -> None:
    ds = SyntheticSTLNetDataset(length=33, noise=0.0)
    vals = [ds[i][1] for i in range(len(ds))]

    assert _robust_eventually_gt(vals, 0.9) > 0.0
    assert _robust_always_leq(vals, 1.0) >= -EPS


def test_reproducible_and_linear_noise_scaling_with_numpy_seed() -> None:
    length = 16
    noise_a = 0.1
    noise_b = 0.2

    state = np.random.get_state()
    try:
        # Identical with same seed and same noise.
        np.random.seed(2024)
        a = SyntheticSTLNetDataset(length=length, noise=noise_a)
        a_vals = [a[i] for i in range(length)]
        np.random.seed(2024)
        b = SyntheticSTLNetDataset(length=length, noise=noise_a)
        b_vals = [b[i] for i in range(length)]
        assert a_vals == b_vals  # deterministic under same seed

        # Linear residual scaling with identical underlying draws.
        np.random.seed(2024)
        c = SyntheticSTLNetDataset(length=length, noise=noise_b)
        for i in range(length):
            t, va = a[i]
            _, vc = c[i]
            clean = 0.5 * (math.sin(2.0 * math.pi * t) + 1.0)
            ra = va - clean
            rc = vc - clean
            if abs(ra) > 1e-15 or abs(rc) > 1e-15:  # avoid 0/0 on exact zeros
                assert _isclose(rc, (noise_b / noise_a) * ra, tol=1e-12)
    finally:
        # Avoid cross-test interference with other suites.
        np.random.set_state(state)


def test_invalid_lengths_raise() -> None:
    with pytest.raises((ValueError, TypeError)):
        _ = SyntheticSTLNetDataset(length=-1, noise=0.0)
    with pytest.raises((ValueError, TypeError)):
        _ = SyntheticSTLNetDataset(length=3.7, noise=0.0)  # type: ignore[arg-type]


def test_invalid_noise_raises() -> None:
    with pytest.raises(ValueError):
        _ = SyntheticSTLNetDataset(length=8, noise=-1.0)


def test_rng_parameter_reproducibility_for_generator_and_randomstate() -> None:
    L = 10
    # Generator
    g1 = np.random.default_rng(123)
    g2 = np.random.default_rng(123)
    d1 = SyntheticSTLNetDataset(length=L, noise=0.5, rng=g1)
    d2 = SyntheticSTLNetDataset(length=L, noise=0.5, rng=g2)
    assert [d1[i] for i in range(L)] == [d2[i] for i in range(L)]

    # RandomState
    r1 = np.random.RandomState(12345)
    r2 = np.random.RandomState(12345)
    e1 = SyntheticSTLNetDataset(length=L, noise=0.5, rng=r1)
    e2 = SyntheticSTLNetDataset(length=L, noise=0.5, rng=r2)
    assert [e1[i] for i in range(L)] == [e2[i] for i in range(L)]


def test_rng_type_validation() -> None:
    class BadRNG:
        pass

    with pytest.raises(TypeError):
        _ = SyntheticSTLNetDataset(length=8, noise=0.1, rng=BadRNG())  # type: ignore[arg-type]


def test_windows_and_robustness_shape_and_values() -> None:
    # Small, hand-checkable example.
    n = 9
    win = 5
    stride = 2
    ds = SyntheticSTLNetDataset(length=n, noise=0.0)
    # Windows of length 5 with stride 2.
    t_win, v_win = ds.windows(length=win, stride=stride)
    # There are n - win + 1 windows before striding; slicing [::stride] keeps 0,2,4 -> 3 windows.
    assert t_win.shape[1] == win and v_win.shape[1] == win
    assert t_win.shape == v_win.shape == (3, win)  # 5 raw windows, keep 0,2,4 -> 3
    assert np.issubdtype(t_win.dtype, np.floating) and np.issubdtype(v_win.dtype, np.floating)

    # Simple spec: always v <= 1 over horizon H=4 samples (window length=5).
    spec = BoundedAtomicSpec(temporal="always", op="<=", threshold=1.0, horizon=4)
    # Convenience wrapper pairs windows and robustness with matching stride.
    t2, v2, rho = ds.windowed_robustness(spec, stride=stride)
    # Matching shapes and equality to the direct window computation.
    assert np.allclose(t2, t_win) and np.allclose(v2, v_win)
    # With clean sine wave in [0,1], robustness to v<=1 should be >= 0.
    assert np.all(rho >= -EPS)
