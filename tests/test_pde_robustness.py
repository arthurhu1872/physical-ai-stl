# tests/test_pde_robustness.py
from __future__ import annotations

import numpy as np
import pytest

# Import the light-weight helpers under test
from physical_ai_stl import pde_example as pe


# ----------------------------- 1D robustness ---------------------------------
def test_compute_robustness_typical_case() -> None:
    sig = np.array([0.2, 0.4, 0.6])
    rob = pe.compute_robustness(sig, lower=0.0, upper=1.0)
    # Elementwise margins are min(sig-lower, upper-sig) = [0.2, 0.4, 0.4]
    assert isinstance(rob, float)
    assert np.isclose(rob, 0.2)


@pytest.mark.parametrize(
    "sig, lower, upper, expected",
    [
        ([0.0, 1.0], 0.0, 1.0, 0.0),     # exactly on the bounds -> zero robustness
        ([0.5, 0.5], 0.0, 1.0, 0.5),     # centered in interval -> margin = 0.5
        ([-0.1, 0.2], 0.0, 1.0, -0.1),   # below lower -> negative margin
        ([0.2, 1.2], 0.0, 1.0, -0.2),    # above upper -> negative margin
    ],
)
def test_compute_robustness_boundaries_and_out_of_range(
    sig, lower, upper, expected
) -> None:
    rob = pe.compute_robustness(np.array(sig, dtype=float), lower, upper)
    assert np.isclose(rob, expected)


def test_compute_robustness_order_invariant() -> None:
    a = np.array([0.25, 0.75, 0.4, 0.6])
    b = a[::-1].copy()
    la, ua = 0.0, 1.0
    assert np.isclose(pe.compute_robustness(a, la, ua), pe.compute_robustness(b, la, ua))


@pytest.mark.parametrize("shift", [-1.3, -0.5, 0.0, 0.42, 2.0])
def test_compute_robustness_translation_invariance(shift: float) -> None:
    sig = np.array([-0.2, 0.1, 0.9, 1.3])
    l, u = 0.0, 1.0
    base = pe.compute_robustness(sig, l, u)
    shifted = pe.compute_robustness(sig + shift, l + shift, u + shift)
    assert np.isclose(base, shifted)


@pytest.mark.parametrize("scale", [0.2, 0.5, 1.0, 2.0, 10.0])
def test_compute_robustness_positive_homogeneity(scale: float) -> None:
    sig = np.array([0.2, 0.4, 0.6])
    l, u = 0.0, 1.0
    base = pe.compute_robustness(sig, l, u)
    scaled = pe.compute_robustness(scale * sig, scale * l, scale * u)
    assert np.isclose(scaled, scale * base)


def test_compute_robustness_monotonic_in_bounds() -> None:
    sig = np.array([0.2, 0.4, 0.6])
    l, u = 0.0, 1.0
    base = pe.compute_robustness(sig, l, u)

    # Tighten: raise lower and lower upper (but keep signal within the new interval)
    tighter = pe.compute_robustness(sig, l + 0.1, u - 0.3)  # new [0.1, 0.7]
    assert tighter <= base + 1e-12

    # Widen: extend both bounds
    wider = pe.compute_robustness(sig, l - 1.0, u + 1.0)
    assert wider >= base - 1e-12


@pytest.mark.parametrize(
    "bad",
    [
        np.array([], dtype=float),          # empty
        np.zeros((0,), dtype=float),        # empty 1d
        np.zeros((1, 1), dtype=float),      # not 1d
    ],
)
def test_compute_robustness_invalid_inputs(bad) -> None:
    with pytest.raises(ValueError):
        pe.compute_robustness(bad, lower=0.0, upper=1.0)


def test_compute_robustness_degenerate_interval() -> None:
    sig = np.array([0.2, 0.3, 0.6])
    r = pe.compute_robustness(sig, lower=0.3, upper=0.3)
    # elementwise min margins: [-0.1, 0.0, -0.3] -> global min -0.3
    assert np.isclose(r, -0.3)


# --------------------------- 2D (spatiotemporal) ------------------------------
def test_compute_spatiotemporal_agrees_with_flatten() -> None:
    mat = np.array([[0.5, 0.6, 0.7], [0.2, 0.4, 0.9]])
    l, u = 0.0, 1.0
    r2d = pe.compute_spatiotemporal_robustness(mat, l, u)
    r1d = pe.compute_robustness(mat.ravel(), l, u)
    assert np.isclose(r2d, r1d)


def test_compute_spatiotemporal_typical_and_constant_cases() -> None:
    mat = np.array([[0.5, 0.6], [0.7, 0.8]])
    rob = pe.compute_spatiotemporal_robustness(mat, 0.0, 1.0)
    assert np.isclose(rob, 0.2)

    const = np.full((3, 4), 0.5, dtype=float)
    assert np.isclose(
        pe.compute_spatiotemporal_robustness(const, 0.0, 1.0), 0.5
    )


@pytest.mark.parametrize(
    "bad",
    [
        np.array([], dtype=float),              # empty
        np.zeros((0, 3), dtype=float),         # one empty dimension
        np.zeros((3, 0), dtype=float),         # the other empty dimension
        np.zeros((2,), dtype=float),           # not 2d
        np.zeros((1, 1, 1), dtype=float),      # not 2d
    ],
)
def test_compute_spatiotemporal_invalid_inputs(bad) -> None:
    with pytest.raises(ValueError):
        pe.compute_spatiotemporal_robustness(bad, lower=0.0, upper=1.0)


def test_spatiotemporal_monotonic_in_bounds() -> None:
    mat = np.array([[0.2, 0.4, 0.6],
                    [0.1, 0.3, 0.5]], dtype=float)
    base = pe.compute_spatiotemporal_robustness(mat, 0.0, 1.0)
    tighter = pe.compute_spatiotemporal_robustness(mat, 0.1, 0.7)
    wider = pe.compute_spatiotemporal_robustness(mat, -1.0, 2.0)
    assert tighter <= base + 1e-12
    assert wider >= base - 1e-12
