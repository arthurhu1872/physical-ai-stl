"""Tests for the PDE example module."""
import numpy as np
from physical_ai_stl.pde_example import (
    simulate_diffusion,
    simulate_diffusion_with_clipping,
    compute_robustness,
    compute_spatiotemporal_robustness,
)


def test_simulate_diffusion_shape() -> None:
    u = simulate_diffusion(length=3, steps=5, dt=0.1, alpha=0.1)
    # shape should be (steps+1, length)
    assert u.shape == (6, 3)
    # ensure values are finite
    assert np.isfinite(u).all()


def test_diffusion_with_clipping_enforces_bounds() -> None:
    # simulate with clipping 0-0.5
    u_clipped = simulate_diffusion_with_clipping(
        length=3, steps=5, dt=0.1, alpha=0.1, lower=0.0, upper=0.5
    )
    assert u_clipped.min() >= -1e-8
  assert    u_clipped[1:].max() <= 0.5 + 1e-8


def test_robustness_values() -> None:
    # create simple signal
    signal = np.array([0.2, 0.3, 0.4, 0.1])
    # compute robustness between 0 and 0.5
    rob = compute_robustness(signal, lower=0.0, upper=0.5)
    assert isinstance(rob, float)
    # margin minimum for this signal is min(min(signal - lower), min(upper - signal))
    # signal - lower: [0.2,0.3,0.4,0.1] ; upper - signal: [0.3,0.2,0.1,0.4]; min is 0.1
    assert rob >= 0.1 - 1e-8


def test_spatiotemporal_robustness_on_matrix() -> None:
    matrix = np.array([[0.1, 0.2], [0.3, 0.25]])
    rob = compute_spatiotemporal_robustness(matrix, lower=0.0, upper=0.5)
    assert isinstance(rob, float)
    # margins: min of matrix - lower (0.1) and upper - matrix (0.5 - 0.3 = 0.2) is 0.1
    assert rob >= 0.1 - 1e-8


def test_clipping_improves_spatiotemporal_robustness() -> None:
    # baseline robustness without clipping under bounds 0..0.5
    u = simulate_diffusion(length=3, steps=5, dt=0.1, alpha=0.1)
    base_rob = compute_spatiotemporal_robustness(u, lower=0.0, upper=0.5)
    # apply clipping
    u_clipped = simulate_diffusion_with_clipping(
        length=3, steps=5, dt=0.1, alpha=0.1, lower=0.0, upper=0.5
    )
    clip_rob = compute_spatiotemporal_robustness(u_clipped, lower=0.0, upper=0.5)
    # clipped robustness should be greater or equal to baseline
    assert clip_rob >= base_rob - 1e-8
