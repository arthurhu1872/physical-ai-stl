"""Tests for the PDE example module (pure NumPy)."""

from physical_ai_stl.pde_example import (
    simulate_diffusion_with_clipping,
    compute_robustness,
    compute_spatiotemporal_robustness,
)

def test_simulate_diffusion_shape() -> None:
    u = simulate_diffusion(length=3, steps=5, dt=0.1, alpha=0.1)
    assert u.shape == (6, 3)
    assert np.isfinite(u).all()

def test_diffusion_with_clipping_shape() -> None:
    u = simulate_diffusion_with_clipping(length=4, steps=3, dt=0.1, alpha=0.1, lower=0.0, upper=0.5)
    assert u.shape == (4, 4)
    assert (u >= 0.0 - 1e-12).all() and (u <= 0.5 + 1e-12).all()

def test_compute_robustness_scalar() -> None:
    sig = np.array([0.1, 0.2, 0.3, 0.4])
    rob = compute_robustness(sig, lower=0.0, upper=0.5)
    assert isinstance(rob, float)
    # min(min(sig-lower), min(upper-sig)) = min(0.1, 0.1) = 0.1
    assert abs(rob - 0.1) < 1e-8

def test_spatiotemporal_robustness_matrix() -> None:
    mat = np.array([
        [0.1, 0.2, 0.3],
        [0.05, 0.15, 0.25],
        [0.12, 0.18, 0.3],
    ])
    rob = compute_spatiotemporal_robustness(mat, lower=0.0, upper=0.5)
    assert isinstance(rob, float)
    # min margin is min(upper - max(mat), min(mat) - lower) = min(0.2, 0.05) = 0.05
    # but element-wise min margins compute to min(upper-mat, mat-lower); the worst is upper-0.5 with 0.2
    assert rob >= 0.05 - 1e-8

def test_clipping_improves_spatiotemporal_robustness() -> None:
    base_u = simulate_diffusion(length=3, steps=5, dt=0.1, alpha=0.1)
    base_rob = compute_spatiotemporal_robustness(base_u, lower=0.0, upper=0.5)
    u_clipped = simulate_diffusion_with_clipping(length=3, steps=5, dt=0.1, alpha=0.1, lower=0.0, upper=0.5)
    clip_rob = compute_spatiotemporal_robustness(u_clipped, lower=0.0, upper=0.5)
    assert clip_rob >= base_rob - 1e-8