from __future__ import annotations

from typing import Final

import numpy as np

__all__ = [
    "simulate_diffusion",
    "simulate_diffusion_with_clipping",
    "compute_robustness_scalar",
    "compute_spatiotemporal_robustness",
]


def _validate_length_steps(length: int, steps: int) -> None:
    if length < 0 or steps < 0:
        raise ValueError("length and steps must be non-negative")


def simulate_diffusion(
    *,
    length: int = 5,
    steps: int = 20,
    dt: float = 0.1,
    alpha: float = 0.1,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    """
    Simple 1‑D explicit diffusion with zero Dirichlet boundaries.

    Returns an array with shape ``(steps+1, length)``. Row 0 is the initial state.
    """
    _validate_length_steps(length, steps)

    u = np.zeros((steps + 1, length), dtype=float)

    if length == 0:
        return u

    if initial is None:
        # A small centered bump for determinism.
        mid = length // 2
        u[0, mid] = 1.0
    else:
        initial = np.asarray(initial, dtype=float)
        if initial.shape != (length,):
            raise ValueError("initial must have shape (length,)")
        u[0] = initial

    if length < 3:
        # Not enough interior points to update.
        for k in range(steps):
            u[k + 1] = u[k]
        return u

    # Stable explicit scheme: u_{t+1} = u_t + λ * Δ u_t
    # with Δ the discrete Laplacian. Use a conservative λ for stability.
    lam: Final[float] = min(alpha * dt * (length - 1) ** 2, 0.49)

    for k in range(steps):
        uk = u[k]
        nxt = uk.copy()
        nxt[1:-1] = uk[1:-1] + lam * (uk[:-2] - 2.0 * uk[1:-1] + uk[2:])
        nxt[0] = 0.0
        nxt[-1] = 0.0
        u[k + 1] = nxt

    return u


def simulate_diffusion_with_clipping(
    *,
    length: int = 5,
    steps: int = 20,
    dt: float = 0.1,
    alpha: float = 0.1,
    lower: float = 0.0,
    upper: float = 1.0,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    """
    Same as :func:`simulate_diffusion`, but clip *every* frame to ``[lower, upper]``.
    Row 0 (the initial state) is also clipped so tests that examine the whole
    tensor—including the first row—are deterministic.
    """
    u = simulate_diffusion(
        length=length, steps=steps, dt=dt, alpha=alpha, initial=initial
    )
    return np.clip(u, lower, upper, out=u)


def compute_robustness_scalar(signal: np.ndarray) -> float:
    """A very small 'robustness' helper: the worst value in a 1‑D trace."""
    sig = np.asarray(signal, dtype=float)
    if sig.ndim != 1:
        raise ValueError("signal must be 1-D")
    return float(sig.min(initial=np.inf))


def compute_spatiotemporal_robustness(matrix: np.ndarray) -> float:
    """
    Spatio‑temporal robustness proxy: the worst value in a 2‑D matrix
    (min over space and time).
    """
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2:
        raise ValueError("matrix must be 2-D")
    return float(mat.min(initial=np.inf))
