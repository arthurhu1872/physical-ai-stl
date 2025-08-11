"""Minimal 1D diffusion (heat) example + simple robustness utilities.

This file is intentionally dependency-light so it can run in CI without
extra packages. The goal is to provide a tiny, correct reference for
Week 1–2 tests.
"""
from __future__ import annotations

import numpy as np


def simulate_diffusion(
    length: int = 5,
    steps: int = 20,
    dt: float = 0.1,
    alpha: float = 0.1,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    """Explicit finite-difference 1D diffusion with zero‑Neumann boundaries.

    Parameters
    ----------
    length : int
        Number of spatial points (>= 2 recommended).
    steps : int
        Number of time steps to simulate (>= 0 is allowed).
    dt : float
        Time step.
    alpha : float
        Diffusivity.
    initial : np.ndarray | None
        Optional initial state of shape (length,).

    Returns
    -------
    np.ndarray
        Array of shape (steps+1, length) containing u[0],...,u[steps].
    """
    if length <= 0:
        raise ValueError("length must be positive")
    if steps < 0:
        raise ValueError("steps must be non-negative")

    u = np.zeros((steps + 1, length), dtype=float)
    if initial is not None:
        initial = np.asarray(initial, dtype=float)
        if initial.shape != (length,):
            raise ValueError("initial must have shape (length,)")
        u[0] = initial
    else:
        # simple default: single hot spot at the left boundary
        u[0, 0] = 1.0

    if steps == 0:
        return u

    diff = alpha * dt
    for n in range(steps):
        # interior updates (explicit finite difference)
        for i in range(1, length - 1):
            u[n + 1, i] = u[n, i] + diff * (u[n, i - 1] - 2.0 * u[n, i] + u[n, i + 1])
        # zero‑Neumann boundaries via copy from the nearest interior cell
        if length > 1:
            u[n + 1, 0] = u[n + 1, 1]
            u[n + 1, -1] = u[n + 1, -2]
        else:
            u[n + 1, 0] = u[n, 0]
    return u


def simulate_diffusion_with_clipping(
    length: int = 5,
    steps: int = 20,
    dt: float = 0.1,
    alpha: float = 0.1,
    lower: float = 0.0,
    upper: float = 1.0,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    """Same as :func:`simulate_diffusion` but clip after each step to [lower, upper]."""
    u = np.zeros((steps + 1, length), dtype=float)
    # initialize u[0]
    init = None if initial is None else np.asarray(initial, dtype=float)
    u[0] = simulate_diffusion(length, 0, dt, alpha, init)[0]

    for n in range(steps):
        nxt = simulate_diffusion(length, 1, dt, alpha, u[n])[1]
        np.clip(nxt, lower, upper, out=nxt)
        u[n + 1] = nxt
    return u


def compute_robustness(signal: np.ndarray, lower: float, upper: float) -> float:
    """Return STL-style robustness for 1D signal being within [lower, upper].

    Robustness is defined as min( signal - lower, upper - signal ).
    """
    sig = np.asarray(signal, dtype=float)
    if sig.ndim != 1:
        raise ValueError("signal must be 1D")
    if sig.size == 0:
        raise ValueError("signal must not be empty")
    margins = np.minimum(sig - lower, upper - sig)
    return float(margins.min())


def compute_spatiotemporal_robustness(
    signal_matrix: np.ndarray, lower: float, upper: float
) -> float:
    """Return min robustness over a 2D (time × space) matrix for staying within bounds."""
    mat = np.asarray(signal_matrix, dtype=float)
    if mat.ndim != 2:
        raise ValueError("signal_matrix must be two-dimensional")
    if mat.size == 0:
        raise ValueError("signal_matrix must not be empty")
    margins = np.minimum(mat - lower, upper - mat)
    return float(margins.min())
