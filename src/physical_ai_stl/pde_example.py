"""Minimal PDE and robustness example for sprint 1.

This module implements a 1D diffusion (heat) equation solver and simple
robustness computations without external STL libraries.
"""

from __future__ import annotations
import numpy as np


def simulate_diffusion(length: int = 5,
                       steps: int = 20,
                       dt: float = 0.1,
                       alpha: float = 0.1,
                       initial: np.ndarray | None = None) -> np.ndarray:
    if length <= 0 or steps <= 0:
        raise ValueError("length and steps must be positive integers")
    u = np.zeros((steps + 1, length), dtype=float)
    if initial is not None:
        if len(initial) != length:
            raise ValueError("initial state must have length equal to 'length'")
        u[0] = np.asarray(initial, dtype=float)
    else:
        u[0, 0] = 1.0
    diff_factor = alpha * dt
    for n in range(steps):
        for i in range(1, length - 1):
            u[n + 1, i] = u[n, i] + diff_factor * (u[n, i - 1] - 2.0 * u[n, i] + u[n, i + 1])
        u[n + 1, 0] = u[n + 1, 1]
        u[n + 1, -1] = u[n + 1, -2]
    return u


def simulate_diffusion_with_clipping(length: int = 5,
                                     steps: int = 20,
                                     dt: float = 0.1,
                                     alpha: float = 0.1,
                                     lower: float = 0.0,
                                     upper: float = 1.0,
                                     initial: np.ndarray | None = None) -> np.ndarray:
    u = simulate_diffusion(length=length, steps=steps, dt=dt, alpha=alpha, initial=initial)
      # Clip the entire state to the specified bounds.
    np.clip(u, lower, upper, out=u)
    return u


def compute_robustness(signal: np.ndarray, lower: float, upper: float) -> float:
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError("signal must be a one-dimensional array")
    if signal.size == 0:
        raise ValueError("signal must not be empty")
    margins = np.minimum(signal - lower, upper - signal)
    return float(margins.min())


def compute_spatiotemporal_robustness(signal_matrix: np.ndarray, lower: float, upper: float) -> float:
    signal_matrix = np.asarray(signal_matrix, dtype=float)
    if signal_matrix.ndim != 2:
        raise ValueError("signal_matrix must be two-dimensional")
    if signal_matrix.size == 0:
        raise ValueError("signal_matrix must not be empty")
       margins = upper - signal_matrix
    return float(margins.min())
