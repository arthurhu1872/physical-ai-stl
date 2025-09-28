from __future__ import annotations

import numpy as np


def _validate_length_steps(length: int, steps: int) -> None:
    if length <= 0:
        raise ValueError("length must be positive")
    if steps < 0:
        raise ValueError("steps must be non‑negative")


def simulate_diffusion(
    length: int = 5,
    steps: int = 20,
    dt: float = 0.1,
    alpha: float = 0.1,
    initial: np.ndarray | None = None,
    *,
    dx: float = 1.0,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    _validate_length_steps(length, steps)

    r = float(alpha) * float(dt) / float(dx) ** 2
    u = np.zeros((steps + 1, length), dtype=dtype)

    if initial is not None:
        init = np.asarray(initial, dtype=dtype)
        if init.shape != (length,):
            raise ValueError("initial must have shape (length,)")
        u[0] = init
    else:
        # simple default: single hot spot at the left boundary
        u[0, 0] = dtype(1.0)

    if steps == 0:
        return u

    if length == 1:
        # Degenerate domain: nothing to diffuse; copy the single value.
        for n in range(steps):
            u[n + 1, 0] = u[n, 0]
        return u

    # Vectorized interior update
    for n in range(steps):
        cur = u[n]
        nxt = u[n + 1]

        # u[n+1, 1:-1] = u[n, 1:-1] + r * (u[n, 0:-2] - 2*u[n, 1:-1] + u[n, 2:])
        nxt[1:-1] = cur[1:-1] + r * (cur[:-2] - 2.0 * cur[1:-1] + cur[2:])

        # copy‑Neumann boundaries (enforce ∂u/∂x = 0 at n+1 by reflection)
        nxt[0] = nxt[1]
        nxt[-1] = nxt[-2]

    return u


def simulate_diffusion_with_clipping(
    length: int = 5,
    steps: int = 20,
    dt: float = 0.1,
    alpha: float = 0.1,
    lower: float = 0.0,
    upper: float = 1.0,
    initial: np.ndarray | None = None,
    *,
    dx: float = 1.0,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    if lower > upper:
        raise ValueError("lower must be <= upper")

    u0 = simulate_diffusion(length, 0, dt, alpha, initial, dx=dx, dtype=dtype)
    # Clip the very first frame
    np.clip(u0[0], lower, upper, out=u0[0])

    if steps == 0:
        return u0

    # Efficient step‑by‑step update with clipping (no extra simulator calls)
    r = float(alpha) * float(dt) / float(dx) ** 2
    out = np.zeros((steps + 1, length), dtype=dtype)
    out[0] = u0[0]

    if length == 1:
        for n in range(steps):
            out[n + 1, 0] = out[n, 0]
        return out

    for n in range(steps):
        cur = out[n]
        nxt = out[n + 1]
        nxt[1:-1] = cur[1:-1] + r * (cur[:-2] - 2.0 * cur[1:-1] + cur[2:])
        # copy‑Neumann
        nxt[0] = nxt[1]
        nxt[-1] = nxt[-2]
        # clip in‑place
        np.clip(nxt, lower, upper, out=nxt)

    return out


def compute_robustness(signal: np.ndarray, lower: float, upper: float) -> float:
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
    mat = np.asarray(signal_matrix, dtype=float)
    if mat.ndim != 2:
        raise ValueError("signal_matrix must be two‑dimensional")
    if mat.size == 0:
        raise ValueError("signal_matrix must not be empty")
    margins = np.minimum(mat - lower, upper - mat)
    return float(margins.min())


# --- Optional helpers (kept tiny; can be handy in notebooks/tests) -----------------


def pointwise_bounds_margin(x: np.ndarray, lower: float, upper: float) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return np.minimum(arr - lower, upper - arr)


def _sliding_extreme(x: np.ndarray, window: int, extreme: str) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be positive")
    x = np.asarray(x, dtype=float)
    from collections import deque

    cmp = (lambda a, b: a <= b) if extreme == "min" else (lambda a, b: a >= b)
    dq: deque[tuple[int, float]] = deque()
    out = np.empty_like(x, dtype=float)
    for i, val in enumerate(x):
        # pop dominated values from the right
        while dq and cmp(val, dq[-1][1]):
            dq.pop()
        dq.append((i, val))
        # drop from the left if window exceeded
        left = i - window + 1
        if dq[0][0] < left:
            dq.popleft()
        out[i] = dq[0][1]
    return out


def stl_globally_robustness(rho_phi: np.ndarray, window: int) -> np.ndarray:
    return _sliding_extreme(rho_phi, window, "min")


def stl_eventually_robustness(rho_phi: np.ndarray, window: int) -> np.ndarray:
    return _sliding_extreme(rho_phi, window, "max")


__all__ = [
    "simulate_diffusion",
    "simulate_diffusion_with_clipping",
    "compute_robustness",
    "compute_spatiotemporal_robustness",
    # optional helpers
    "pointwise_bounds_margin",
    "stl_globally_robustness",
    "stl_eventually_robustness",
]
