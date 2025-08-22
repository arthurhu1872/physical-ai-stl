# src/physical_ai_stl/datasets/stlnet_synthetic.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np


def _sliding_window(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError(f"window must be positive; got {window}")
    n = int(np.asarray(x).shape[0])
    if window > n:
        # Empty view (no windows fit); keep a consistent 2‑D shape.
        return np.empty((0, window), dtype=x.dtype)

    try:
        return np.lib.stride_tricks.sliding_window_view(x, window)  # type: ignore[attr-defined]
    except Exception:
        # Fallback: stack whole windows; fast enough for our tiny default sizes.
        starts = np.arange(0, n - window + 1, dtype=int)
        idx = starts[:, None] + np.arange(window, dtype=int)[None, :]
        return x[idx]


@dataclass(frozen=True)
class BoundedAtomicSpec:

    temporal: str  # "always" | "eventually"
    op: str        # "<=" | ">="
    threshold: float
    horizon: int = 0

    def __post_init__(self) -> None:
        temporal = self.temporal.lower()
        if temporal not in {"always", "eventually"}:
            raise ValueError(f"temporal must be 'always' or 'eventually'; got {self.temporal!r}")
        op = self.op
        if op not in {"<=", ">="}:
            raise ValueError(f"op must be '<=' or '>='; got {self.op!r}")
        H = int(self.horizon)
        if H < 0:
            raise ValueError(f"horizon must be non‑negative; got {self.horizon!r}")

    # Public API ---------------------------------------------------------------

    def robustness(self, v: np.ndarray, stride: int = 1) -> np.ndarray:
        v = np.asarray(v, dtype=float).reshape(-1)
        H = int(self.horizon)
        if v.ndim != 1:
            raise ValueError("v must be 1‑D (scalar signal).")
        if stride <= 0:
            raise ValueError("stride must be a positive integer.")
        window = H + 1
        Wins = _sliding_window(v, window)
        if Wins.size == 0:
            return np.empty((0,), dtype=float)
        Wins = Wins[::stride, :]
        # Atomic robustness
        if self.op == "<=":
            r = self.threshold - Wins
        else:  # self.op == ">="
            r = Wins - self.threshold
        # Temporal reduction
        if self.temporal == "always":
            rho = np.min(r, axis=1)
        else:  # "eventually"
            rho = np.max(r, axis=1)
        return rho

    def satisfied(self, v: np.ndarray, stride: int = 1) -> np.ndarray:
        return self.robustness(v, stride=stride) > 0.0


class SyntheticSTLNetDataset:

    __slots__ = ("_data",)

    def __init__(self, length: int = 100, noise: float = 0.05, rng: Optional[object] = None) -> None:
        if not isinstance(length, (int, np.integer)):
            raise TypeError(f"length must be an integer; got {type(length).__name__}")
        if length < 0:
            raise ValueError(f"length must be non‑negative; got {length}")
        if noise < 0:
            raise ValueError(f"noise must be non‑negative; got {noise}")
        n = int(length)
        # Time axis (linspace) – exact 0 and 1 endpoints when n>0.
        t = np.linspace(0.0, 1.0, num=n, dtype=float) if n > 0 else np.empty((0,), dtype=float)

        # Clean, exactly bounded sinusoid on [0, 1].
        clean = 0.5 * (np.sin(2.0 * np.pi * t) + 1.0)

        if n == 0:
            v = clean  # empty
        elif noise == 0.0:
            v = clean
        else:
            # Draw noise from the requested RNG without disturbing global state unless asked.
            if rng is None:
                eps = np.random.randn(n)  # respects external np.random.seed
            else:
                # Support both Generator.standard_normal and RandomState.randn
                if hasattr(rng, "standard_normal"):
                    eps = rng.standard_normal(n)  # type: ignore[attr-defined]
                elif hasattr(rng, "randn"):
                    eps = rng.randn(n)  # type: ignore[attr-defined]
                else:
                    raise TypeError("rng must be a NumPy Generator or RandomState‑like object.")
            v = clean + float(noise) * eps

        self._data = np.stack((t, v), axis=1) if n > 0 else np.empty((0, 2), dtype=float)

    # Sequence protocol --------------------------------------------------------

    def __len__(self) -> int:  # pragma: no cover - trivial
        return int(self._data.shape[0])

    def __getitem__(self, idx: int) -> Tuple[float, float]:
        t, v = self._data[idx]  # NumPy handles negative/overflow checks.
        return float(t), float(v)

    # Convenience accessors ----------------------------------------------------

    @property
    def array(self) -> np.ndarray:
        return self._data

    @property
    def t(self) -> np.ndarray:
        return self._data[:, 0]

    @property
    def v(self) -> np.ndarray:
        return self._data[:, 1]

    # STL helpers --------------------------------------------------------------

    def windows(self, length: int, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        if length <= 0:
            raise ValueError("window length must be positive.")
        if stride <= 0:
            raise ValueError("stride must be a positive integer.")
        t_win = _sliding_window(self.t, int(length))
        v_win = _sliding_window(self.v, int(length))
        return t_win[::stride, :], v_win[::stride, :]

    def windowed_robustness(
        self,
        spec: BoundedAtomicSpec,
        stride: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        H = int(spec.horizon)
        t_win, v_win = self.windows(H + 1, stride=stride)
        rho = spec.robustness(self.v, stride=stride)
        return t_win, v_win, rho


__all__ = ["SyntheticSTLNetDataset", "BoundedAtomicSpec"]
