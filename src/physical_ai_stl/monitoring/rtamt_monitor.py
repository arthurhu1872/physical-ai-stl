"""Helpers for building and evaluating STL specs with RTAMT (offline)."""

from __future__ import annotations

from typing import Iterable, List, Tuple


def _import_rtamt():
    try:
        import rtamt  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "rtamt is not installed. Run `pip install rtamt` to use this module."
        ) from e
    return rtamt


def stl_always_upper_bound(var: str = "u", u_max: float = 1.0):
    """Return a parsed RTAMT discrete-time spec: G (var <= u_max)."""
    rtamt = _import_rtamt()
    spec = rtamt.StlDiscreteTimeSpecification()
    spec.declare_var(var, "float")
    spec.spec = f"always ({var} <= {float(u_max)})"
    spec.parse()
    return spec


def evaluate_series(spec, var: str, series: Iterable[float], dt: float = 1.0) -> float:
    """Evaluate robustness for `var` time series against `spec` offline."""
    rtamt = _import_rtamt()
    ts = [(i * dt, float(v)) for i, v in enumerate(series)]
    # RTAMT wants parallel arrays of names and series
    rob = spec.evaluate([var], [ts])[0][1]
    return float(rob)
