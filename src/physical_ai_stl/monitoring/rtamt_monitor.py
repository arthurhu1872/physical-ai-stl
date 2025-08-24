"""Helpers for building and evaluating STL specs with RTAMT (offline)."""
from __future__ import annotations
from typing import Iterable

def _import_rtamt():
    try:
        import rtamt  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "rtamt is not installed. Run 'pip install rtamt' to use this module."
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

def stl_response_within(var: str, boundary: str, theta: float, tau: int):
    """Return spec: always( boundary >= theta -> eventually[0:tau] (var >= theta) )."""
    rtamt = _import_rtamt()
    spec = rtamt.StlDiscreteTimeSpecification()
    spec.declare_var(var, "float")
    spec.declare_var(boundary, "float")
    spec.spec = (
        f"always ( ({boundary} >= {float(theta)}) -> eventually[0:{int(tau)}] ({var} >= {float(theta)}) )"
    )
    spec.parse()
    return spec

def evaluate_series(spec, var: str, series: Iterable[float], dt: float = 1.0) -> float:
    """Evaluate robustness for a single variable time series against spec."""
    # Build (time, value) pairs
    ts = [(i * dt, float(v)) for i, v in enumerate(series)]
    # RTAMT Python API accepts a dict of var->series
    rob = spec.evaluate({var: ts})
    try:
        return float(rob)
    except Exception:
        if isinstance(rob, (list, tuple)):
            if not rob:
                return 0.0
            first = rob[0]
            if isinstance(first, (list, tuple)):
                return float(first[1] if len(first) > 1 else first[0])
            else:
                return float(first)
        else:
            return float(rob)

__all__ = [
    "stl_always_upper_bound",
    "stl_response_within",
    "evaluate_series",
]
