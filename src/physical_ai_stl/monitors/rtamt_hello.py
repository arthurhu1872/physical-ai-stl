"""Tiny RTAMT 'hello world' used by tests as a smoke-check."""
from __future__ import annotations

def stl_hello_offline() -> float:
    # Import inside the function to allow test to skip cleanly when missing.
    try:
        import rtamt  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("rtamt not installed") from e

    spec = rtamt.StlDiscreteTimeSpecification()
    spec.declare_var("u", "float")
    spec.spec = "always (u <= 1.0)"
    spec.parse()
    # simple, safe-to-evaluate time-series
    ts = [(0, 0.2), (1, 0.4), (2, 1.1)]
    # robustness at t0
    rob = spec.evaluate(["u"], [ts])[0][1]
    return float(rob)

__all__ = ["stl_hello_offline"]
