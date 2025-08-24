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
    # robustness at t0 (handle output format differences)
    rob_val = spec.evaluate(["u"], [ts])
    try:
        rob_val = float(rob_val)
    except Exception:
        if isinstance(rob_val, (list, tuple)):
            if not rob_val:
                rob_val = 0.0
            else:
                first = rob_val[0]
                if isinstance(first, (list, tuple)):
                    rob_val = first[1] if len(first) > 1 else first[0]
                else:
                    rob_val = first
        rob_val = float(rob_val)
    return rob_val

__all__ = ["stl_hello_offline"]
