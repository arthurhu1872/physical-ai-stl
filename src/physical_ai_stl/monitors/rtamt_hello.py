from __future__ import annotations


def _coerce_scalar(rob: object) -> float:
    try:
        # Fast path: already a scalar (int/float/NumPy scalar).
        return float(rob)  # type: ignore[arg-type]
    except Exception:
        # Fallbacks for list/tuple containers seen across RTAMT releases.
        if isinstance(rob, list | tuple):
            if not rob:
                return 0.0
            first = rob[0]
            # Common shape: [(t, value), ...]  → take the value.
            if isinstance(first, list | tuple):
                return float(first[1] if len(first) > 1 else first[0])
            # Otherwise, assume [value, ...].
            return float(first)
        # Last attempt (e.g., custom numeric types).
        return float(rob)  # type: ignore[arg-type]


def stl_hello_offline() -> float:
    # Import inside the function to allow the test to skip when RTAMT is missing.
    try:
        import rtamt  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("rtamt not installed") from e

    # Build a minimal discrete‑time specification: G (u <= 1.0).
    spec = rtamt.StlDiscreteTimeSpecification()
    spec.declare_var("u", "float")
    spec.spec = "always (u <= 1.0)"
    spec.parse()

    # Simple, safe‑to‑evaluate time series (t, u(t)) with unit sampling.
    ts = [(0, 0.2), (1, 0.4), (2, 1.1)]

    # Call evaluate() in a version‑robust way.
    try:
        # Preferred in newer releases: dict of {var: series}.
        rob = spec.evaluate({"u": ts})
    except Exception:
        # Older API variants accept var/series pairs as positional args,
        # e.g., spec.evaluate(['u', ts]) (dense‑time docs show this layout).
        try:
            rob = spec.evaluate(["u", ts])
        except Exception:
            # Very old variant: names list + series list.
            rob = spec.evaluate(["u"], [ts])

    return _coerce_scalar(rob)


__all__ = ["stl_hello_offline"]
