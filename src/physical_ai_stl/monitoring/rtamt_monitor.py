from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Tuple, Dict, List, Any, Optional

# ----- Internal utilities ----------------------------------------------------

_RTAMT = None  # cached module to avoid repeated imports


def _import_rtamt():
    global _RTAMT
    if _RTAMT is not None:
        return _RTAMT
    try:
        import rtamt  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "rtamt is not installed. Install it with 'pip install rtamt' "
            "or see https://github.com/nickovic/rtamt for build notes."
        ) from e
    _RTAMT = rtamt
    return rtamt


# Treat "time series" inputs liberally:
# - [v0, v1, ...] with uniform dt
# - [(t0, v0), (t1, v1), ...] explicit timestamps
TimeSeries = Iterable[float] | Iterable[Tuple[float, float]]


def _normalize_series(series: TimeSeries, dt: float | None) -> list[tuple[float, float]]:
    it = iter(series)
    try:
        first = next(it)
    except StopIteration:
        return []
    # Explicit timestamps given?
    if isinstance(first, (list, tuple)) and len(first) >= 2:
        t0, v0 = first[0], first[1]
        out = [(float(t0), float(v0))]
        for el in it:
            t, v = el  # type: ignore[misc]
            out.append((float(t), float(v)))
        return out
    # Otherwise: treat as regularly sampled values.
    step = 1.0 if dt is None else float(dt)
    out = [(0.0, float(first))]
    k = 1
    for v in it:
        out.append((k * step, float(v)))
        k += 1
    return out


def _coerce_scalar(rob: object) -> float:
    # Fast path: already a numeric scalar (float/int/NumPy scalar).
    try:
        return float(rob)  # type: ignore[arg-type]
    except Exception:
        pass

    # Common container fallbacks.
    if isinstance(rob, (list, tuple)):
        if not rob:
            return 0.0
        first = rob[0]
        # Shape: [(t, value), ...] → take the value
        if isinstance(first, (list, tuple)):
            return float(first[1] if len(first) > 1 else first[0])
        # Shape: [value, ...]
        return float(first)
    # Last attempt (custom numeric types).
    return float(rob)  # type: ignore[arg-type]


# ----- Spec builders ---------------------------------------------------------

def stl_always_upper_bound(var: str = "u", u_max: float = 1.0,
                           *, time_semantics: str = "dense"):
    rtamt = _import_rtamt()
    if time_semantics == "dense":
        Spec = getattr(rtamt, 'StlDenseTimeSpecification', None) or getattr(rtamt, 'StlDenseTimeOfflineSpecification')
        spec = Spec()
    else:
        spec = rtamt.StlDiscreteTimeSpecification()
    spec.declare_var(var, "float")
    spec.spec = f"always ({var} <= {float(u_max)})"
    spec.parse()
    return spec


def stl_response_within(var: str, boundary: str, theta: float, tau: int,
                        *, time_semantics: str = "dense"):
    rtamt = _import_rtamt()
    if time_semantics == "dense":
        SpecCls = getattr(rtamt, 'StlDenseTimeSpecification', None) or getattr(rtamt, 'StlDenseTimeOfflineSpecification')
    else:
        SpecCls = rtamt.StlDiscreteTimeSpecification
    spec = SpecCls()
    spec.declare_var(var, "float")
    spec.declare_var(boundary, "float")
    spec.spec = (f"always ( ({boundary} >= {float(theta)}) -> "
                 f"eventually[0:{int(tau)}] ({var} >= {float(theta)}) )")
    spec.parse()
    return spec


# ----- Evaluation helpers ----------------------------------------------------

def evaluate_series(spec: Any, var: str, series: TimeSeries, *, dt: float = 1.0) -> float:
    ts = _normalize_series(series, dt)
    # Try the modern fastest path first: dict mapping.
    try:
        rob = spec.evaluate({var: ts})
    except Exception:
        # Fallbacks for older API signatures.
        try:
            rob = spec.evaluate([var, ts])
        except Exception:
            rob = spec.evaluate([var], [ts])
    return _coerce_scalar(rob)


def evaluate_multi(spec: Any,
                   data: Mapping[str, TimeSeries] | Sequence[tuple[str, TimeSeries]],
                   *, dt: float | Mapping[str, float] = 1.0) -> float:
    if isinstance(data, Mapping):
        items = list(data.items())
    else:
        items = list(data)

    dt_map: Mapping[str, float]
    if isinstance(dt, Mapping):
        dt_map = dt
    else:
        dt_map = {name: float(dt) for name, _ in items}

    # Normalize all series to timestamped lists.
    series_map: Dict[str, List[tuple[float, float]]] = {
        name: _normalize_series(s, dt_map.get(name))
        for name, s in items
    }

    # Try evaluate with the most ergonomic modern signature first.
    try:
        rob = spec.evaluate(series_map)
    except Exception:
        # Fallback: positional pairs.
        try:
            pairs = [[name, series_map[name]] for name, _ in items]
            rob = spec.evaluate(*pairs)  # type: ignore[misc]
        except Exception:
            names = [name for name, _ in items]
            series_list = [series_map[name] for name, _ in items]
            rob = spec.evaluate(names, series_list)

    return _coerce_scalar(rob)


def satisfied(robustness: float) -> bool:
    return float(robustness) >= 0.0


__all__ = [
    "stl_always_upper_bound",
    "stl_response_within",
    "evaluate_series",
    "evaluate_multi",
    "satisfied",
]
