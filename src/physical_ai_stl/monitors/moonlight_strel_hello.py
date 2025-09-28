from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

# Prefer the shared helpers if present (keeps behavior consistent across the repo).
try:  # pragma: no cover - optional dependency path
    # Import names plainly; assign helper aliases below to keep isort happy.
    from physical_ai_stl.monitoring.moonlight_helper import (
        build_grid_graph,
        field_to_signal,
        get_monitor,
        load_script_from_file,
    )
    _helper_build_grid_graph = build_grid_graph
    _helper_field_to_signal = field_to_signal
    _helper_get_monitor = get_monitor
    _helper_load_script_from_file = load_script_from_file
except Exception:  # pragma: no cover
    build_grid_graph = None  # type: ignore[assignment]
    field_to_signal = None  # type: ignore[assignment]
    get_monitor = None  # type: ignore[assignment]
    load_script_from_file = None  # type: ignore[assignment]
    _helper_build_grid_graph = None  # type: ignore[assignment]
    _helper_field_to_signal = None  # type: ignore[assignment]
    _helper_get_monitor = None  # type: ignore[assignment]
    _helper_load_script_from_file = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------

# Location of the demo .mls script within the repository.
_MLS_RELATIVE = ("scripts", "specs", "contain_hotspot.mls")

# Safe fallback for environments where the file is unavailable (e.g. pip installs).
_MLS_INLINE = (
    "signal { bool hot; }\n"
    "domain boolean;\n"
    "formula contain = eventually (!(somewhere (hot)));\n"
)


def _resolve_spec_file() -> Path | None:
    # Allow explicit override via env var
    env = os.environ.get("PHYSICAL_AI_STL_MLS_PATH")
    if env:
        p = Path(env)
        if p.is_file():
            return p

    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        candidate = parent.joinpath(*_MLS_RELATIVE)
        if candidate.is_file():
            return candidate

    # Also try relative to CWD when running ad‑hoc scripts
    cwd_candidate = Path.cwd().joinpath(*_MLS_RELATIVE)
    if cwd_candidate.is_file():
        return cwd_candidate

    return None


def _build_grid_graph_local(nx: int, ny: int) -> list[list[float]]:
    n = nx * ny
    adj = [[0.0] * n for _ in range(n)]

    def idx(i: int, j: int) -> int:
        return i * ny + j

    for i in range(nx):
        for j in range(ny):
            u = idx(i, j)
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < nx and 0 <= jj < ny:
                    v = idx(ii, jj)
                    adj[u][v] = 1.0
                    adj[v][u] = 1.0  # ensure symmetry
    return adj


def _field_to_signal_local(u: np.ndarray, threshold: float | None) -> list[list[list[float]]]:
    if u.ndim != 3:
        raise ValueError(f"Expected a (nx, ny, nt) array; got shape {u.shape}")
    nx, ny, nt = u.shape
    n_nodes = nx * ny
    arr = u.reshape(n_nodes, nt).T
    if threshold is not None:
        arr = (arr >= threshold).astype(float)
    else:
        arr = arr.astype(float)
    # MoonLight expects a feature dimension per node.
    return arr[..., None].tolist()


def _get_monitor_local(mls: Any, formula: str) -> Any:
    # Java binding typically exposes getMonitor(name)
    try:
        return mls.getMonitor(formula)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"MoonLight script does not expose getMonitor('{formula}')") from e


def _monitor_graph_time_series(mon: Any, graph: Any, sig: Any) -> Any:
    # Most recent bindings
    for name in ("monitor_graph_time_series", "monitorGraphTimeSeries"):
        fn = getattr(mon, name, None)
        if callable(fn):
            return fn(graph, sig)
    # Older variants sometimes use a generic 'monitor' for both forms
    fn = getattr(mon, "monitor", None)
    if callable(fn):  # pragma: no cover - only hit on old releases
        try:
            return fn(graph, sig)
        except TypeError:
            # Some ancient versions might flip the order
            return fn(sig, graph)
    raise RuntimeError("MoonLight STREL monitor: no compatible monitor method found.")


def _to_ndarray(out: Any) -> np.ndarray:
    try:
        arr = np.asarray(out, dtype=float)
        return arr  # (N, 2) in the common case
    except Exception:  # pragma: no cover
        # Fallback: take first value if a mapping {node -> series}
        if isinstance(out, dict) and out:
            first = next(iter(out.values()))
            return _to_ndarray(first)
        raise


def strel_hello() -> np.ndarray:
    # If the optional dependency isn't present, fail fast with a clear error.
    if load_script_from_file is None:
        raise RuntimeError("MoonLight not available; install with `pip install moonlight`.")

    # Load the STREL script (from file if available; otherwise use inline fallback).
    spec_path = _resolve_spec_file()
    if spec_path is not None:
        mls = _helper_load_script_from_file(str(spec_path))  # type: ignore[arg-type]
    else:  # pragma: no cover - rare when packaging w/o scripts/
        # Fall back to compiling the in‑memory script.
        from moonlight import ScriptLoader  # type: ignore
        mls = ScriptLoader.loadFromText(_MLS_INLINE)

    # Build a tiny 3×3 grid graph and a 2‑frame field with a transient hotspot.
    build_graph = _helper_build_grid_graph or _build_grid_graph_local
    to_signal = _helper_field_to_signal or _field_to_signal_local
    get_mon = _helper_get_monitor or _get_monitor_local

    graph = build_graph(3, 3)
    field = np.zeros((3, 3, 2), dtype=float)
    field[1, 1, 0] = 2.0  # hotspot at the center at t=0
    sig = to_signal(field, threshold=1.0)

    mon = get_mon(mls, "contain")
    out = _monitor_graph_time_series(mon, graph, sig)

    return _to_ndarray(out)


__all__ = ["strel_hello"]
