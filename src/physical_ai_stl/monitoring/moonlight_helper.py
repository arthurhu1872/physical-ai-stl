from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Sequence, Tuple

import numpy as _np


# ---------------------------------------------------------------------------
# Lazy MoonLight import
# ---------------------------------------------------------------------------
def _import_moonlight():
    try:
        from moonlight import ScriptLoader  # type: ignore
    except Exception as e:  # pragma: no cover - exercised only when missing
        raise ImportError(
            "MoonLight is not available. Install with 'pip install moonlight'\n"
            "and ensure a compatible Java runtime (Java 21+ is recommended)."
        ) from e
    return ScriptLoader


# ---------------------------------------------------------------------------
# Loading scripts and getting monitors
# ---------------------------------------------------------------------------
def load_script_from_file(path: str | Path):
    ScriptLoader = _import_moonlight()
    text = Path(path).read_text(encoding="utf-8")
    return ScriptLoader.loadFromText(text)


def get_monitor(mls, name: str):
    try:
        return mls.getMonitor(name)
    except Exception as e:
        raise KeyError(f"MoonLight formula not found: {name!r}") from e


# ---------------------------------------------------------------------------
# Grid graph utilities
# ---------------------------------------------------------------------------
def _grid_adjacency(nx: int, ny: int, weight: float) -> List[List[float]]:
    if nx <= 0 or ny <= 0:
        raise ValueError(f"grid dimensions must be positive, got nx={nx}, ny={ny}")
    n = nx * ny
    adj = [[0.0] * n for _ in range(n)]

    def idx(i: int, j: int) -> int:
        return i * ny + j

    for i in range(nx):
        for j in range(ny):
            u = idx(i, j)
            # 4-neighborhood: up, down, left, right
            if i > 0:
                v = idx(i - 1, j)
                adj[u][v] = weight
                adj[v][u] = weight
            if i + 1 < nx:
                v = idx(i + 1, j)
                adj[u][v] = weight
                adj[v][u] = weight
            if j > 0:
                v = idx(i, j - 1)
                adj[u][v] = weight
                adj[v][u] = weight
            if j + 1 < ny:
                v = idx(i, j + 1)
                adj[u][v] = weight
                adj[v][u] = weight
    return adj


def _grid_triples(nx: int, ny: int, weight: float) -> List[List[float]]:
    if nx <= 0 or ny <= 0:
        raise ValueError(f"grid dimensions must be positive, got nx={nx}, ny={ny}")
    triples: List[List[float]] = []

    def idx(i: int, j: int) -> int:
        return i * ny + j

    for i in range(nx):
        for j in range(ny):
            u = idx(i, j)
            if i + 1 < nx:
                v = idx(i + 1, j)
                triples.append([float(u), float(v), float(weight)])
                triples.append([float(v), float(u), float(weight)])
            if j + 1 < ny:
                v = idx(i, j + 1)
                triples.append([float(u), float(v), float(weight)])
                triples.append([float(v), float(u), float(weight)])
    return triples


def build_grid_graph(
    n_x: int,
    n_y: int,
    *,
    weight: float = 1.0,
    return_format: Literal["adjacency", "triples", "nodes_edges"] = "adjacency",
) -> (
    List[List[float]]
    | List[List[float]]
    | Tuple[_np.ndarray, _np.ndarray]
):
    if return_format == "adjacency":
        return _grid_adjacency(n_x, n_y, float(weight))
    elif return_format == "triples":
        return _grid_triples(n_x, n_y, float(weight))
    elif return_format == "nodes_edges":
        # Nodes as a shaped grid and directed edges (u,v) in integer dtype.
        nodes = _np.arange(n_x * n_y, dtype=_np.int64).reshape(n_x, n_y)
        edges: list[tuple[int, int]] = []
        for i in range(n_x):
            for j in range(n_y):
                v = int(nodes[i, j])
                if i + 1 < n_x:
                    edges.append((v, int(nodes[i + 1, j])))
                    edges.append((int(nodes[i + 1, j]), v))
                if j + 1 < n_y:
                    edges.append((v, int(nodes[i, j + 1])))
                    edges.append((int(nodes[i, j + 1]), v))
        return nodes, _np.asarray(edges, dtype=_np.int64)
    else:  # pragma: no cover
        raise ValueError(f"Unknown return_format: {return_format!r}")


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------
def _as_py_float_nested(a: _np.ndarray) -> List[List[List[float]]]:
    # Ensure we have shape (T, N, F)
    if a.ndim == 2:
        a = a[:, :, None]
    if a.ndim != 3:
        raise ValueError(f"Expected 2D/3D array, got shape {tuple(a.shape)}")
    # Avoid materializing a copy unless dtype/contiguity requires it.
    a = _np.asarray(a, dtype=float, order="C")
    # Convert to nested lists of native floats.
    return a.tolist()  # NumPy casts to Python float during tolist()


def field_to_signal(
    u: _np.ndarray,
    threshold: float | None = None,
    *,
    layout: Literal["xy_t", "t_xy"] = "xy_t",
) -> List[List[List[float]]]:
    a = _np.asarray(u)
    if layout == "xy_t":
        if a.ndim != 3:
            raise ValueError(f"Expected (nx, ny, nt) for layout 'xy_t'; got {a.shape}")
        nx, ny, nt = a.shape
        flat = a.reshape(nx * ny, nt).T  # (T, N)
    elif layout == "t_xy":
        if a.ndim != 3:
            raise ValueError(f"Expected (nt, nx, ny) for layout 't_xy'; got {a.shape}")
        nt, nx, ny = a.shape
        flat = a.reshape(nt, nx * ny)  # (T, N)
    else:  # pragma: no cover
        raise ValueError(f"Unknown layout {layout!r}")

    if threshold is not None:
        flat = (flat >= threshold).astype(float, copy=False)
    else:
        flat = flat.astype(float, copy=False)

    return _as_py_float_nested(flat)  # adds feature dim


__all__ = [
    "load_script_from_file",
    "get_monitor",
    "build_grid_graph",
    "field_to_signal",
]
