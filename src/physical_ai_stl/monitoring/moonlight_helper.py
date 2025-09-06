"""Helper functions for spatio-temporal monitoring using MoonLight."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
def _import_moonlight():
    try:
        from moonlight import ScriptLoader  # type: ignore
    except Exception:  # pragma: no cover
        raise ImportError("moonlight is not installed; 'pip install moonlight'.") from e
    return ScriptLoader

def load_script_from_file(path: str):
    """Load a MoonLight script from text file."""
    ScriptLoader = _import_moonlight()
    with open(path, encoding="utf-8") as f:
        return ScriptLoader.loadFromText(f.read())

def get_monitor(mls, name: str):
    """Get a monitor by name from a loaded script."""
    return mls.getMonitor(name)

def build_grid_graph(n_x: int, n_y: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (nodes, edges) for an n_x by n_y grid graph."""
    nodes = np.arange(n_x * n_y).reshape(n_x, n_y)
    edges: list[tuple[int, int]] = []
    for i in range(n_x):
        for j in range(n_y):
            v = nodes[i, j]
            if i + 1 < n_x:
                edges.append((v, nodes[i + 1, j]))
                edges.append((nodes[i + 1, j], v))
            if j + 1 < n_y:
                edges.append((v, nodes[i, j + 1]))
                edges.append((nodes[i, j + 1], v))
    return nodes, np.array(edges, dtype=int)

def field_to_signal(u: np.ndarray, threshold: float | None = None) -> list[list[list[float]]]:
    """Convert a (n_x, n_y, n_t) field to MoonLight's node-wise signal format."""
    n_x, n_y, n_t = u.shape
    n_nodes = n_x * n_y
    if threshold is not None:
        signal_array = (u.reshape(n_nodes, n_t).T >= threshold).astype(float)
    else:
        signal_array = u.reshape(n_nodes, n_t).T.astype(float)
    # Add a feature dimension of size 1 for each node, then convert to nested lists
    signal_list: list[list[list[float]]] = signal_array[..., None].tolist()
    return signal_list

__all__ = [
    "load_script_from_file",
    "get_monitor",
    "build_grid_graph",
    "field_to_signal",
]