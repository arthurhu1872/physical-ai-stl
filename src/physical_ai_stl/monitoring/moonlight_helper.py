"""Helper functions for spatio-temporal monitoring using MoonLight.

MoonLight exposes a Python API for loading STREL scripts and evaluating
formulae on signals defined over time and space.  This module wraps the
common patterns used in the week-2 example: loading a script from a
file, building a regular grid graph, converting field tensors to the
signal format expected by MoonLight, and retrieving monitors.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from moonlight import ScriptLoader  # type: ignore


def load_script_from_file(path: str):
    """Load a MoonLight script from a file on disk."""
    return ScriptLoader.loadFromFile(path)


def get_monitor(script, formula_name: str):
    """Retrieve a monitor for the named formula from a loaded script."""
    return script.getMonitor(formula_name)


def build_grid_graph(n_x: int, n_y: int, weight: float = 1.0) -> Tuple[List[List[List[float]]], List[float]]:
    """Build an undirected 2D grid graph for MoonLight.

    Returns a list of edges for a single time step and a dummy list of times.

    Parameters
    ----------
    n_x, n_y : int
        Number of nodes along the x and y axes.
    weight : float
        Edge weight assigned to each grid adjacency.

    Returns
    -------
    Tuple[List[List[List[float]]], List[float]]
        ``(graph, times)`` where ``graph`` is a single-element list containing
        all undirected edges as triples ``[i, j, weight]`` and ``times`` is a
        single-element list with a dummy time stamp.
    """
    edges: List[List[float]] = []
    def idx(i: int, j: int) -> int:
        return i * n_y + j
    for i in range(n_x):
        for j in range(n_y):
            if i + 1 < n_x:
                edges.append([float(idx(i, j)), float(idx(i + 1, j)), float(weight)])
                edges.append([float(idx(i + 1, j)), float(idx(i, j)), float(weight)])
            if j + 1 < n_y:
                edges.append([float(idx(i, j)), float(idx(i, j + 1)), float(weight)])
                edges.append([float(idx(i, j + 1)), float(idx(i, j)), float(weight)])
    # Graph is static in time: single snapshot
    return [edges], [0.0]


def field_to_signal(u: np.ndarray, threshold: float | None = None) -> List[List[List[float]]]:
    """Convert a 3D field array into a MoonLight signal.

    Parameters
    ----------
    u : np.ndarray
        Array of shape ``(n_x, n_y, n_t)``.
    threshold : float or None
        If provided, produces a boolean signal where entries are 1.0 when
        ``u >= threshold`` and 0.0 otherwise.  If ``None``, the raw values
        are returned as a real-valued signal.

    Returns
    -------
    List[List[List[float]]]
        Nested list of shape ``(n_t, n_nodes, 1)`` representing the signal
        per node per time step.
    """
    n_x, n_y, n_t = u.shape
    n_nodes = n_x * n_y
    signal: List[List[List[float]]] = []
    for t in range(n_t):
        frame = u[:, :, t].reshape(n_nodes, 1)
        if threshold is not None:
            frame = (frame >= threshold).astype(float)
        signal.append(frame.tolist())
    return signal
