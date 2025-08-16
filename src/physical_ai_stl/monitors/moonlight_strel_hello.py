"""Simple spatio-temporal MoonLight example using STREL."""

from __future__ import annotations

import numpy as np
from typing import List, Tuple


def _build_grid_graph(n: int) -> Tuple[np.ndarray, np.ndarray]:
    nodes = np.arange(n * n).reshape(n, n)
    edges: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(n):
            v = nodes[i, j]
            if i + 1 < n:
                edges.append((v, nodes[i + 1, j]))
                edges.append((nodes[i + 1, j], v))
            if j + 1 < n:
                edges.append((v, nodes[i, j + 1]))
                edges.append((nodes[i, j + 1], v))
    return nodes, np.array(edges, dtype=int)


def _field_to_signal(u: np.ndarray, threshold: float) -> List[List[List[float]]]:
    n_x, n_y, n_t = u.shape
    n_nodes = n_x * n_y
    signal: List[List[List[float]]] = []
    for t in range(n_t):
        frame = u[:, :, t].reshape(n_nodes, 1)
        frame = (frame >= threshold).astype(float)
        signal.append(frame.tolist())
    return signal


def strel_hello() -> np.ndarray:
    try:
        from moonlight import ScriptLoader  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("moonlight not installed") from e

    with open("scripts/specs/contain_hotspot.mls", "r", encoding="utf-8") as f:
        script = f.read()
    mls = ScriptLoader.loadFromText(script)
    mon = mls.getMonitor("contain")

    graph = _build_grid_graph(3)
    field = np.zeros((3, 3, 2))
    field[1, 1, 0] = 2.0  # a hotspot at t=0
    sig = _field_to_signal(field, threshold=1.0)
    out = mon.monitor_graph_time_series(graph, sig)
    return np.array(out, dtype=float)

__all__ = ["strel_hello"]
