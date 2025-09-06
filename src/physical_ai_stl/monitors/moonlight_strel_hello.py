"""Simple spatio-temporal MoonLight example using STREL."""

from __future__ import annotations
try:
    from physical_ai_stl.monitoring.moonlight_helper import (
        build_grid_graph,
        field_to_signal,
        get_monitor,
        load_script_from_file,
    )
except Exception:  # pragma: no cover
    build_grid_graph = field_to_signal = get_monitor = load_script_from_file = None  # type: ignore


def strel_hello() -> np.ndarray:
    if load_script_from_file is None:
        raise RuntimeError("MoonLight not available")
    mls = load_script_from_file("scripts/specs/contain_hotspot.mls")
    mon = get_monitor(mls, "contain")
    graph = build_grid_graph(3, 3)
    field = np.zeros((3, 3, 2))
    field[1, 1, 0] = 2.0  # a hotspot at t=0
    sig = field_to_signal(field, threshold=1.0)
    out = mon.monitor_graph_time_series(graph, sig)
    return np.array(out, dtype=float)


__all__ = ["strel_hello"]