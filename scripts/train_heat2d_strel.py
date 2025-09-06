"""Train heat2d PINN and optionally audit with a MoonLight spec."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import argparse

from physical_ai_stl.training.grids import grid2d
try:
    from physical_ai_stl.monitoring.moonlight_helper import (
        build_grid_graph,
        field_to_signal,
        get_monitor,
        load_script_from_file,
    )
except Exception as exc:  # pragma: no cover
    build_grid_graph = field_to_signal = get_monitor = load_script_from_file = None  # type: ignore
    _moonlight_error = exc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--nx", type=int, default=16)
    ap.add_argument("--ny", type=int, default=16)
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--mls", type=Path, default=Path("scripts/specs/contain_hotspot.mls"))
    args = ap.parse_args()

    # placeholder training using numpy grids only (no torch dependency here)
    _ = grid2d(args.nx, args.ny)

    if args.audit:
        if load_script_from_file is None:
            print(f"[MoonLight] Skipping audit (moonlight not available): {_moonlight_error}")
            return
        frames = [np.zeros((args.nx, args.ny)) for _ in range(3)]
        u = np.stack(frames, axis=-1)
        mls = load_script_from_file(str(args.mls))
        mon = get_monitor(mls, "contain")
        graph = build_grid_graph(args.nx, args.ny)
        sig = field_to_signal(u, threshold=0.1)
        out = mon.monitor_graph_time_series(graph, sig)
        print("[MoonLight] monitor output (first 3 entries):", out[:3])


if __name__ == "__main__":
    main()