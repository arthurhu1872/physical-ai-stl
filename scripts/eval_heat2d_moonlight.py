"""Audit a saved Heat2D run using MoonLight (if available)."""

from __future__ import annotations

from pathlib import Path
import argparse
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
    ap.add_argument("--frames-dir", type=Path, default=Path("results/heat2d_frames"))
    ap.add_argument("--mls", type=Path, default=Path("scripts/specs/contain_hotspot.mls"))
    ap.add_argument("--nx", type=int, default=16)
    ap.add_argument("--ny", type=int, default=16)
    args = ap.parse_args()

    if load_script_from_file is None:
        print(f"[MoonLight] Skipping audit (moonlight not available): {_moonlight_error}")
        return

    frames = sorted(args.frames_dir.glob("*.npy"))
    if not frames:
        print(f"No frames found in {args.frames_dir}")
        return

    mls = load_script_from_file(str(args.mls))
    mon = get_monitor(mls, "contain")
    graph = build_grid_graph(args.nx, args.ny)

    arrs = [np.load(p) for p in frames]
    u = np.stack(arrs, axis=-1)
    sig = field_to_signal(u, threshold=float(u.mean() + 0.5 * u.std()))
    out = mon.monitor_graph_time_series(graph, sig)
    print("[MoonLight] monitor output (first 3 entries):", out[:3])


if __name__ == "__main__":
    main()