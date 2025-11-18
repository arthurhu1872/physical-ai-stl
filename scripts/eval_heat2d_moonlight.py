from __future__ import annotations
"""Evaluate a saved 2‑D heat‑equation rollout against a MoonLight STREL spec.

This script is a small, dependency‑light evaluation harness used to *monitor*
spatio‑temporal properties of 2‑D fields (e.g., a Heat2D simulation) with
[MoonLight](https://github.com/MoonLightSuite/moonlight), via the helper
glue in :mod:`physical_ai_stl.monitoring.moonlight_helper`.

Key features
------------
- Accepts either a directory of per‑time `.npy` frames or a single 3‑D `.npy`
  tensor with layout `(nx, ny, nt)` (or `(nt, nx, ny)` via `--layout t_xy`).
- Auto‑detects boolean vs real‑valued semantics from the `.mls` script and
  chooses whether to binarize the field accordingly; can be overridden with
  `--binarize/--no-binarize`.
- Robust, version‑agnostic call into MoonLight monitors (handles
  `monitor_graph_time_series`, `monitorGraphTimeSeries`, and older
  `monitor(...)` variants).
- Fast and memory‑aware: uses memory‑mapped loads for large `.npy` files and
  avoids materializing copies unless necessary.
- Produces a concise human‑readable summary and (optionally) a JSON artifact.

Example
-------
Evaluate the included *contain hotspot* spec (eventually, everywhere cool):

    python scripts/eval_heat2d_moonlight.py \
        --frames-dir results/heat2d \
        --mls scripts/specs/contain_hotspot.mls --formula contain \
        --out-json results/heat2d_moonlight.json

This will **auto‑binarize** the field because the spec declares a boolean
signal, using a threshold computed as `mean + k·std` (see `--z-k`) or a
quantile threshold if `--quantile` is provided.

Notes
-----
- If MoonLight (or the Python/JPype bridge) is unavailable, the script exits
  gracefully with a short message so it can be invoked in environments without
  Java during CI.
- The graph is a 4‑neighborhood grid with configurable edge weight
  (`--adj-weight`) to match common STREL examples.
"""

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

# Optional MoonLight glue — imported lazily & guarded (graceful skip if missing)
try:  # pragma: no cover
    from physical_ai_stl.monitoring.moonlight_helper import (
        build_grid_graph,
        field_to_signal,
        get_monitor,
        load_script_from_file,
    )

    _MOONLIGHT_IMPORT_ERROR: BaseException | None = None
except Exception as exc:  # pragma: no cover
    build_grid_graph = field_to_signal = get_monitor = load_script_from_file = None  # type: ignore
    _MOONLIGHT_IMPORT_ERROR = exc


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _natural_key(p: Path) -> tuple[tuple[int, int | str], ...]:
    """Natural sort helper for files like frame_1.npy, frame_10.npy, ..."""
    s = p.name
    parts = re.split(r"(\d+)", s)
    key: list[tuple[int, int | str]] = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part))
    return tuple(key)


def _find_default_frames_dir() -> Path | None:
    # Common locations in this repo
    candidates = [
        Path("results/heat2d"),  # src/physical_ai_stl/experiments/heat2d.py saves here
        Path("results"),         # broad fallback (used with --glob)
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _glob_frames(frames_dir: Path, pattern: str) -> list[Path]:
    frames = sorted(frames_dir.glob(pattern), key=_natural_key)
    return [p for p in frames if p.suffix == ".npy" and p.is_file()]


def _load_field_from_frames(frames: Sequence[Path]) -> tuple[np.ndarray, int, int, int]:
    if not frames:
        raise FileNotFoundError("No .npy frames found (empty list).")
    # map first to get shape, memory-map the rest for speed & low peak memory
    first = np.load(frames[0], mmap_mode="r")
    if first.ndim != 2:
        raise ValueError(f"Expected 2‑D frame, got shape {first.shape} at {frames[0]}")
    nx, ny = int(first.shape[0]), int(first.shape[1])
    arrs = [np.asarray(first)]
    for p in frames[1:]:
        a = np.load(p, mmap_mode="r")
        if a.shape != (nx, ny):
            raise ValueError(f"Frame shape mismatch at {p}: got {a.shape}, expected {(nx, ny)}")
        arrs.append(np.asarray(a))
    # stack time as last axis: (nx, ny, nt)
    u = np.stack(arrs, axis=-1)
    nt = int(u.shape[-1])
    return u, nx, ny, nt


def _load_field_from_npy(path: Path, layout: str = "xy_t") -> tuple[np.ndarray, int, int, int]:
    a = np.load(path, mmap_mode="r")
    if a.ndim != 3:
        raise ValueError(f"Expected 3‑D array in {path}, got shape {a.shape}")
    if layout == "xy_t":
        nx, ny, nt = int(a.shape[0]), int(a.shape[1]), int(a.shape[2])
        return np.asarray(a), nx, ny, nt
    if layout == "t_xy":
        nt, nx, ny = int(a.shape[0]), int(a.shape[1]), int(a.shape[2])
        # return as (nx, ny, nt) to keep downstream consistent
        return np.asarray(a).transpose(1, 2, 0), nx, ny, nt
    raise ValueError(f"Invalid layout '{layout}'; use 'xy_t' or 't_xy'.")


def _read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def _spec_declares_boolean_signal(mls_text: str) -> bool:
    """Heuristically detect boolean semantics from an MLS script.

    We check whether a `bool` signal is declared and/or `domain boolean;`.
    This mirrors common MoonLight examples and keeps the CLI ergonomic.
    """
    mt_sig = re.search(r"signal\s*\{[^}]*\bbool\b", mls_text, flags=re.IGNORECASE | re.DOTALL)
    mt_dom = re.search(r"domain\s+boolean\s*;", mls_text, flags=re.IGNORECASE)
    return bool(mt_sig or mt_dom)


def _auto_threshold(u: np.ndarray, z_k: float | None, quantile: float | None) -> float:
    flat = np.asarray(u, dtype=float).reshape(-1)
    if quantile is not None:
        if not (0.0 < quantile < 1.0):
            raise ValueError("--quantile must be in (0, 1)")
        return float(np.quantile(flat, quantile))
    # fall back to z-score style threshold
    m = float(flat.mean())
    s = float(flat.std(ddof=0))
    k = 0.5 if z_k is None else float(z_k)
    return float(m + k * s)


def _slice_time(u: np.ndarray, t_start: int | None, t_end: int | None) -> np.ndarray:
    """Slice a field `(nx, ny, nt)` by time indices without copying if possible."""
    if t_start is None and t_end is None:
        return u
    t0 = 0 if t_start is None else int(t_start)
    t1 = u.shape[-1] if t_end is None else int(t_end)
    if t0 < 0 or t1 < 0 or t0 > t1 or t1 > u.shape[-1]:
        raise ValueError(f"Invalid time slice [{t0}:{t1}] for nt={u.shape[-1]}")
    return u[..., t0:t1]


def _monitor_graph_time_series(mon: Any, graph: Any, sig: Any) -> Any:
    """Robustly call into MoonLight monitor across API variants.

    We try several method names and argument patterns to handle different
    MoonLight Python bindings:

    - monitor_graph_time_series(graph, signal)
    - monitorGraphTimeSeries(graph, signal)
    - monitor(graph, signalTimeArray, signalValues)
    - monitor(graph, signal)           (older variants)
    - monitor(signalTimeArray, signal) (non-graph variants, just in case)
    """
    # Derive a simple time array from the number of time steps in `sig`.
    # field_to_signal returns a list-of-lists structure: [t][node][feature].
    try:
        n_times = len(sig)
    except TypeError:
        n_times = None

    time_array = list(range(n_times)) if n_times is not None else None

    last_exc: Exception | None = None

    for name in ("monitor_graph_time_series", "monitorGraphTimeSeries", "monitor"):
        fn = getattr(mon, name, None)
        if not callable(fn):
            continue

        # Candidate call patterns, from most to least specific.
        patterns: list[tuple[Any, ...]] = []

        # Graph + signal only (modern helpers often wrap the time array internally).
        patterns.append((graph, sig))

        # Graph + explicit time array + signal values.
        if time_array is not None:
            patterns.append((graph, time_array, sig))

        # Time array + signal (non-graph variant).
        if time_array is not None:
            patterns.append((time_array, sig))

        # Just the signal (some older scalar specs).
        patterns.append((sig,))

        for args in patterns:
            try:
                return fn(*args)
            except TypeError as e:
                # Wrong signature, try the next pattern.
                last_exc = e
                continue

    # If we get here, nothing worked.
    msg = (
        "Failed to call MoonLight monitor with any known signature. "
        "Last error was: "
        f"{last_exc!r}"
    )
    raise RuntimeError(msg)


def _summarize_spatiotemporal_output(out: object) -> dict:
    arr = np.asarray(out, dtype=float)
    if arr.ndim == 1:
        # single value per time (global)
        per_time = arr
    elif arr.ndim == 2:
        # time x nodes
        per_time = arr.min(axis=1)  # require satisfaction at *all* nodes
    else:
        # unexpected, but try to squeeze
        arr2 = np.squeeze(arr)
        if arr2.ndim == 1:
            per_time = arr2
        elif arr2.ndim == 2:
            per_time = arr2.min(axis=1)
        else:
            raise ValueError(f"Unexpected monitor output shape {arr.shape}")

    # Boolean semantics typically return +/-1; treat >0 as True.
    satisfied_idx = np.flatnonzero(per_time > 0.0)
    satisfied_eventually = bool(satisfied_idx.size > 0)
    first_sat_idx = int(satisfied_idx[0]) if satisfied_eventually else -1

    return {
        "out_shape": tuple(arr.shape),
        "per_time_len": int(per_time.shape[0]),
        "satisfied_eventually": satisfied_eventually,
        "first_satisfaction_index": first_sat_idx,
        # convenience stats for debugging
        "per_time_min": float(np.min(per_time)),
        "per_time_max": float(np.max(per_time)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Audit a saved Heat2D run with a MoonLight STREL spec.",
    )

    # ---- Inputs --------------------------------------------------------------
    src = ap.add_argument_group("input source")
    src.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Directory containing 2‑D .npy frames (one file per time).",
    )
    src.add_argument(
        "--glob",
        type=str,
        default="*.npy",
        help="Glob pattern for frame files inside --frames-dir.",
    )
    src.add_argument(
        "--field",
        type=Path,
        default=None,
        help="Single .npy file with a 3‑D array (nx, ny, nt) or (nt, nx, ny).",
    )
    src.add_argument(
        "--layout",
        type=str,
        default="xy_t",
        choices=("xy_t", "t_xy"),
        help="Axis order of --field if provided.",
    )

    grid = ap.add_argument_group("grid override (usually auto-detected)")
    grid.add_argument("--nx", type=int, default=None, help="Grid size in x (rows).")
    grid.add_argument("--ny", type=int, default=None, help="Grid size in y (cols).")

    spec = ap.add_argument_group("MoonLight spec")
    spec.add_argument(
        "--mls",
        type=Path,
        default=Path("scripts/specs/contain_hotspot.mls"),
        help="Path to a MoonLight .mls script.",
    )
    spec.add_argument(
        "--formula",
        type=str,
        default="contain",
        help="Formula name inside the .mls script.",
    )

    timeg = ap.add_argument_group("time window (optional)")
    timeg.add_argument("--t-start", type=int, default=None, help="Start index (inclusive). ")
    timeg.add_argument("--t-end", type=int, default=None, help="End index (exclusive).")

    graphg = ap.add_argument_group("graph")
    graphg.add_argument("--adj-weight", type=float, default=1.0, help="Edge weight for the grid graph.")

    binz = ap.add_argument_group("binarization (used if spec expects boolean semantics)")
    binz.add_argument(
        "--binarize",
        dest="binarize",
        action="store_true",
        help="Force binary signal (>= threshold → 1 else 0).",
    )
    binz.add_argument(
        "--no-binarize",
        dest="binarize",
        action="store_false",
        help="Force real-valued signal (no thresholding).",
    )
    binz.set_defaults(binarize=None)  # None = decide automatically from spec
    binz.add_argument(
        "--z-k",
        type=float,
        default=0.5,
        help="Threshold = mean + k*std (ignored if --quantile is set).",
    )
    binz.add_argument(
        "--quantile",
        type=float,
        default=None,
        help="If set, threshold = this quantile of all field values (0<q<1).",
    )
    binz.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Absolute threshold; if set it overrides --quantile and --z-k.",
    )

    outg = ap.add_argument_group("output")
    outg.add_argument("--out-json", type=Path, default=None, help="Write a JSON summary to this path.")

    args = ap.parse_args()

    # ---- MoonLight availability guard -----------------------------------------------------------
    if load_script_from_file is None:  # pragma: no cover
        print(f"[MoonLight] Skipping audit (moonlight not available): {_MOONLIGHT_IMPORT_ERROR}")
        sys.exit(0)

    # ---- Load frames ---------------------------------------------------------------------------
    if args.field is not None:
        u, nx, ny, nt = _load_field_from_npy(args.field, layout=args.layout)
    else:
        frames_dir = args.frames_dir or _find_default_frames_dir()
        if frames_dir is None:
            raise FileNotFoundError(
                "No frames source found. Provide --field or --frames-dir (e.g., results/heat2d)."
            )
        frames = _glob_frames(frames_dir, args.glob)
        if not frames:
            raise FileNotFoundError(f"No frames matched {frames_dir}/{args.glob}")
        u, nx, ny, nt = _load_field_from_frames(frames)

    # Optional grid override
    if args.nx is not None:
        nx = int(args.nx)
    if args.ny is not None:
        ny = int(args.ny)

    # Optional time slice (avoid copies; returns a view when possible)
    u = _slice_time(u, args.t_start, args.t_end)
    nt = int(u.shape[-1])

    print(f"[input] field: shape=(nx={nx}, ny={ny}, nt={nt})")
    print(f"[spec]  mls={args.mls}  formula={args.formula}")
    print(f"[graph] 4-neighborhood grid weight={args.adj_weight}")

    # ---- Ensure spec exists; create a minimal default if missing -------------------------------
    if not args.mls.exists():
        args.mls.parent.mkdir(parents=True, exist_ok=True)
        args.mls.write_text(
            "signal { bool hot; }\n"
            "domain boolean;\n"
            "formula contain = eventually (!(somewhere (hot)));\n"
        )
        print(f"[spec] Created default MoonLight spec at {args.mls}")

    mls = load_script_from_file(str(args.mls))
    mon = get_monitor(mls, args.formula)
    graph = build_grid_graph(nx, ny, weight=float(args.adj_weight))

    # ---- Decide binarization based on spec (unless user overrode) ------------------------------
    mls_text = _read_text(args.mls)
    spec_is_boolean = _spec_declares_boolean_signal(mls_text)
    do_binarize: bool = spec_is_boolean if args.binarize is None else bool(args.binarize)

    # ---- Compute threshold if needed -----------------------------------------------------------
    thr: float | None
    if do_binarize:
        if args.threshold is not None:
            thr = float(args.threshold)
        else:
            thr = _auto_threshold(u, z_k=args.z_k, quantile=args.quantile)
    else:
        thr = None  # pass real values (robustness domain)

    # ---- Convert field to MoonLight signal + monitor -------------------------------------------
    # field_to_signal returns nested lists with shape [t][node][feature] to cross the JNI boundary.
    sig = field_to_signal(u, threshold=None if thr is None else float(thr))

    # Version‑agnostic bridge to the underlying Java method(s).
    out = _monitor_graph_time_series(mon, graph, sig)

    # ---- Summarize results ---------------------------------------------------------------------
    summary = _summarize_spatiotemporal_output(out)
    summary.update(
        {
            "nx": nx,
            "ny": ny,
            "nt": nt,
            "frames_source": str(args.field or (args.frames_dir or _find_default_frames_dir() or "")),
            "mls": str(args.mls),
            "formula": str(args.formula),
            "binarized": bool(do_binarize),
            "threshold": None if thr is None else float(thr),
        }
    )

    # Human‑friendly printout
    print("\n[summary]")
    print(f"  output shape: {summary['out_shape']}")
    print(f"  per‑time length: {summary['per_time_len']}")
    if summary["satisfied_eventually"]:
        print("  verdict: PASS — property satisfied at least once")
        print(f"  first satisfaction index: t={summary['first_satisfaction_index']}")
    else:
        print("  verdict: FAIL — property never satisfied over the horizon")
    print(f"  per‑time min/max: {summary['per_time_min']:.3g} .. {summary['per_time_max']:.3g}\n")

    # Optional JSON artifact
    if args.out_json is not None:
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(summary, indent=2))
        print(f"[MoonLight] Wrote JSON summary to {outp}")


if __name__ == "__main__":
    main()
