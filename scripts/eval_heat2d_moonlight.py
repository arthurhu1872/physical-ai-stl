from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
import argparse
import json
import re
import sys

import numpy as np

# Optional MoonLight glue — imported lazily & guarded (graceful skip if missing)
try:  # pragma: no cover
    from physical_ai_stl.monitoring.moonlight_helper import (
        build_grid_graph,
        field_to_signal,
        get_monitor,
        load_script_from_file,
    )
    _MOONLIGHT_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:  # pragma: no cover
    build_grid_graph = field_to_signal = get_monitor = load_script_from_file = None  # type: ignore
    _MOONLIGHT_IMPORT_ERROR = exc


# ---------------------------------------------------------------------------

def _natural_key(p: Path) -> Tuple:
    s = p.name
    parts = re.split(r"(\d+)", s)
    key: List[Tuple[int, object]] = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part))
    return tuple(key)


def _find_default_frames_dir() -> Optional[Path]:
    candidates = [
        Path("results/heat2d_frames"),  # older/default
        Path("results/heat2d"),         # experiment/heat2d.py saves here
        Path("results"),                # broad fallback (used with --glob)
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _glob_frames(frames_dir: Path, pattern: str) -> List[Path]:
    frames = sorted(frames_dir.glob(pattern), key=_natural_key)
    return [p for p in frames if p.suffix == ".npy" and p.is_file()]


def _load_field_from_frames(frames: Sequence[Path]) -> Tuple[np.ndarray, int, int, int]:
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


def _load_field_from_npy(path: Path, layout: str = "xy_t") -> Tuple[np.ndarray, int, int, int]:
    a = np.load(path, mmap_mode="r")
    if a.ndim != 3:
        raise ValueError(f"Expected 3‑D array in {path}, got shape {a.shape}")
    if layout == "xy_t":
        nx, ny, nt = int(a.shape[0]), int(a.shape[1]), int(a.shape[2])
        return np.asarray(a), nx, ny, nt
    elif layout == "t_xy":
        nt, nx, ny = int(a.shape[0]), int(a.shape[1]), int(a.shape[2])
        # return as (nx, ny, nt) to keep downstream consistent
        return np.asarray(a).transpose(1, 2, 0), nx, ny, nt
    else:
        raise ValueError(f"Invalid layout '{layout}'; use 'xy_t' or 't_xy'.")


def _read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def _spec_declares_boolean_signal(mls_text: str) -> bool:
    # Also check for 'domain boolean;' which sets Boolean semantics.
    mt_sig = re.search(r"signal\s*\{[^}]*\bbool\b", mls_text, flags=re.IGNORECASE | re.DOTALL)
    mt_dom = re.search(r"domain\s+boolean\s*;", mls_text, flags=re.IGNORECASE)
    return bool(mt_sig or mt_dom)


def _auto_threshold(u: np.ndarray, z_k: float | None, quantile: float | None) -> float:
    flat = u.reshape(-1).astype(float)
    if quantile is not None:
        if not (0.0 < quantile < 1.0):
            raise ValueError("--quantile must be in (0, 1)")
        return float(np.quantile(flat, quantile))
    # fall back to z-score style threshold
    m = float(flat.mean())
    s = float(flat.std(ddof=0))
    k = 0.5 if z_k is None else float(z_k)
    return float(m + k * s)


def _summarize_spatiotemporal_output(out: object) -> dict:
    arr = np.asarray(out, dtype=float)
    if arr.ndim == 1:
        # single value per time (global)
        per_time = arr
        per_node = None
    elif arr.ndim == 2:
        # time x nodes
        per_time = arr.min(axis=1)  # require satisfaction at all nodes
        per_node = arr
    else:
        # unexpected, but try to squeeze
        arr2 = np.squeeze(arr)
        if arr2.ndim == 1:
            per_time = arr2
            per_node = None
        elif arr2.ndim == 2:
            per_time = arr2.min(axis=1)
            per_node = arr2
        else:
            raise ValueError(f"Unexpected monitor output shape {arr.shape}")

    # Boolean semantics typically return +/-1; treat >0 as True.
    sat_mask = per_time > 0.0
    satisfied_eventually = bool(sat_mask.any())
    first_sat_idx = int(np.argmax(sat_mask)) if satisfied_eventually else -1

    return dict(
        out_shape=tuple(arr.shape),
        per_time_len=int(per_time.shape[0]),
        satisfied_eventually=satisfied_eventually,
        first_satisfaction_index=first_sat_idx,
        # convenience stats for debugging
        per_time_min=float(np.min(per_time)),
        per_time_max=float(np.max(per_time)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Audit a saved Heat2D run with a MoonLight STREL spec."
    )
    src = ap.add_argument_group("input source")
    src.add_argument("--frames-dir", type=Path, default=None,
                     help="Directory containing 2‑D .npy frames (one file per time).")
    src.add_argument("--glob", type=str, default="*.npy",
                     help="Glob pattern for frame files inside --frames-dir.")
    src.add_argument("--field", type=Path, default=None,
                     help="Single .npy file with a 3‑D array (nx, ny, nt) or (nt, nx, ny).")
    src.add_argument("--layout", type=str, default="xy_t", choices=("xy_t", "t_xy"),
                     help="Axis order of --field if provided.")

    grid = ap.add_argument_group("grid override (usually auto-detected)")
    grid.add_argument("--nx", type=int, default=None, help="Grid size in x (rows).")
    grid.add_argument("--ny", type=int, default=None, help="Grid size in y (cols).")

    spec = ap.add_argument_group("MoonLight spec")
    spec.add_argument("--mls", type=Path, default=Path("scripts/specs/contain_hotspot.mls"),
                      help="Path to a MoonLight .mls script.")
    spec.add_argument("--formula", type=str, default="contain",
                      help="Formula name inside the .mls script.")

    binz = ap.add_argument_group("binarization (used if spec expects boolean semantics)")
    binz.add_argument("--binarize", dest="binarize", action="store_true",
                      help="Force binary signal (>= threshold → 1 else 0).")
    binz.add_argument("--no-binarize", dest="binarize", action="store_false",
                      help="Force real-valued signal (no thresholding).")
    binz.set_defaults(binarize=None)  # None = decide automatically from spec
    binz.add_argument("--z-k", type=float, default=0.5,
                      help="Threshold = mean + k*std (ignored if --quantile is set).")
    binz.add_argument("--quantile", type=float, default=None,
                      help="If set, threshold = this quantile of all field values (0<q<1).")
    binz.add_argument("--threshold", type=float, default=None,
                      help="Absolute threshold; if set it overrides --quantile and --z-k.")

    outg = ap.add_argument_group("output")
    outg.add_argument("--out-json", type=Path, default=None, help="Write a JSON summary to this path.")

    args = ap.parse_args()

    # ---- MoonLight availability guard -----------------------------------------------------------
    if load_script_from_file is None:  # pragma: no cover
        print(f"[MoonLight] Skipping audit (moonlight not available): {_MOONLIGHT_IMPORT_ERROR}")
        sys.exit(0)

    # ---- Load frames ---------------------------------------------------------------------------
    u: np.ndarray
    nx: int
    ny: int
    nt: int

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

    # Optional explicit grid override
    if args.nx is not None and args.nx != nx:
        print(f"[warn] Overriding nx: detected {nx}, using {args.nx}")
        nx = int(args.nx)
    if args.ny is not None and args.ny != ny:
        print(f"[warn] Overriding ny: detected {ny}, using {args.ny}")
        ny = int(args.ny)

    # ---- Ensure spec exists; create a minimal default if missing -------------------------------
    if not args.mls.exists():
        args.mls.parent.mkdir(parents=True, exist_ok=True)
        args.mls.write_text("signal { bool hot; }\ndomain boolean;\nformula contain = eventually (!(somewhere (hot)));\n")
        print(f"[spec] Created default MoonLight spec at {args.mls}")

    mls = load_script_from_file(str(args.mls))
    mon = get_monitor(mls, args.formula)
    graph = build_grid_graph(nx, ny)

    # ---- Decide binarization based on spec (unless user overrode) ------------------------------
    mls_text = _read_text(args.mls)
    spec_is_boolean = _spec_declares_boolean_signal(mls_text)
    do_binarize: bool = spec_is_boolean if args.binarize is None else bool(args.binarize)

    # ---- Compute threshold if needed -----------------------------------------------------------
    thr: Optional[float]
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

    # Modern MoonLight Python interface supports a convenience method for STREL graph+signal.
    # Our helper makes this uniform across versions.
    out = mon.monitor_graph_time_series(graph, sig)  # type: ignore[attr-defined]

    # ---- Summarize results ---------------------------------------------------------------------
    summary = _summarize_spatiotemporal_output(out)
    summary.update(
        dict(
            nx=nx, ny=ny, nt=nt,
            frames_source=str(args.field or (args.frames_dir or _find_default_frames_dir() or "")),
            mls=str(args.mls), formula=str(args.formula),
            binarized=bool(do_binarize),
            threshold=None if thr is None else float(thr),
        )
    )

    # Human-friendly printout
    print("\n[MoonLight] STREL audit summary")
    print("  grid:", f"{nx} x {ny}", "  time steps:", nt)
    print("  spec:", f"{args.mls.name}  (formula: {args.formula})")
    print("  binarize:", do_binarize, "  threshold:", "n/a" if thr is None else f"{thr:.6g}")
    print("  output shape:", summary["out_shape"])
    if summary["satisfied_eventually"]:
        print(f"  verdict: PASS — property satisfied at some time index t={summary['first_satisfaction_index']}")
    else:
        print("  verdict: FAIL — property never satisfied over the horizon")
    print(f"  per-time min/max: {summary['per_time_min']:.3g} .. {summary['per_time_max']:.3g}\n")

    # Optional JSON artifact
    if args.out_json is not None:
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(summary, indent=2))
        print(f"[MoonLight] Wrote JSON summary to {outp}")


if __name__ == "__main__":
    main()
