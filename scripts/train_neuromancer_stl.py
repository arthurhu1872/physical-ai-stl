from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------

def _ensure_package_on_path(pkg: str = "physical_ai_stl") -> None:
    try:
        __import__(pkg)
        return
    except Exception:
        pass

    # Try to locate a sibling 'src/<pkg>' relative to this file.
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "src" / pkg
        if candidate.exists():
            sys.path.insert(0, str(parent / "src"))
            try:
                __import__(pkg)
            except Exception as e:  # pragma: no cover
                raise ImportError(f"Found {candidate}, but failed to import {pkg}.") from e
            return
    # If we get here, we will fail later upon import with a clear error.
    # No raise, so users with custom layouts can still succeed.

def _as_jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _as_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_jsonable(v) for v in obj]
    # Fallback to string
    return str(obj)

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main() -> None:
    _ensure_package_on_path()

    # First‑party imports inside the function keep top‑level block
    # strictly future -> stdlib -> third‑party (Ruff/isort‑friendly).
    from physical_ai_stl import about, optional_dependencies
    from physical_ai_stl.utils.seed import seed_everything
    from physical_ai_stl.frameworks import neuromancer_stl_demo as nm_demo
    from physical_ai_stl.frameworks.neuromancer_stl_demo import DemoConfig, train_demo

    ap = argparse.ArgumentParser(
        description="Train a tiny Neuromancer demo with an STL-style bound."
    )
    ap.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    ap.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    ap.add_argument("--bound", type=float, default=0.8, help="Upper safety bound.")
    ap.add_argument("--weight", type=float, default=100.0, help="Penalty weight for violations.")
    ap.add_argument("--n", type=int, default=256, help="Number of training samples.")
    ap.add_argument("--seed", type=int, default=7, help="Global RNG seed for reproducibility.")
    ap.add_argument("--device", type=str, default="cpu", help='PyTorch device ("cpu", "cuda", etc.).')
    ap.add_argument("--mode", choices=["both", "pytorch", "neuromancer"], default="both",
                    help="Which path(s) to run. Default: both.")
    ap.add_argument(
        "--out",
        type=str,
        default="artifacts/neuromancer_stl_results.json",
        help="Where to write the JSON results.",
    )
    ap.add_argument("--quiet", action="store_true", help="Reduce console output.")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print the JSON results on stdout.")
    args = ap.parse_args()

    # Seed everything for reproducibility (safe if torch unavailable).
    seed_everything(args.seed)

    # Build config for the demo module.
    cfg = DemoConfig(
        n=args.n,
        epochs=args.epochs,
        lr=args.lr,
        bound=args.bound,
        weight=args.weight,
        device=args.device,
        seed=args.seed,
    )

    # Small, informative banner.
    if not args.quiet:
        print("=" * 76)
        print("physical-ai-stl: Neuromancer STL demo")
        print("-" * 76)
        print(about())  # one-line env summary
        print("-" * 76)
        print(f"Config: {cfg}")
        print(f"Mode:   {args.mode}")
        print("=" * 76)

    # Run the selected training path(s).
    results: Dict[str, Optional[Dict[str, float]]]
    if args.mode == "both":
        results = train_demo(cfg)  # returns {'pytorch': {...}, 'neuromancer': {...|None}}
    else:
        # Call the private helpers directly to avoid wasted work when the user
        # explicitly requests a single path. Fall back to train_demo if needed.
        try:
            data = nm_demo._make_data(cfg.n, device=cfg.device)  # type: ignore[attr-defined]
            if args.mode == "pytorch":
                pt = nm_demo._train_pytorch(cfg, data)  # type: ignore[attr-defined]
                results = {"pytorch": pt, "neuromancer": None}
            else:
                nm = nm_demo._train_neuromancer(cfg, data)  # type: ignore[attr-defined]
                results = {"pytorch": None, "neuromancer": nm}
        except Exception:
            # Conservative: if internals drift, just use the public API.
            both = train_demo(cfg)
            if args.mode == "pytorch":
                both["neuromancer"] = None
            else:
                both["pytorch"] = None
            results = both

    # Attach a compact environment report (versions/availability of optional deps).
    results_env = optional_dependencies()
    payload: Dict[str, Any] = {"config": cfg.__dict__, "results": results, "env": results_env}

    # Write JSON to disk.
    out_path = Path(args.out)
    if out_path.parent:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(_as_jsonable(payload), f, indent=2, sort_keys=True)
    if not args.quiet:
        print(f"Saved results to {out_path}")

    # Echo results to stdout for convenience.
    pretty = args.pretty or not args.quiet
    if pretty:
        print(json.dumps(_as_jsonable(payload), indent=2, sort_keys=True))
    else:
        print(json.dumps(_as_jsonable(payload)))

if __name__ == "__main__":
    main()
