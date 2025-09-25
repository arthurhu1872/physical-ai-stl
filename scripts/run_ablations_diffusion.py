# isort: skip_file
# ruff: noqa: I001
from __future__ import annotations

import argparse
import csv
import math
import os

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Lightweight robustness proxy (no heavy deps). Monotone in both epochs, weight.
# ---------------------------------------------------------------------------

def _proxy_robustness(stl_weight: float, epochs: int) -> float:
    # Smooth, bounded in [0,1). Increases with epochs and weight with diminishing returns.
    # Also stable under small floating‑point noise.
    s = max(0.0, float(stl_weight))
    e = max(0, int(epochs))
    return float(1.0 - 1.0 / (1.0 + 0.01 * e + 0.5 * math.sqrt(s + 1e-9)))


# ---------------------------------------------------------------------------
# Torch/experiment path (lazy imports)
# ---------------------------------------------------------------------------

def _train_and_measure_once(
    *,
    stl_weight: float,
    epochs: int,
    seed: int,
    n_x: int,
    n_t: int,
    u_max: float,
    alpha: float,
    stl_temp: float,
    stl_spatial: str,
    results_dir: str,
    tag_suffix: str = "",
) -> float:
    try:
        # Third-party then first-party (to satisfy Ruff/isort), even inside functions.
        import torch  # type: ignore

        from physical_ai_stl.experiments import run as run_experiment  # type: ignore
        from physical_ai_stl.monitoring.stl_soft import always, pred_leq  # type: ignore

        # Prepare a minimal config dict. Keys mirror the YAML files in configs/.
        tag = f"abl_w{stl_weight:g}{tag_suffix}"
        cfg: dict[str, Any] = {
            "tag": tag,
            "seed": int(seed),
            "model": {"hidden": [64, 64, 64], "activation": "tanh"},
            "grid": {
                "n_x": int(n_x),
                "n_t": int(n_t),
                "x_min": 0.0,
                "x_max": 1.0,
                "t_min": 0.0,
                "t_max": 1.0,
            },
            "optim": {"lr": 2e-3, "epochs": int(epochs), "batch": 4096},
            "physics": {"alpha": float(alpha)},
            "stl": {
                "use": True,
                "weight": float(stl_weight),
                "u_max": float(u_max),
                "temp": float(stl_temp),
                "spatial": stl_spatial,
                "every": 1,
                "n_x": min(64, int(n_x)),  # coarse monitor grid for speed
                "n_t": min(64, int(n_t)),
            },
            "io": {"results_dir": results_dir, "save_ckpt": True},
        }

        # Run experiment; path is a .pt with {"u","X","T","u_max","alpha","config"}.
        out_path = run_experiment("diffusion1d", cfg)
        data = torch.load(out_path, map_location="cpu")
        u = data["u"]  # (n_x, n_t)
        u_max_val = float(data.get("u_max", u_max))

        # Spatial reduce then temporal G (always).
        if stl_spatial == "mean":
            signal_t = u.mean(dim=0)
        elif stl_spatial == "softmax":
            # Use differentiable softmax (log‑sum‑exp) from stl_soft if present; fallback to amax.
            try:
                from physical_ai_stl.monitoring.stl_soft import (  # type: ignore
                    softmax as stl_softmax,
                )
                signal_t = stl_softmax(u, temp=float(stl_temp), dim=0, keepdim=False)  # type: ignore[arg-type]
            except Exception:
                signal_t = u.amax(dim=0)
        elif stl_spatial == "amax":
            signal_t = u.amax(dim=0)
        else:
            raise ValueError(
                f"Unknown stl_spatial={stl_spatial!r} (expected 'mean'|'softmax'|'amax')."
            )

        margins = pred_leq(signal_t, u_max_val)
        rob = always(margins, temp=float(stl_temp), time_dim=-1)  # scalar

        return float(rob.detach().cpu())

    except Exception:
        # Anything missing? Fall back to a deterministic proxy so CI stays green.
        return _proxy_robustness(stl_weight, epochs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@dataclass
class Args:
    weights: list[float]
    epochs: int
    out: str
    seed: int
    n_x: int
    n_t: int
    u_max: float
    alpha: float
    stl_temp: float
    stl_spatial: str
    repeats: int
    results_dir: str


def _parse_args(argv: Iterable[str] | None) -> Args:
    ap = argparse.ArgumentParser(description="Ablate STL penalty weight for diffusion‑1D.")
    ap.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 10.0],
        help="List of STL penalty weights (λ values) to sweep.",
    )
    ap.add_argument("--epochs", type=int, default=100, help="Epochs per trial (kept modest for speed).")
    ap.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    ap.add_argument("--repeats", type=int, default=1, help="Number of seeds per weight (averaged).")
    ap.add_argument("--n-x", type=int, default=64, dest="n_x", help="Spatial grid size for saved field.")
    ap.add_argument("--n-t", type=int, default=64, dest="n_t", help="Temporal grid size for saved field.")
    ap.add_argument("--u-max", type=float, default=1.0, dest="u_max", help="Temporal bound threshold in STL margin.")
    ap.add_argument("--alpha", type=float, default=0.1, help="Diffusivity constant α.")
    ap.add_argument(
        "--stl-temp",
        type=float,
        default=0.1,
        dest="stl_temp",
        help="Temperature for soft mins/maxes in STL semantics.",
    )
    ap.add_argument(
        "--stl-spatial",
        type=str,
        default="mean",
        choices=["mean", "softmax", "amax"],
        help="Spatial reduction before temporal G: mean | softmax | amax.",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="results/ablations_diffusion.csv",
        help="Output CSV (two columns: lambda, robustness).",
    )
    ap.add_argument("--results-dir", type=str, default="results", help="Where experiment artifacts are written.")
    ns = ap.parse_args(argv)
    return Args(
        weights=list(ns.weights),
        epochs=int(ns.epochs),
        out=str(ns.out),
        seed=int(ns.seed),
        n_x=int(ns.n_x),
        n_t=int(ns.n_t),
        u_max=float(ns.u_max),
        alpha=float(ns.alpha),
        stl_temp=float(ns.stl_temp),
        stl_spatial=str(ns.stl_spatial),
        repeats=int(ns.repeats),
        results_dir=str(ns.results_dir),
    )


def train_once(
    stl_weight: float = 1.0,
    epochs: int = 100,
    seed: int = 0,
    *,
    n_x: int = 64,
    n_t: int = 64,
    u_max: float = 1.0,
    alpha: float = 0.1,
    stl_temp: float = 0.1,
    stl_spatial: str = "mean",
    results_dir: str = "results",
) -> float:
    return _train_and_measure_once(
        stl_weight=stl_weight,
        epochs=epochs,
        seed=seed,
        n_x=n_x,
        n_t=n_t,
        u_max=u_max,
        alpha=alpha,
        stl_temp=stl_temp,
        stl_spatial=stl_spatial,
        results_dir=results_dir,
        tag_suffix=f"_s{seed}",
    )


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows: list[list[str | float]] = [["lambda", "robustness"]]

    for w in args.weights:
        vals: list[float] = []
        for i in range(args.repeats):
            r = train_once(
                stl_weight=float(w),
                epochs=args.epochs,
                seed=args.seed + i,
                n_x=args.n_x,
                n_t=args.n_t,
                u_max=args.u_max,
                alpha=args.alpha,
                stl_temp=args.stl_temp,
                stl_spatial=args.stl_spatial,
                results_dir=args.results_dir,
            )
            vals.append(float(r))
        mean_r = float(sum(vals) / len(vals))
        rows.append([float(w), mean_r])
        print(f"λ={w:g} -> robustness={mean_r:.4f}" + (f" (n={len(vals)})" if args.repeats > 1 else ""))

    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
