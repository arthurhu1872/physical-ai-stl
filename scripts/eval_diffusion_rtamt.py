"""Evaluate RTAMT robustness for a saved diffusion PINN checkpoint."""

from __future__ import annotations

import argparse
import sys
import torch

from physical_ai_stl.monitoring.rtamt_monitor import (
    stl_always_upper_bound,
    evaluate_series,
)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="results/diffusion_week2.pt",
                    help="Path to a saved checkpoint (torch.save dict with keys 'u', 'u_max', etc.).")
    ap.add_argument("--var", type=str, default="u",
                    help="Tensor key in the checkpoint dict to evaluate (default: 'u').")
    ap.add_argument("--dt", type=float, default=1.0,
                    help="Sampling period for the time series given to RTAMT.")
    ap.add_argument("--u-max", type=float, default=None,
                    help="Override upper bound. If not set, falls back to ckpt['u_max'] or 1.0.")
    args = ap.parse_args()

    try:
        ckpt = torch.load(args.ckpt, map_location="cpu")
    except Exception as e:
        print(f"Failed to load checkpoint '{args.ckpt}': {e}", file=sys.stderr)
        sys.exit(2)

    if args.var not in ckpt and "u" in ckpt:
        print(f"[warn] '{args.var}' not in checkpoint, using 'u' instead.")
        args.var = "u"

    try:
        u = ckpt[args.var]  # expected shape [n_x, n_t] or [*, n_t]
    except Exception as e:
        print(f"Checkpoint missing tensor '{args.var}': {e}", file=sys.stderr)
        sys.exit(3)

    # Prefer CLI override; else checkpoint; else fallback to 1.0
    if args.u_max is not None:
        u_max = float(args.u_max)
    else:
        u_max = float(ckpt.get("u_max", 1.0))

    spec = stl_always_upper_bound(var="u", u_max=u_max)

    # Evaluate mean over spatial dimension(s) to get a time series
    if hasattr(u, "mean"):
        series = u.mean(dim=0).tolist()
    else:
        raise TypeError("Expected a tensor-like with .mean(dim=0).")

    rob = evaluate_series(spec, var="u", series=series, dt=float(args.dt))
    print(f"RTAMT robustness (G mean_x {args.var} <= {u_max}): {rob:.6f}")

if __name__ == "__main__":
    main()
