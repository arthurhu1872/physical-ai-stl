"""Evaluate RTAMT robustness for a saved diffusion PINN checkpoint."""

from __future__ import annotations
from pathlib import Path
from physical_ai_stl.monitoring.rtamt_monitor import (
from typing import Any
import argparse
    evaluate_series,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="results/diffusion_week2.pt", help="Checkpoint path")
    ap.add_argument("--var", type=str, default="u", help="Variable name")
    ap.add_argument("--dt", type=float, default=0.1, help="Time step")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return

    try:
        import torch  # local import to avoid hard dependency at import time
    except Exception as exc:  # pragma: no cover
        print(f"Torch not available: {exc}")
        return

    data: dict[str, Any] = torch.load(ckpt_path, map_location="cpu")
    tensor = data.get(args.var)
    if tensor is None:
        raise KeyError(f"Variable {args.var!r} missing in checkpoint")

    u_max = float(data.get("u_max", 1.0))
    spec = stl_always_upper_bound(var=args.var, u_max=u_max)

    # Mean over spatial dims (0 or 0/1) to form a time series
    if hasattr(tensor, "mean"):
        # try dim=0 then fallback to flatten except time
        try:
            series = tensor.mean(dim=0).tolist()
        except Exception:
            series = tensor.reshape(-1, tensor.shape[-1]).mean(dim=0).tolist()  # type: ignore[attr-defined]
    else:
        raise TypeError("Expected a tensor-like with .mean(dim=0)")

    rob = evaluate_series(spec, var=args.var, series=series, dt=float(args.dt))
    print(f"RTAMT robustness (G mean_x {args.var} <= {u_max}): {rob:.6f}")


if __name__ == "__main__":
    main()