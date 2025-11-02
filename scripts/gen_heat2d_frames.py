#!/usr/bin/env python3
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np

def heat2d(nx: int, ny: int, nt: int, dt: float, alpha: float, seed: int|None):
    rng = np.random.default_rng(seed)
    u = np.zeros((nx, ny), dtype=np.float32)

    # Initialize with a smooth hotspot at the center plus tiny noise
    x = np.linspace(-1, 1, nx, dtype=np.float32)
    y = np.linspace(-1, 1, ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="ij")
    sigma = 0.15
    u = np.exp(-(X**2 + Y**2) / (2*sigma**2)).astype(np.float32)
    u += 0.01 * rng.standard_normal(size=(nx, ny), dtype=np.float32)

    # 5-point Laplacian (explicit scheme); stability requires 4*alpha*dt <= 1
    assert 4.0*alpha*dt <= 1.0 + 1e-9, "Choose smaller dt or alpha for stability."

    frames = [u.copy()]
    for _ in range(nt-1):
        lap = (
            np.roll(u,  1, axis=0) + np.roll(u, -1, axis=0) +
            np.roll(u,  1, axis=1) + np.roll(u, -1, axis=1) -
            4.0*u
        )
        u = u + alpha*dt*lap
        frames.append(u.copy())
    return np.stack(frames, axis=-1)  # (nx, ny, nt)

def main() -> int:
    p = argparse.ArgumentParser(description="Generate 2D heat-equation frames (.npy) for MoonLight STREL demos.")
    p.add_argument("--nx", type=int, default=32)
    p.add_argument("--ny", type=int, default=32)
    p.add_argument("--nt", type=int, default=100)
    p.add_argument("--dt", type=float, default=0.05, help="Time step; keep 4*alpha*dt <= 1")
    p.add_argument("--alpha", type=float, default=0.5, help="Diffusivity; keep 4*alpha*dt <= 1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=Path, default=Path("assets/heat2d_scalar"))
    p.add_argument("--also-pack", action="store_true", help="Also save a single field_xy_t.npy (xy_t layout)")
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    field = heat2d(args.nx, args.ny, args.nt, args.dt, args.alpha, args.seed)  # (nx, ny, nt)

    # Per-time frames
    for t in range(args.nt):
        np.save(args.outdir / f"frame_{t:04d}.npy", field[..., t])

    # Optional packed 3D array in (nx, ny, nt) layout
    if args.also_pack:
        np.save(args.outdir / "field_xy_t.npy", field)

    print(f"Wrote {args.nt} frames to {args.outdir}/ (and packed file if requested).")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
