"""Train heat2d PINN and optionally audit with a MoonLight spec."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from tqdm import trange

from physical_ai_stl.models.mlp import MLP
from physical_ai_stl.training.grids import grid2d
from physical_ai_stl.physics.heat2d import residual_heat2d, bc_ic_heat2d
from physical_ai_stl.monitoring.moonlight_helper import (
    load_script_from_file,
    get_monitor,
    build_grid_graph,
    field_to_signal,
)
from physical_ai_stl.utils.seed import seed_everything

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--ny", type=int, default=64)
    ap.add_argument("--nt", type=int, default=16)
    ap.add_argument("--results", type=str, default="results")
    ap.add_argument("--tag", type=str, default="week3")
    ap.add_argument("--mls", type=str, default="scripts/specs/contain_hotspot.mls")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(0)

    X, Y, T, XYT = grid2d(n_x=args.nx, n_y=args.ny, n_t=args.nt, device=device)
    model = MLP(in_dim=3, out_dim=1, hidden=(64, 64, 64), activation=nn.Tanh()).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    for _ in trange(args.epochs, desc="train_heat2d_strel"):
        idx = torch.randint(0, XYT.shape[0], (4096,), device=device)
        coords = XYT[idx]
        res = residual_heat2d(model, coords, alpha=float(args.alpha))
        loss_pde = res.square().mean()
        loss_bcic = bc_ic_heat2d(model, device=device)
        loss = loss_pde + loss_bcic
        opt.zero_grad()
        loss.backward()
        opt.step()

    out_dir = Path(args.results)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    with torch.no_grad():
        for k in [0, args.nt // 2, args.nt - 1]:
            inp = torch.stack([X[:, :, k].reshape(-1), Y[:, :, k].reshape(-1), T[:, :, k].reshape(-1)], dim=-1)
            u = model(inp).reshape(args.nx, args.ny)
            p = out_dir / f"heat2d_{args.tag}_t{k}.npy"
            np.save(p, u.cpu().numpy())
            frames.append(str(p))
    print(f"Saved frames: {frames}")

    # Optional MoonLight audit (requires MoonLight to be installed)
    try:
        script = load_script_from_file(args.mls)
        mon = get_monitor(script, "contain")
        graph = build_grid_graph(args.nx, args.ny)
        # Build a toy spatio-temporal signal from our saved frames
        arrs = [np.load(p) for p in frames]
        u = np.stack(arrs, axis=-1)  # shape (nx, ny, len(frames))
        sig = field_to_signal(u, threshold=float(u.mean() + 0.5 * u.std()))
        out = mon.monitor_graph_time_series(graph, sig)
        print("[MoonLight] monitor output (first 3 entries):", out[:3])
    except Exception as e:
        print(f"[MoonLight] Skipping audit (install moonlight to enable): {e}")

if __name__ == "__main__":
    main()
