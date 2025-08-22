"""Train diffusion1d PINN with an STL penalty."""

import argparse
from pathlib import Path

import torch
from torch import nn, optim
from tqdm import trange

from physical_ai_stl.models.mlp import MLP
from physical_ai_stl.training.grids import grid1d
from physical_ai_stl.physics.diffusion1d import pde_residual, boundary_loss
from physical_ai_stl.monitoring.stl_soft import pred_leq, always, STLPenalty
from physical_ai_stl.utils.seed import seed_everything

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-stl", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--nx", type=int, default=128)
    p.add_argument("--nt", type=int, default=101)
    p.add_argument("--results", type=str, default="results")
    p.add_argument("--tag", type=str, default="week2")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(0)

    # Grid + model
    X, T, XT = grid1d(n_x=args.nx, n_t=args.nt, device=device)
    model = MLP(in_dim=2, out_dim=1, hidden=(64, 64, 64), activation=nn.Tanh()).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    # Physics + STL
    alpha = float(args.alpha)
    penalty = STLPenalty(weight=float(args.lambda_stl), margin=0.0)
    u_max = 1.0
    batch = 4096

    for _ in trange(args.epochs, desc="train_diffusion_stl"):
        # sample interior points
        idx = torch.randint(0, XT.shape[0], (batch,), device=device)
        coords = XT[idx]

        res = pde_residual(model, coords, alpha=alpha)
        loss_pde = res.square().mean()

        # BC/IC
        loss_bc = boundary_loss(model, device=device)

        # Temporal STL: G (mean_x u <= u_max)
        with torch.no_grad():
            inp = torch.stack([X.reshape(-1), T.reshape(-1)], dim=-1)
        u = model(inp).reshape(args.nx, args.nt)   # u(x,t)
        u_mean = u.mean(dim=0)                     # time-series (mean over x)
        margins = pred_leq(u_mean, u_max)          # margin = c - u(t)
        rob = always(margins, temp=0.1, time_dim=0)
        loss_stl = penalty(rob)

        loss = loss_pde + loss_bc + loss_stl
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Save artifact for evaluation script
    out_dir = Path(args.results)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"diffusion_{args.tag}.pt"
    torch.save({"u": u.cpu(), "X": X.cpu(), "T": T.cpu(), "u_max": float(u_max)}, out_path)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
