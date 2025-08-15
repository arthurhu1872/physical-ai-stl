"""Run ablation study over STL penalty weights for diffusion1d PINN."""

from __future__ import annotations

import argparse
import csv
import os

import torch
from torch import nn, optim
from tqdm import trange

from physical_ai_stl.models.mlp import MLP
from physical_ai_stl.training.grids import grid1d
from physical_ai_stl.physics.diffusion1d import pde_residual, boundary_loss
from physical_ai_stl.monitoring.stl_soft import pred_leq, always, STLPenalty

def _seed(seed: int = 0) -> None:
    import numpy as np
    import random

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def train_once(stl_weight: float, epochs: int = 100, seed: int = 0) -> float:
    _seed(seed)
    device = "cpu"
    n_x, n_t = 128, 64
    X, T, XT = grid1d(n_x=n_x, n_t=n_t, device=device)
    model = MLP(in_dim=2, out_dim=1, hidden=(64, 64, 64), activation=nn.Tanh()).to(device)
    opt = optim.Adam(model.parameters(), lr=2e-3)
    alpha, u_max, penalty, batch = 0.1, 1.0, STLPenalty(weight=stl_weight, margin=0.0), 4096
    for _ in trange(epochs, desc=f"λ={stl_weight}"):
        idx = torch.randint(0, XT.shape[0], (batch,), device=device)
        coords = XT[idx]
        res = pde_residual(model, coords, alpha=alpha)
        loss_pde = res.square().mean()
        loss_bcic = boundary_loss(model, device=device)
        with torch.no_grad():
            inp = torch.stack([X.reshape(-1), T.reshape(-1)], dim=-1)
        u = model(inp).reshape(n_x, n_t)
        u_mean = u.mean(dim=0)
        margins = pred_leq(u_mean, u_max)
        rob = always(margins, temp=0.1, time_dim=0)
        loss = loss_pde + loss_bcic + penalty(rob)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        inp = torch.stack([X.reshape(-1), T.reshape(-1)], dim=-1)
        u = model(inp).reshape(n_x, n_t)
        u_mean = u.mean(dim=0)
    return float(always(pred_leq(u_mean, u_max), temp=0.1, time_dim=0))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", type=float, default=[0.0, 0.1, 0.5, 1.0])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--out", type=str, default="results/ablations_diffusion.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = [["lambda", "robustness"]]
    for w in args.weights:
        r = train_once(stl_weight=w, epochs=args.epochs, seed=args.seed)
        rows.append([w, r])
        print(f"λ={w} -> robustness={r:.4f}")
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
