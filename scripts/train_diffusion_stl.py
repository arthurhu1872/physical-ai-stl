import os
from pathlib import Path

import torch
from torch import nn, optim
from tqdm import trange

from physical_ai_stl.models.mlp import MLP
from physical_ai_stl.training.grids import grid1d
from physical_ai_stl.physics.diffusion1d import pde_residual, boundary_loss
from physical_ai_stl.monitoring.stl_soft import pred_leq, always, STLPenalty


def seed_everything(seed: int = 0) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    seed_everything(0)
    os.makedirs("results", exist_ok=True)

    device = "cpu"
    n_x, n_t = 128, 64
    X, T, XT = grid1d(n_x=n_x, n_t=n_t, device=device)

    model = MLP(in_dim=2, out_dim=1, hidden=(64, 64, 64), activation=nn.Tanh()).to(device)
    opt = optim.Adam(model.parameters(), lr=2e-3)

    alpha = 0.1
    u_max = 1.0
    stl_weight = 0.5
    penalty = STLPenalty(weight=stl_weight, margin=0.0)

    epochs = 200
    batch = 4096

    for _ in trange(epochs, desc="train-1d-diffusion-stl"):
        # sample interior points
        idx = torch.randint(0, XT.shape[0], (batch,), device=device)
        coords = XT[idx]
        res = pde_residual(model, coords, alpha=alpha)
        loss_pde = res.square().mean()

        # BC/IC
        loss_bcic = boundary_loss(model, device=device)

        # Temporal STL: G (mean_x u <= u_max)
        with torch.no_grad():
            inp = torch.stack([X.reshape(-1), T.reshape(-1)], dim=-1)
        u = model(inp).reshape(n_x, n_t)          # u(x,t)
        u_mean = u.mean(dim=0)                    # series in t
        margins = pred_leq(u_mean, u_max)         # c - u(t)
        rob = always(margins, temp=0.1, time_dim=0)
        loss_stl = penalty(rob)

        loss = loss_pde + loss_bcic + loss_stl
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Save artifact for the audit script
    with torch.no_grad():
        inp = torch.stack([X.reshape(-1), T.reshape(-1)], dim=-1)
        u = model(inp).reshape(n_x, n_t)
    torch.save(
        {"u": u.cpu(), "X": X.cpu(), "T": T.cpu(), "u_max": float(u_max), "alpha": float(alpha)},
        "results/diffusion_week1.pt",
    )
    print("Saved: results/diffusion_week1.pt")


if __name__ == "__main__":
    main()
