import os
import torch
from tqdm import trange
from pathlib import Path

from src.physical_ai_stl.models.mlp import MLP
from src.physical_ai_stl.training.grids import grid1d
from src.physical_ai_stl.physics.diffusion1d import pde_residual, boundary_loss
from src.physical_ai_stl.monitoring.stl_soft import pred_leq, always, STLPenalty


def seed_everything(seed: int = 0) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    """Train a 1D diffusion PINN with an STL safety specification."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(0)

    # Hyperparameters
    alpha = 0.1  # diffusion coefficient
    n_x, n_t = 128, 101  # spatial and temporal resolution
    epochs = 3000
    lr = 1e-3
    lambda_stl = 1.0  # weight for STL penalty
    temp = 0.1  # temperature for softmin
    u_max = 1.0  # safety threshold (u <= u_max)
    margin = 0.05  # robustness margin

    # Model and optimizer
    model = MLP(in_dim=2, out_dim=1, hidden=(128, 128, 128)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Generate grid
    X, T, _ = grid1d(n_x=n_x, n_t=n_t, device=device)

    # STL penalty function
    stl_penalty = STLPenalty(margin=margin)

    for epoch in trange(epochs, desc="Training Week1"):
        optimizer.zero_grad()

        # Physics-informed loss (PDE residual)
        res = pde_residual(None, X, T, model, alpha=alpha)
        loss_pde = (res ** 2).mean()

        # Boundary and initial condition loss
        loss_bc = boundary_loss(model, device=device)

        # STL safety specification: always(u <= u_max) over time for each spatial point
        # Compute robustness using soft semantics
        xt = torch.stack([X.reshape(-1), T.reshape(-1)], dim=-1).requires_grad_(True)
        u_pred = model(xt).reshape(n_x, n_t)
        r_t = pred_leq(u_pred, u_max)  # robustness per time step
        r_x = always(r_t, temp=temp, dim=1)  # aggregate over time via softmin
        robustness = r_x.mean()  # mean over spatial points

        loss_stl = stl_penalty(robustness)

        # Total loss
        loss = loss_pde + loss_bc + lambda_stl * loss_stl
        loss.backward()
        optimizer.step()

        # Logging every 200 epochs
        if epoch % 200 == 0:
            print(
                f"epoch {epoch:04d} | total={loss.item():.4e} | pde={loss_pde.item():.4e} | bc={loss_bc.item():.4e} | stl={loss_stl.item():.4e}"
            )

    # Save results for evaluation
    with torch.no_grad():
        xt = torch.stack([X.reshape(-1), T.reshape(-1)], dim=-1)
        u_out = model(xt).reshape(n_x, n_t).detach().cpu()

    Path("results").mkdir(exist_ok=True, parents=True)
    torch.save(
        {
            "u": u_out,
            "X": X.cpu(),
            "T": T.cpu(),
            "u_max": u_max,
            "alpha": alpha,
        },
        "results/diffusion_week1.pt",
    )
    print("Saved results to results/diffusion_week1.pt")


if __name__ == "__main__":
    main()
