"""PDE residual + BC/IC for the 1D diffusion equation (PINN)."""
from __future__ import annotations

import torch

from ..models.mlp import MLP
def pde_residual(model: MLP, coords: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """Residual u_t - alpha * u_xx at flattened coords (x, t)."""
    coords = coords.requires_grad_(True)
    u = model(coords)  # (N,1)
    du = torch.autograd.grad(u, coords, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = du[:, 0:1]
    u_t = du[:, 1:2]
    u_xx = torch.autograd.grad(u_x, coords, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][
        :, 0:1
    ]
    return u_t - alpha * u_xx

def boundary_loss(
    model: MLP,
    x_left: float = 0.0,
    x_right: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    device: torch.device | str = "cpu",
    n_boundary: int = 256,
    n_initial: int = 512,
) -> torch.Tensor:
    """Dirichlet BC (u=0 at x=left/right) + sinusoidal IC at t=0."""
    t = torch.rand(n_boundary, 1, device=device) * (t_max - t_min) + t_min
    left = torch.cat([torch.full_like(t, x_left), t], dim=1)
    right = torch.cat([torch.full_like(t, x_right), t], dim=1)
    loss_b = (model(left).square().mean() + model(right).square().mean())

    x = torch.rand(n_initial, 1, device=device) * (x_right - x_left) + x_left
    ic = torch.cat([x, torch.zeros_like(x)], dim=1)
    target = torch.sin(torch.pi * x)
    loss_ic = (model(ic) - target).square().mean()
    return loss_b + loss_ic

__all__ = ["pde_residual", "boundary_loss"]