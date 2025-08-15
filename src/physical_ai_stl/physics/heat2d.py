"""PDE residual + BC/IC for the 2D heat equation (PINN)."""
from __future__ import annotations

import torch

from ..models.mlp import MLP


def residual_heat2d(model: MLP, coords: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """Residual u_t - alpha * (u_xx + u_yy) for coords (x,y,t)."""
    coords = coords.requires_grad_(True)
    u = model(coords)
    du = torch.autograd.grad(u, coords, torch.ones_like(u), create_graph=True)[0]
    u_x = du[:, 0:1]
    u_y = du[:, 1:2]
    u_t = du[:, 2:3]
    u_xx = torch.autograd.grad(u_x, coords, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, coords, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    return u_t - alpha * (u_xx + u_yy)


def bc_ic_heat2d(
    model: MLP,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    device: torch.device | str = "cpu",
    n_boundary: int = 512,
    n_initial: int = 512,
) -> torch.Tensor:
    """Dirichlet boundaries (u=0 on all sides) + Gaussian bump IC at t=0."""
    t = torch.rand(n_boundary, 1, device=device) * (t_max - t_min) + t_min
    sides = torch.randint(0, 4, (n_boundary, 1), device=device)
    xb = torch.rand(n_boundary, 1, device=device) * (x_max - x_min) + x_min
    yb = torch.rand(n_boundary, 1, device=device) * (y_max - y_min) + y_min
    xb[sides == 0] = x_min
    xb[sides == 1] = x_max
    yb[sides == 2] = y_min
    yb[sides == 3] = y_max
    bc_coords = torch.cat([xb, yb, t], dim=1)
    loss_b = model(bc_coords).square().mean()

    x0 = torch.rand(n_initial, 1, device=device) * (x_max - x_min) + x_min
    y0 = torch.rand(n_initial, 1, device=device) * (y_max - y_min) + y_min
    ic_coords = torch.cat([x0, y0, torch.zeros_like(x0)], dim=1)
    target = torch.exp(-50.0 * ((x0 - 0.5).square() + (y0 - 0.5).square()))
    loss_ic = (model(ic_coords) - target).square().mean()
    return loss_b + loss_ic


__all__ = ["residual_heat2d", "bc_ic_heat2d"]
