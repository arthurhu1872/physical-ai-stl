"""PDE residual and boundary conditions for the 2D heat equation.

This module implements the residual of the 2D heat equation and
Dirichlet/initial conditions.  It is used in the week‑2 example to
train a PINN that solves a heat diffusion problem in two spatial
dimensions.  The network approximates ``u(x,y,t)`` and the residual
enforces ``u_t = alpha * (u_xx + u_yy)``.
"""

from __future__ import annotations

import torch
from typing import Tuple

from ..models.mlp import MLP


def heat2d_residual(
    model: MLP,
    X: torch.Tensor,
    Y: torch.Tensor,
    T: torch.Tensor,
    alpha: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute residual and field for the 2D heat equation.

    Parameters
    ----------
    model : MLP
        Network mapping ``(x, y, t)`` to ``u``.
    X, Y, T : torch.Tensor
        Spatial and temporal grids of shape ``(n_x, n_y, n_t)``.
    alpha : float
        Diffusion coefficient.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A pair ``(residual, u_field)`` where ``residual`` has shape
        ``(n_x * n_y * n_t, 1)`` and ``u_field`` has shape
        ``(n_x, n_y, n_t)``.
    """
    XYT = torch.stack([
        X.reshape(-1),
        Y.reshape(-1),
        T.reshape(-1),
    ], dim=-1).requires_grad_(True)
    u = model(XYT)
    du = torch.autograd.grad(u, XYT, torch.ones_like(u), create_graph=True)[0]
    u_x, u_y, u_t = du[:, 0:1], du[:, 1:2], du[:, 2:3]
    d2u_x = torch.autograd.grad(u_x, XYT, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    d2u_y = torch.autograd.grad(u_y, XYT, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    res = u_t - alpha * (d2u_x + d2u_y)
    u_field = u.reshape_as(X)
    return res, u_field


def boundary_loss_2d(
    model: MLP,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    device: torch.device | str = "cpu",
    n_boundary: int = 256,
    n_initial: int = 512,
) -> torch.Tensor:
    """Compute loss enforcing Dirichlet boundaries and Gaussian initial condition.

    The 2D heat equation example uses homogeneous Dirichlet boundaries on
    all four sides and a Gaussian bump as the initial temperature
    distribution.  This function samples random boundary and initial
    points and returns a mean‑squared penalty.

    Parameters
    ----------
    model : MLP
        Network approximating ``u(x,y,t)``.
    x_min, x_max, y_min, y_max : float
        Spatial domain bounds.
    t_min, t_max : float
        Temporal domain bounds.
    device : torch.device or str
        Device on which to allocate samples.
    n_boundary : int
        Number of random boundary samples.
    n_initial : int
        Number of random initial samples.

    Returns
    -------
    torch.Tensor
        Scalar tensor representing the boundary and initial losses.
    """
    # Boundary conditions: u = 0 on all edges
    t = torch.rand(n_boundary, 1, device=device) * (t_max - t_min) + t_min
    xs = torch.rand(n_boundary, 1, device=device) * (x_max - x_min) + x_min
    ys = torch.rand(n_boundary, 1, device=device) * (y_max - y_min) + y_min
    # Choose random sides: 0=x_min, 1=x_max, 2=y_min, 3=y_max
    sides = torch.randint(0, 4, (n_boundary, 1), device=device)
    xb = xs.clone()
    yb = ys.clone()
    xb[sides == 0] = x_min
    xb[sides == 1] = x_max
    yb[sides == 2] = y_min
    yb[sides == 3] = y_max
    bc_coords = torch.cat([xb, yb, t], dim=1)
    loss_b = model(bc_coords).square().mean()
    # Initial condition: Gaussian bump at t=0
    x0 = torch.rand(n_initial, 1, device=device) * (x_max - x_min) + x_min
    y0 = torch.rand(n_initial, 1, device=device) * (y_max - y_min) + y_min
    ic_coords = torch.cat([x0, y0, torch.zeros_like(x0)], dim=1)
    # Centre the bump at (0.5,0.5)
    target = torch.exp(-50.0 * ((x0 - 0.5).square() + (y0 - 0.5).square()))
    loss_ic = (model(ic_coords) - target).square().mean()
    return loss_b + loss_ic
