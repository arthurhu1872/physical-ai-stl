"""PDE residual and boundary conditions for the 1D diffusion equation.

This module implements the residual of the heat equation in one spatial
dimension and simple boundary and initial conditions.  These functions
are used in the week‑1 example to train a physics-informed neural network
(PINN) that solves the diffusion equation while obeying Dirichlet
boundary conditions and an initial sine profile.
"""

from __future__ import annotations

import torch
from typing import Tuple

from ..models.mlp import MLP


def pde_residual(
    model: MLP,
    X: torch.Tensor,
    T: torch.Tensor,
    alpha: float = 0.1,
) -> torch.Tensor:
    """Compute the PDE residual for the diffusion equation.

    The 1D diffusion (heat) equation reads ``u_t = alpha * u_xx``.  Given
    a network ``model`` that maps ``(x,t)`` to ``u``, this function
    computes the residual ``u_t - alpha * u_xx`` over a flattened grid.

    Parameters
    ----------
    model : MLP
        Neural network approximating the solution ``u(x,t)``.
    X, T : torch.Tensor
        Spatial and temporal grids of shape ``(n_x, n_t)`` as returned
        from :func:`~physical_ai_stl.training.grids.grid1d`.
    alpha : float
        Diffusion coefficient.

    Returns
    -------
    torch.Tensor
        Residual vector of shape ``(n_x * n_t, 1)``.
    """
    # Flatten coordinates and enable gradients
    xt = torch.stack([X.reshape(-1), T.reshape(-1)], dim=-1).requires_grad_(True)
    u = model(xt)  # forward pass
    # First-order derivatives
    du = torch.autograd.grad(u, xt, torch.ones_like(u), create_graph=True)[0]
    u_x = du[:, 0:1]
    u_t = du[:, 1:2]
    # Second-order spatial derivative
    d2u_x = torch.autograd.grad(u_x, xt, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    return u_t - alpha * d2u_x


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
    """Compute loss enforcing Dirichlet boundary and sine initial conditions.

    The diffusion PINN uses homogeneous Dirichlet boundaries ``u(0,t)=0``
    and ``u(1,t)=0`` as well as a sine initial condition
    ``u(x,0) = sin(pi x)``.  This function randomly samples boundary and
    initial points to form mean‑squared errors penalising deviations
    from these conditions.

    Parameters
    ----------
    model : MLP
        Neural network approximating the solution ``u(x,t)``.
    x_left, x_right : float
        Spatial domain bounds.
    t_min, t_max : float
        Temporal domain bounds.
    device : torch.device or str
        Device on which to allocate the samples.
    n_boundary : int
        Number of random time samples for each boundary.
    n_initial : int
        Number of random spatial samples for the initial condition.

    Returns
    -------
    torch.Tensor
        Scalar tensor representing the boundary and initial condition loss.
    """
    # Random time samples for boundary conditions
    t = torch.rand(n_boundary, 1, device=device) * (t_max - t_min) + t_min
    left = torch.cat([torch.full_like(t, x_left), t], dim=1)
    right = torch.cat([torch.full_like(t, x_right), t], dim=1)
    # Enforce u=0 at both boundaries
    loss_b = (model(left).square().mean() + model(right).square().mean())
    # Random spatial samples for initial condition
    x = torch.rand(n_initial, 1, device=device) * (x_right - x_left) + x_left
    ic = torch.cat([x, torch.zeros_like(x)], dim=1)
    target = torch.sin(torch.pi * x)
    loss_ic = (model(ic) - target).square().mean()
    return loss_b + loss_ic
