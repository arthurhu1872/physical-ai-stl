"""Grid generation utilities for PDE examples.

These functions build 1D and 2D space–time grids used for discretising
partial differential equations.  The returned tensors live on a user
provided device and can be reshaped to flat coordinate arrays when
passing inputs to a neural network.
"""

from __future__ import annotations

import torch
from typing import Tuple


def grid1d(
    n_x: int = 128,
    n_t: int = 100,
    x_min: float = 0.0,
    x_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a regular 1D space–time grid.

    Parameters
    ----------
    n_x : int
        Number of spatial points.
    n_t : int
        Number of temporal points.
    x_min, x_max : float
        Domain bounds in space.
    t_min, t_max : float
        Domain bounds in time.
    device : torch.device or str
        Device on which to allocate the tensors.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple ``(X, T, XT)`` where ``X`` and ``T`` have shape
        ``(n_x, n_t)`` and ``XT`` has shape ``(n_x * n_t, 2)`` containing
        flattened coordinates suitable for passing into an MLP.
    """
    x = torch.linspace(x_min, x_max, n_x, device=device)
    t = torch.linspace(t_min, t_max, n_t, device=device)
    X, T = torch.meshgrid(x, t, indexing="ij")
    XT = torch.stack([X.reshape(-1), T.reshape(-1)], dim=-1)
    return X, T, XT


def grid2d(
    n_x: int = 64,
    n_y: int = 64,
    n_t: int = 50,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a regular 2D space–time grid.

    Parameters
    ----------
    n_x, n_y : int
        Number of spatial points in the x and y dimensions.
    n_t : int
        Number of temporal points.
    x_min, x_max, y_min, y_max : float
        Domain bounds in space.
    t_min, t_max : float
        Domain bounds in time.
    device : torch.device or str
        Device on which to allocate the tensors.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple ``(X, Y, T, XYT)`` where ``X``, ``Y`` and ``T`` have shape
        ``(n_x, n_y, n_t)`` and ``XYT`` has shape ``(n_x * n_y * n_t, 3)``.
    """
    x = torch.linspace(x_min, x_max, n_x, device=device)
    y = torch.linspace(y_min, y_max, n_y, device=device)
    t = torch.linspace(t_min, t_max, n_t, device=device)
    X, Y, T = torch.meshgrid(x, y, t, indexing="ij")
    XYT = torch.stack([
        X.reshape(-1),
        Y.reshape(-1),
        T.reshape(-1),
    ], dim=-1)
    return X, Y, T, XYT
