"""Grid generation utilities for PDE examples."""
from __future__ import annotations

from typing import Tuple

import torch
def grid1d(
    n_x: int = 128,
    n_t: int = 100,
    x_min: float = 0.0,
    x_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.linspace(x_min, x_max, n_x, device=device)
    t = torch.linspace(t_min, t_max, n_t, device=device)
    X, T = torch.meshgrid(x, t, indexing="ij")
    XT = torch.stack([X.reshape(-1), T.reshape(-1)], dim=-1)
    return X, T, XT

def grid2d(
    n_x: int = 64,
    n_y: int = 64,
    n_t: int = 50,
    x_min: float = 0.0, x_max: float = 1.0,
    y_min: float = 0.0, y_max: float = 1.0,
    t_min: float = 0.0, t_max: float = 1.0,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.linspace(x_min, x_max, n_x, device=device)
    y = torch.linspace(y_min, y_max, n_y, device=device)
    t = torch.linspace(t_min, t_max, n_t, device=device)
    X, Y, T = torch.meshgrid(x, y, t, indexing="ij")
    XYT = torch.stack([X.reshape(-1), Y.reshape(-1), T.reshape(-1)], dim=-1)
    return X, Y, T, XYT

__all__ = ["grid1d", "grid2d"]