from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor

from ..models.mlp import MLP

# Optional imports (kept local to avoid import cycles at module load time)
try:  # pragma: no cover - convenience only
    from ..training.grids import sample_boundary_2d
except Exception:  # pragma: no cover
    sample_boundary_2d = None  # type: ignore


# -----------------------------------------------------------------------------
# Core PDE residual
# -----------------------------------------------------------------------------

def residual_heat2d(model: MLP, coords: Tensor, alpha: float = 0.1) -> Tensor:
    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError("coords must have shape (N, 3) with columns [x, y, t]")
    coords = coords.requires_grad_(True)

    # Forward
    u: Tensor = model(coords)  # (N,1)

    # First derivatives
    du = torch.autograd.grad(
        u, coords, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_x = du[:, 0:1]
    u_y = du[:, 1:2]
    u_t = du[:, 2:3]

    # Second derivatives (diagonal Hessian entries)
    u_xx = torch.autograd.grad(
        u_x, coords, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0][:, 0:1]
    u_yy = torch.autograd.grad(
        u_y, coords, grad_outputs=torch.ones_like(u_y), create_graph=True
    )[0][:, 1:2]

    return u_t - alpha * (u_xx + u_yy)


# -----------------------------------------------------------------------------
# Boundary / initial condition helpers (soft penalties by default)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SquareDomain2D:
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    @property
    def width(self) -> float:
        return float(self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return float(self.y_max - self.y_min)


def gaussian_ic(
    x: Tensor,
    y: Tensor,
    center: tuple[float, float] = (0.5, 0.5),
    sharpness: float = 50.0,
) -> Tensor:
    cx, cy = center
    r2 = (x - cx).square() + (y - cy).square()
    return torch.exp(-sharpness * r2)


def bc_ic_heat2d(
    model: MLP,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    *,
    device: torch.device | str = "cpu",
    dtype: Optional[torch.dtype] = None,
    n_boundary: int = 512,
    n_initial: int = 512,
    ic: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    boundary_split: Optional[tuple[float, float, float, float]] = None,
) -> Tensor:
    if dtype is None:
        dtype = torch.get_default_dtype()

    dom = SquareDomain2D(x_min, x_max, y_min, y_max, t_min, t_max)

    # ------------------- Boundary: u=0 on all spatial faces -------------------
    if sample_boundary_2d is not None:
        bc_coords = sample_boundary_2d(
            n_boundary, dom.x_min, dom.x_max, dom.y_min, dom.y_max, dom.t_min, dom.t_max,
            method="sobol", device=device, dtype=dtype, split=boundary_split
        )
    else:  # fallback: uniform RNG
        t = torch.rand((n_boundary, 1), device=device, dtype=dtype) * (dom.t_max - dom.t_min) + dom.t_min
        sides = torch.randint(0, 4, (n_boundary, 1), device=device)
        xb = torch.rand((n_boundary, 1), device=device, dtype=dtype) * (dom.x_max - dom.x_min) + dom.x_min
        yb = torch.rand((n_boundary, 1), device=device, dtype=dtype) * (dom.y_max - dom.y_min) + dom.y_min
        xb[sides == 0] = dom.x_min
        xb[sides == 1] = dom.x_max
        yb[sides == 2] = dom.y_min
        yb[sides == 3] = dom.y_max
        bc_coords = torch.cat([xb, yb, t], dim=1)

    loss_b = model(bc_coords).square().mean()

    # ------------------- Initial condition at t=0 -------------------
    x0 = torch.rand((n_initial, 1), device=device, dtype=dtype) * dom.width + dom.x_min
    y0 = torch.rand((n_initial, 1), device=device, dtype=dtype) * dom.height + dom.y_min
    ic_coords = torch.cat([x0, y0, torch.zeros_like(x0)], dim=1)

    if ic is None:
        cx = 0.5 * (dom.x_min + dom.x_max)
        cy = 0.5 * (dom.y_min + dom.y_max)
        target = gaussian_ic(x0, y0, center=(cx, cy), sharpness=50.0)
    else:
        target = ic(x0, y0)

    loss_ic = (model(ic_coords) - target).square().mean()
    return loss_b + loss_ic


# -----------------------------------------------------------------------------
# Optional: exact Dirichlet satisfaction via an output‑space mask
# -----------------------------------------------------------------------------

def make_dirichlet_mask(x_min: float, x_max: float, y_min: float, y_max: float, *, pow: int = 1) -> Callable[[Tensor], Tensor]:
    def mask(coords: Tensor) -> Tensor:
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        mx = (x - x_min) * (x_max - x)
        my = (y - y_min) * (y_max - y)
        m = (mx * my).clamp_min(0.0)
        if pow != 1:
            m = m.pow(pow)
        return m
    return mask


class MaskedModel(torch.nn.Module):
    def __init__(self, base: torch.nn.Module, mask: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.base = base
        self.mask = mask

    def forward(self, coords: Tensor) -> Tensor:  # pragma: no cover - thin wrapper
        return self.mask(coords) * self.base(coords)


__all__ = [
    "residual_heat2d",
    "bc_ic_heat2d",
    # extras
    "SquareDomain2D",
    "gaussian_ic",
    "make_dirichlet_mask",
    "MaskedModel",
]
