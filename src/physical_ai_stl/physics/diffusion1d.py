# src/physical_ai_stl/physics/diffusion1d.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn, Tensor

from ..models.mlp import MLP  # import kept for type hints/backwards compat


# ---------------------------------------------------------------------------
# Autograd helpers
# ---------------------------------------------------------------------------

def _grad(y: Tensor, x: Tensor) -> Tensor:
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


# ---------------------------------------------------------------------------
# Core: PDE residual
# ---------------------------------------------------------------------------

def pde_residual(model: nn.Module, coords: Tensor, alpha: float | Tensor = 0.1) -> Tensor:
    if coords.ndim != 2 or coords.shape[-1] != 2:
        raise ValueError("coords must have shape (N, 2) with columns [x, t]")
    coords = coords.requires_grad_(True)

    # Forward
    u: Tensor = model(coords)  # (N,1)

    # First derivatives
    du = _grad(u, coords)          # (N,2) -> [u_x, u_t]
    u_x = du[:, 0:1]
    u_t = du[:, 1:2]

    # Second derivative w.r.t. x
    u_xx = _grad(u_x, coords)[:, 0:1]

    # Cast alpha to tensor for correct device/dtype math
    if not torch.is_tensor(alpha):
        alpha = torch.tensor(alpha, device=coords.device, dtype=u.dtype)
    return u_t - alpha * u_xx


def residual_loss(
    model: nn.Module,
    coords: Tensor,
    alpha: float | Tensor = 0.1,
    reduction: str = "mean",
) -> Tensor:
    r = pde_residual(model, coords, alpha)
    sq = r.square()
    if reduction == "mean":
        return sq.mean()
    if reduction == "sum":
        return sq.sum()
    if reduction == "none":
        return sq
    raise ValueError("reduction must be 'mean', 'sum', or 'none'")


# ---------------------------------------------------------------------------
# Boundary/Initial conditions
# ---------------------------------------------------------------------------

def _unit_samples(
    n: int,
    d: int,
    *,
    method: str,
    device: torch.device | str,
    dtype: torch.dtype | None,
    seed: int | None = None,
) -> Tensor:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if method == "sobol":
        # SobolEngine draws on CPU; move after cast.
        engine = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=seed)
        u = engine.draw(n).to(dtype=dtype, device="cpu")
        return u.to(device=device)
    if method == "uniform":
        return torch.rand(n, d, device=device, dtype=dtype)
    raise ValueError("method must be 'sobol' or 'uniform'")


def _as_tensor(x: float | Tensor, *, device: torch.device | str, dtype: torch.dtype) -> Tensor:
    return x if torch.is_tensor(x) else torch.tensor(x, device=device, dtype=dtype)


def sine_ic(
    x: Tensor,
    *,
    x_left: float = 0.0,
    x_right: float = 1.0,
    amplitude: float | Tensor = 1.0,
) -> Tensor:
    L = (x_right - x_left)
    k = torch.pi / _as_tensor(L, device=x.device, dtype=x.dtype)
    A = _as_tensor(amplitude, device=x.device, dtype=x.dtype)
    return A * torch.sin(k * (x - x_left))


def bc_ic_targets(
    x: Tensor,
    t: Tensor,
    *,
    x_left: float,
    x_right: float,
    bc_left: float | Callable[[Tensor], Tensor] = 0.0,
    bc_right: float | Callable[[Tensor], Tensor] = 0.0,
    ic: Callable[[Tensor], Tensor] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    if ic is None:
        u0 = sine_ic(x, x_left=x_left, x_right=x_right)
    else:
        u0 = ic(x)
    if callable(bc_left):
        uL = bc_left(t)
    else:
        uL = torch.full_like(t, fill_value=float(bc_left))
    if callable(bc_right):
        uR = bc_right(t)
    else:
        uR = torch.full_like(t, fill_value=float(bc_right))
    return uL, uR, u0


def boundary_loss(
    model: MLP | nn.Module,
    x_left: float = 0.0,
    x_right: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    device: torch.device | str = "cpu",
    n_boundary: int = 256,
    n_initial: int = 512,
    *,  # new keyword‑only options (backwards‑compatible defaults)
    dtype: torch.dtype | None = None,
    method: str = "sobol",
    seed: int | None = None,
    bc_left: float | Callable[[Tensor], Tensor] = 0.0,
    bc_right: float | Callable[[Tensor], Tensor] = 0.0,
    ic: Callable[[Tensor], Tensor] | None = None,
    w_boundary: float = 1.0,
    w_initial: float = 1.0,
) -> Tensor:
    if dtype is None:
        dtype = torch.get_default_dtype()

    # ---- Boundary samples (split across left/right) ----
    if n_boundary > 0:
        u = _unit_samples(n_boundary, 1, method=method, device=device, dtype=dtype, seed=seed)
        t = t_min + u * (t_max - t_min)               # (Nb,1)
        xL = torch.full_like(t, fill_value=x_left)
        xR = torch.full_like(t, fill_value=x_right)
        left = torch.cat([xL, t], dim=1)              # (Nb,2)
        right = torch.cat([xR, t], dim=1)             # (Nb,2)

        # Targets and losses
        target_L, target_R, _ = bc_ic_targets(
            x=torch.empty(0, 1, device=device, dtype=dtype),  # unused for BCs
            t=t, x_left=x_left, x_right=x_right,
            bc_left=bc_left, bc_right=bc_right, ic=None
        )
        pred_L = model(left)
        pred_R = model(right)
        loss_bc = (pred_L - target_L).square().mean() + (pred_R - target_R).square().mean()
    else:
        loss_bc = torch.zeros((), device=device, dtype=dtype)

    # ---- Initial condition samples at t=0 ----
    if n_initial > 0:
        u = _unit_samples(
            n_initial, 1, method=method, device=device, dtype=dtype,
            seed=None if seed is None else seed + 7
        )
        x = x_left + u * (x_right - x_left)           # (Ni,1)
        ic_coords = torch.cat([x, torch.zeros_like(x)], dim=1)  # t=0
        _, _, target_ic = bc_ic_targets(
            x=x, t=torch.zeros_like(x), x_left=x_left, x_right=x_right,
            bc_left=bc_left, bc_right=bc_right, ic=ic
        )
        loss_ic = (model(ic_coords) - target_ic).square().mean()
    else:
        loss_ic = torch.zeros((), device=device, dtype=dtype)

    return _as_tensor(w_boundary, device=device, dtype=dtype) * loss_bc \
         + _as_tensor(w_initial, device=device, dtype=dtype) * loss_ic


# ---------------------------------------------------------------------------
# Extras (optional but useful)
# ---------------------------------------------------------------------------

@dataclass
class Interval1D:
    x_left: float = 0.0
    x_right: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    @property
    def length(self) -> float:
        return float(self.x_right - self.x_left)


def sine_solution(
    x: Tensor,
    t: Tensor,
    alpha: float | Tensor = 0.1,
    *,
    x_left: float = 0.0,
    x_right: float = 1.0,
    amplitude: float | Tensor = 1.0,
) -> Tensor:
    L = (x_right - x_left)
    k = torch.pi / _as_tensor(L, device=x.device, dtype=x.dtype)  # spatial wavenumber
    A = _as_tensor(amplitude, device=x.device, dtype=x.dtype)
    if not torch.is_tensor(alpha):
        alpha = torch.tensor(alpha, device=x.device, dtype=x.dtype)
    return A * torch.exp(-alpha * (k ** 2) * t) * torch.sin(k * (x - x_left))


def make_dirichlet_mask_1d(x_left: float = 0.0, x_right: float = 1.0) -> Callable[[Tensor], Tensor]:
    def mask(coords: Tensor) -> Tensor:
        if coords.ndim != 2 or coords.shape[-1] != 2:
            raise ValueError("coords must have shape (N, 2) with columns [x, t]")
        x = coords[:, 0:1]
        return (x - x_left) * (x_right - x)
    return mask


class MaskedModel(nn.Module):
    def __init__(self, base: nn.Module, mask: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.base = base
        self.mask = mask

    def forward(self, coords: Tensor) -> Tensor:  # pragma: no cover - thin wrapper
        return self.mask(coords) * self.base(coords)


__all__ = [
    # core
    "pde_residual", "residual_loss", "boundary_loss",
    # extras
    "Interval1D", "sine_ic", "sine_solution", "make_dirichlet_mask_1d", "MaskedModel",
]
