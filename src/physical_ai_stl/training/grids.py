from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _as_tensor(x: float | Tensor, *, device: torch.device | str, dtype: torch.dtype) -> Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def _linspace(min_v: float, max_v: float, n: int, *, device, dtype) -> Tensor:
    if n <= 0:
        raise ValueError("n must be positive.")
    return torch.linspace(min_v, max_v, n, device=device, dtype=dtype)


def _stack_flat(*meshes: Tensor) -> Tensor:
    if not meshes:
        raise ValueError("No meshes provided.")
    flat_cols = [m.reshape(-1) for m in meshes]
    return torch.stack(flat_cols, dim=-1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def grid1d(
    n_x: int = 128,
    n_t: int = 100,
    x_min: float = 0.0,
    x_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    device: str | torch.device = "cpu",
    dtype: Optional[torch.dtype] = None,
    return_cartesian: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    if dtype is None:
        dtype = torch.get_default_dtype()
    x = _linspace(x_min, x_max, n_x, device=device, dtype=dtype)
    t = _linspace(t_min, t_max, n_t, device=device, dtype=dtype)
    X, T = torch.meshgrid(x, t, indexing="ij")
    if return_cartesian:
        XT = torch.cartesian_prod(x, t)
    else:
        XT = _stack_flat(X, T)
    return X, T, XT


def grid2d(
    n_x: int = 64,
    n_y: int = 64,
    n_t: int = 50,
    x_min: float = 0.0, x_max: float = 1.0,
    y_min: float = 0.0, y_max: float = 1.0,
    t_min: float = 0.0, t_max: float = 1.0,
    device: str | torch.device = "cpu",
    dtype: Optional[torch.dtype] = None,
    return_cartesian: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if dtype is None:
        dtype = torch.get_default_dtype()
    x = _linspace(x_min, x_max, n_x, device=device, dtype=dtype)
    y = _linspace(y_min, y_max, n_y, device=device, dtype=dtype)
    t = _linspace(t_min, t_max, n_t, device=device, dtype=dtype)
    X, Y, T = torch.meshgrid(x, y, t, indexing="ij")
    if return_cartesian:
        XYT = torch.cartesian_prod(x, y, t)
    else:
        XYT = _stack_flat(X, Y, T)
    return X, Y, T, XYT


def grid3d(
    n_x: int = 32,
    n_y: int = 32,
    n_z: int = 32,
    n_t: int = 20,
    x_min: float = 0.0, x_max: float = 1.0,
    y_min: float = 0.0, y_max: float = 1.0,
    z_min: float = 0.0, z_max: float = 1.0,
    t_min: float = 0.0, t_max: float = 1.0,
    device: str | torch.device = "cpu",
    dtype: Optional[torch.dtype] = None,
    return_cartesian: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    if dtype is None:
        dtype = torch.get_default_dtype()
    x = _linspace(x_min, x_max, n_x, device=device, dtype=dtype)
    y = _linspace(y_min, y_max, n_y, device=device, dtype=dtype)
    z = _linspace(z_min, z_max, n_z, device=device, dtype=dtype)
    t = _linspace(t_min, t_max, n_t, device=device, dtype=dtype)
    X, Y, Z, T = torch.meshgrid(x, y, z, t, indexing="ij")
    if return_cartesian:
        XYZT = torch.cartesian_prod(x, y, z, t)
    else:
        XYZT = _stack_flat(X, Y, Z, T)
    return X, Y, Z, T, XYZT


# ---------------------------------------------------------------------------
# Spacing and masks
# ---------------------------------------------------------------------------

def spacing1d(
    n_x: int, n_t: int, x_min: float, x_max: float, t_min: float, t_max: float,
    *, device: str | torch.device = "cpu", dtype: Optional[torch.dtype] = None
) -> tuple[Tensor, Tensor]:
    if dtype is None:
        dtype = torch.get_default_dtype()
    dx = _as_tensor((x_max - x_min) / (max(n_x - 1, 1)), device=device, dtype=dtype)
    dt = _as_tensor((t_max - t_min) / (max(n_t - 1, 1)), device=device, dtype=dtype)
    return dx, dt


def spacing2d(
    n_x: int, n_y: int, n_t: int,
    x_min: float, x_max: float, y_min: float, y_max: float, t_min: float, t_max: float,
    *, device: str | torch.device = "cpu", dtype: Optional[torch.dtype] = None
) -> tuple[Tensor, Tensor, Tensor]:
    if dtype is None:
        dtype = torch.get_default_dtype()
    dx = _as_tensor((x_max - x_min) / (max(n_x - 1, 1)), device=device, dtype=dtype)
    dy = _as_tensor((y_max - y_min) / (max(n_y - 1, 1)), device=device, dtype=dtype)
    dt = _as_tensor((t_max - t_min) / (max(n_t - 1, 1)), device=device, dtype=dtype)
    return dx, dy, dt


# ---------------------------------------------------------------------------
# Random/quasi-random samplers for collocation, boundary, and initial points
# ---------------------------------------------------------------------------

def _unit_samples(
    num: int, dim: int, *, method: str, device, dtype, seed: Optional[int]
) -> Tensor:
    if method not in {"uniform", "sobol"}:
        raise ValueError("method must be 'uniform' or 'sobol'")
    if method == "sobol" and dim > 0:
        engine = torch.quasirandom.SobolEngine(dim, scramble=True, seed=seed)
        u = engine.draw(num)
    else:
        u = torch.rand(num, dim)
    return u.to(device=device, dtype=dtype)


def sample_interior_1d(
    n: int,
    x_min: float = 0.0, x_max: float = 1.0,
    t_min: float = 0.0, t_max: float = 1.0,
    *,
    method: str = "sobol",
    device: str | torch.device = "cpu",
    dtype: Optional[torch.dtype] = None,
    seed: Optional[int] = None,
) -> Tensor:
    if dtype is None:
        dtype = torch.get_default_dtype()
    u = _unit_samples(n, 2, method=method, device=device, dtype=dtype, seed=seed)
    mins = torch.tensor([x_min, t_min], device=device, dtype=dtype)
    maxs = torch.tensor([x_max, t_max], device=device, dtype=dtype)
    return mins + u * (maxs - mins)


def sample_interior_2d(
    n: int,
    x_min: float = 0.0, x_max: float = 1.0,
    y_min: float = 0.0, y_max: float = 1.0,
    t_min: float = 0.0, t_max: float = 1.0,
    *,
    method: str = "sobol",
    device: str | torch.device = "cpu",
    dtype: Optional[torch.dtype] = None,
    seed: Optional[int] = None,
) -> Tensor:
    if dtype is None:
        dtype = torch.get_default_dtype()
    u = _unit_samples(n, 3, method=method, device=device, dtype=dtype, seed=seed)
    mins = torch.tensor([x_min, y_min, t_min], device=device, dtype=dtype)
    maxs = torch.tensor([x_max, y_max, t_max], device=device, dtype=dtype)
    return mins + u * (maxs - mins)


def sample_boundary_1d(
    n_total: int,
    x_min: float = 0.0, x_max: float = 1.0,
    t_min: float = 0.0, t_max: float = 1.0,
    *,
    method: str = "sobol",
    device: str | torch.device = "cpu",
    dtype: Optional[torch.dtype] = None,
    seed: Optional[int] = None,
) -> Tensor:
    if dtype is None:
        dtype = torch.get_default_dtype()
    n_left = n_total // 2
    n_right = n_total - n_left
    # Sample times with low-discrepancy sequence (1D)
    u_t_left = _unit_samples(n_left, 1, method=method, device=device, dtype=dtype, seed=seed)
    u_t_right = _unit_samples(
        n_right, 1, method=method, device=device, dtype=dtype,
        seed=None if seed is None else seed + 1
    )
    t_left = t_min + u_t_left[:, 0:1] * (t_max - t_min)
    t_right = t_min + u_t_right[:, 0:1] * (t_max - t_min)
    left = torch.cat([torch.full_like(t_left, fill_value=x_min), t_left], dim=1)
    right = torch.cat([torch.full_like(t_right, fill_value=x_max), t_right], dim=1)
    return torch.cat([left, right], dim=0)


def sample_boundary_2d(
    n_total: int,
    x_min: float = 0.0, x_max: float = 1.0,
    y_min: float = 0.0, y_max: float = 1.0,
    t_min: float = 0.0, t_max: float = 1.0,
    *,
    method: str = "sobol",
    device: str | torch.device = "cpu",
    dtype: Optional[torch.dtype] = None,
    seed: Optional[int] = None,
    split: Optional[Sequence[float]] = None,
) -> Tensor:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if split is None:
        # even split across 4 faces
        split = (0.25, 0.25, 0.25, 0.25)
    if len(split) != 4 or not torch.isclose(torch.tensor(sum(split)), torch.tensor(1.0)):
        raise ValueError("split must have 4 weights summing to 1.0")
    counts = [int(round(p * n_total)) for p in split]
    # adjust to match exactly n_total
    while sum(counts) < n_total:
        counts[counts.index(max(counts))] += 1
    while sum(counts) > n_total:
        counts[counts.index(max(counts))] -= 1

    # sample times and tangential coordinates
    def _sample_time(n: int, seed_shift: int = 0) -> Tensor:
        if n == 0:
            return torch.empty(0, 1, device=device, dtype=dtype)
        u = _unit_samples(
            n, 1, method=method, device=device, dtype=dtype,
            seed=None if seed is None else seed + seed_shift
        )
        return t_min + u[:, 0:1] * (t_max - t_min)

    def _sample_coord(n: int, lo: float, hi: float, seed_shift: int) -> Tensor:
        if n == 0:
            return torch.empty(0, 1, device=device, dtype=dtype)
        u = _unit_samples(
            n, 1, method=method, device=device, dtype=dtype,
            seed=None if seed is None else seed + seed_shift
        )
        return lo + u[:, 0:1] * (hi - lo)

    # Faces: left (x=x_min), right (x=x_max), bottom (y=y_min), top (y=y_max)
    t_left = _sample_time(counts[0], 0); y_left = _sample_coord(counts[0], y_min, y_max, 10)
    t_right = _sample_time(counts[1], 1); y_right = _sample_coord(counts[1], y_min, y_max, 11)
    t_bottom = _sample_time(counts[2], 2); x_bottom = _sample_coord(counts[2], x_min, x_max, 12)
    t_top = _sample_time(counts[3], 3); x_top = _sample_coord(counts[3], x_min, x_max, 13)

    left = torch.cat([torch.full_like(t_left, x_min), y_left, t_left], dim=1)
    right = torch.cat([torch.full_like(t_right, x_max), y_right, t_right], dim=1)
    bottom = torch.cat([x_bottom, torch.full_like(t_bottom, y_min), t_bottom], dim=1)
    top = torch.cat([x_top, torch.full_like(t_top, y_max), t_top], dim=1)
    return torch.cat([left, right, bottom, top], dim=0)


# ---------------------------------------------------------------------------
# Convenience dataclass for an axis-aligned box domain
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Box1D:
    x_min: float = 0.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    def grid(self, n_x: int, n_t: int, *, device="cpu", dtype: Optional[torch.dtype] = None,
             return_cartesian: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        return grid1d(n_x, n_t, self.x_min, self.x_max, self.t_min, self.t_max,
                      device=device, dtype=dtype, return_cartesian=return_cartesian)

    def sample_interior(self, n: int, *, method: str = "sobol", device="cpu",
                        dtype: Optional[torch.dtype] = None, seed: Optional[int] = None) -> Tensor:
        return sample_interior_1d(n, self.x_min, self.x_max, self.t_min, self.t_max,
                                  method=method, device=device, dtype=dtype, seed=seed)

    def sample_boundary(self, n_total: int, *, method: str = "sobol", device="cpu",
                        dtype: Optional[torch.dtype] = None, seed: Optional[int] = None) -> Tensor:
        return sample_boundary_1d(n_total, self.x_min, self.x_max, self.t_min, self.t_max,
                                  method=method, device=device, dtype=dtype, seed=seed)


@dataclass(frozen=True)
class Box2D:
    x_min: float = 0.0; x_max: float = 1.0
    y_min: float = 0.0; y_max: float = 1.0
    t_min: float = 0.0; t_max: float = 1.0

    def grid(self, n_x: int, n_y: int, n_t: int, *, device="cpu", dtype: Optional[torch.dtype] = None,
             return_cartesian: bool = False) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return grid2d(n_x, n_y, n_t, self.x_min, self.x_max, self.y_min, self.y_max,
                      self.t_min, self.t_max, device=device, dtype=dtype, return_cartesian=return_cartesian)

    def sample_interior(self, n: int, *, method: str = "sobol", device="cpu",
                        dtype: Optional[torch.dtype] = None, seed: Optional[int] = None) -> Tensor:
        return sample_interior_2d(n, self.x_min, self.x_max, self.y_min, self.y_max,
                                  self.t_min, self.t_max, method=method, device=device, dtype=dtype, seed=seed)

    def sample_boundary(self, n_total: int, *, method: str = "sobol", device="cpu",
                        dtype: Optional[torch.dtype] = None, seed: Optional[int] = None,
                        split: Optional[Sequence[float]] = None) -> Tensor:
        return sample_boundary_2d(n_total, self.x_min, self.x_max, self.y_min, self.y_max,
                                  self.t_min, self.t_max, method=method, device=device, dtype=dtype,
                                  seed=seed, split=split)


__all__ = [
    # original API
    "grid1d", "grid2d",
    # new generators
    "grid3d",
    # spacing
    "spacing1d", "spacing2d",
    # samplers
    "sample_interior_1d", "sample_interior_2d", "sample_boundary_1d", "sample_boundary_2d",
    # domains
    "Box1D", "Box2D",
]
