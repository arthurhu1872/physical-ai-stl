from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

try:  # optional torch dependency (but commonly available in this repo)
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:  # pragma: no cover - allow pure NumPy usage
    _HAS_TORCH = False

ArrayLike = Union["torch.Tensor", np.ndarray, Sequence[float]]


# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------

def _to_numpy(x: ArrayLike) -> np.ndarray:
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _ensure_dir(path: Union[str, Path]) -> None:
    p = Path(path)
    if p.parent != Path("."):
        p.parent.mkdir(parents=True, exist_ok=True)


def _maybe_downsample(img: np.ndarray, max_elems: int = 2_000_000) -> Tuple[np.ndarray, int, int]:
    rows, cols = img.shape[-2], img.shape[-1]
    elems = rows * cols
    if elems <= max_elems:
        return img, 1, 1
    # choose integer stride factors to get close to max_elems
    stride = int(np.ceil(np.sqrt(elems / max_elems)))
    return img[::stride, ::stride], stride, stride


def _extent_from_coords(y: np.ndarray, x: np.ndarray) -> Tuple[float, float, float, float]:
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    return (x_min, x_max, y_min, y_max)


def _reshape_to_grid(u: np.ndarray, n_y: int, n_x: int) -> np.ndarray:
    if u.ndim == 2 and u.shape == (n_y, n_x):
        return u
    if u.ndim == 2 and u.shape == (n_x, n_y):
        return u.T
    if u.ndim == 1 and u.size == n_y * n_x:
        return u.reshape(n_y, n_x)
    raise ValueError(f"u with shape {u.shape} cannot be arranged to ({n_y}, {n_x}).")


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def plot_u_xt(
    u: ArrayLike,
    x: ArrayLike,
    t: ArrayLike,
    *,
    out: Union[str, Path] = "figs/diffusion_heatmap.png",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    add_colorbar: bool = True,
    title: str = "1‑D Diffusion PINN (u)",
    max_elems: int = 2_000_000,
    interpolation: str = "nearest",
    dpi: int = 150,
) -> Path:
    u_np = _to_numpy(u)
    x_np = _to_numpy(x).reshape(-1)
    t_np = _to_numpy(t).reshape(-1)

    grid = _reshape_to_grid(u_np, n_y=x_np.size, n_x=t_np.size)
    grid = np.asarray(grid, dtype=float)

    grid = np.ma.masked_invalid(grid)  # NaNs appear hollow instead of crashing
    grid_ds, sy, sx = _maybe_downsample(grid, max_elems=max_elems)

    # Adjust extents if downsampled
    extent = _extent_from_coords(y=x_np[::sy], x=t_np[::sx])

    _ensure_dir(out)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(
        grid_ds,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
    )
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title(title)
    if add_colorbar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("u(x,t)")
    fig.tight_layout()
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return Path(out)


def plot_u_xy_frame(
    u_xy: ArrayLike,
    *,
    x: Optional[ArrayLike] = None,
    y: Optional[ArrayLike] = None,
    out: Union[str, Path] = "figs/heat2d_t0.png",
    add_colorbar: bool = True,
    title: str = "2‑D Heat (single frame)",
    dpi: int = 150,
    interpolation: str = "nearest",
) -> Path:
    u_np = _to_numpy(u_xy)
    # If x/y are given we prefer to use their sizes to infer orientation; otherwise
    # assume square-ish and keep as‑is.
    if x is not None and y is not None:
        x_np = _to_numpy(x).reshape(-1)
        y_np = _to_numpy(y).reshape(-1)
        grid = _reshape_to_grid(u_np, n_y=y_np.size, n_x=x_np.size)
        extent = _extent_from_coords(y=y_np, x=x_np)
    else:
        grid = u_np if u_np.ndim == 2 else np.reshape(u_np, (-1, int(np.sqrt(u_np.size))))
        extent = None  # pixel extent

    grid = np.ma.masked_invalid(np.asarray(grid, dtype=float))

    _ensure_dir(out)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        extent=extent,  # type: ignore[arg-type]
        interpolation=interpolation,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    if add_colorbar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("u(x,y)")
    fig.tight_layout()
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return Path(out)


def plot_time_slices(
    u: ArrayLike,
    x: ArrayLike,
    t: ArrayLike,
    *,
    times: Optional[Iterable[float]] = None,
    num_slices: int = 4,
    out: Union[str, Path] = "figs/diffusion_slices.png",
    u_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
    title: str = "u(x,t) at selected times",
    dpi: int = 150,
) -> Path:
    u_np = _to_numpy(u)
    x_np = _to_numpy(x).reshape(-1)
    t_np = _to_numpy(t).reshape(-1)
    U = _reshape_to_grid(u_np, n_y=x_np.size, n_x=t_np.size)  # shape (n_x, n_t)

    if times is None:
        idx = np.linspace(0, t_np.size - 1, num=num_slices, dtype=int)
    else:
        times_arr = np.asarray(list(times), dtype=float)
        # nearest indices
        idx = np.clip(np.searchsorted(t_np, times_arr), 0, t_np.size - 1)

    _ensure_dir(out)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for k in idx:
        ax.plot(x_np, U[:, k], label=f"t={t_np[k]:.3g}")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title(title)
    if u_bounds is not None:
        umin, umax = u_bounds
        if umin is not None:
            ax.axhline(float(umin), linestyle="--", linewidth=1.0, alpha=0.6, label="u_min")
        if umax is not None:
            ax.axhline(float(umax), linestyle="--", linewidth=1.0, alpha=0.6, label="u_max")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", framealpha=0.8, fontsize="small")
    fig.tight_layout()
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return Path(out)


def plot_spatial_mean_over_time(
    u: ArrayLike,
    t: ArrayLike,
    *,
    mean_dims: Optional[Tuple[int, ...]] = None,
    out: Union[str, Path] = "figs/mean_over_time.png",
    u_max: Optional[float] = None,
    var_name: str = "u",
    title: Optional[str] = None,
    dpi: int = 150,
) -> Path:
    u_np = _to_numpy(u)
    t_np = _to_numpy(t).reshape(-1)
    if mean_dims is None:
        # average all but final axis (assumed time)
        mean_axes = tuple(range(u_np.ndim - 1))
    else:
        mean_axes = tuple(mean_dims)
    series = u_np.mean(axis=mean_axes)
    if series.ndim != 1 or series.size != t_np.size:
        series = series.reshape(-1)
        if series.size != t_np.size:
            raise ValueError("Cannot align mean series with provided time vector.")

    _ensure_dir(out)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t_np, series)
    ax.set_xlabel("t")
    ax.set_ylabel(f"mean_x {var_name}")
    if title is None:
        title = f"Temporal evolution of mean_x {var_name}"
    ax.set_title(title)
    if u_max is not None:
        ax.axhline(float(u_max), linestyle="--", linewidth=1.0, alpha=0.6, label=f"{var_name}_max")
        ax.legend(loc="best", framealpha=0.8, fontsize="small")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return Path(out)


# Backwards‑compatible aliases (if other scripts import the old names)
def plot_u_1d(u: ArrayLike, X: ArrayLike, T: ArrayLike, out: Union[str, Path] = "figs/diffusion_heatmap.png") -> Path:

    return plot_u_xt(u=u, x=X, t=T, out=out)


def plot_u_2d_frame(u_frame: ArrayLike, out: Union[str, Path] = "figs/heat2d_t0.png") -> Path:
    return plot_u_xy_frame(u_xy=u_frame, out=out)
