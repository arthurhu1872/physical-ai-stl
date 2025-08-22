from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from ..models.mlp import MLP
from ..physics.heat2d import residual_heat2d, bc_ic_heat2d
from ..training.grids import grid2d
from ..utils.logger import CSVLogger
from ..utils.seed import seed_everything

# STL soft semantics (optional, used if cfg.stl_use is True)
try:
    from ..monitoring.stl_soft import STLPenalty, pred_leq, always, softmin
    _HAS_STL = True
except Exception:  # pragma: no cover - optional dep inside package
    _HAS_STL = False


__all__ = ["Heat2DConfig", "run_heat2d"]


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass
class Heat2DConfig:
    # --- model ---
    hidden: tuple[int, ...] = (64, 64, 64)
    activation: str = "tanh"           # 'tanh' | 'relu' | 'gelu' | 'sine' (SIREN) | ...

    # --- grid / domain ---
    n_x: int = 64
    n_y: int = 64
    n_t: int = 16
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    # --- optimization ---
    lr: float = 2e-3
    epochs: int = 200
    batch: int = 4096
    weight_decay: float = 0.0
    amsgrad: bool = False
    scheduler: str = "none"            # 'none' | 'cosine' | 'step'
    step_size: int = 100               # for 'step'
    gamma: float = 0.5                 # for 'step'
    grad_clip: Optional[float] = None  # e.g., 1.0 to clip global grad norm
    compile: bool = False              # try torch.compile if available
    amp: bool = True                   # use autocast+GradScaler on CUDA
    device: str = "auto"               # 'auto' | 'cpu' | 'cuda' | 'mps'

    # --- physics ---
    alpha: float = 0.1                 # diffusivity
    bcic_weight: float = 1.0           # weight for BC/IC penalty

    # --- residual‑aware resampling (RAR) ---
    rar_pool: int = 0                  # if > 0, evaluate residual on this many pool points
    rar_hard_frac: float = 0.5         # fraction of batch from top‑|r| points
    rar_every: int = 10                # how often (epochs) to use RAR; 0 disables

    # --- STL penalty (soft, differentiable) ---
    stl_use: bool = False
    stl_weight: float = 0.0
    stl_u_min: Optional[float] = None
    stl_u_max: Optional[float] = None
    stl_margin: float = 0.0
    stl_beta: float = 10.0
    stl_temp: float = 0.1
    stl_nx: int = 32
    stl_ny: int = 32
    stl_nt: int = 16
    stl_every: int = 10
    stl_x_min: Optional[float] = None  # if None, uses [x_min,x_max]
    stl_x_max: Optional[float] = None
    stl_y_min: Optional[float] = None
    stl_y_max: Optional[float] = None
    stl_t_min: Optional[float] = None  # if None, uses [t_min,t_max]
    stl_t_max: Optional[float] = None

    # --- output / logging ---
    results_dir: str = "results"
    tag: str = "run"
    save_ckpt: bool = True
    save_frames: bool = True
    frames_idx: Iterable[int] = (0, 8, 15)
    save_figs: bool = True
    print_every: int = 10
    seed: int = 0


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def _select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)

def _maybe_compile(model: nn.Module, enabled: bool) -> nn.Module:
    if not enabled:
        return model
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:   # PyTorch < 2.0
        return model
    try:  # pragma: no cover - optional perf feature
        return compile_fn(model, mode="default")
    except Exception:
        return model

def _ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def _gradmag_numpy(u_2d: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(u_2d)
    gy = np.zeros_like(u_2d)
    gx[1:-1, :] = 0.5 * (u_2d[2:, :] - u_2d[:-2, :])
    gy[:, 1:-1] = 0.5 * (u_2d[:, 2:] - u_2d[:, :-2])
    return np.sqrt(gx * gx + gy * gy)

def _stl_penalty(
    model: nn.Module,
    cfg: Heat2DConfig,
    device: torch.device,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not cfg.stl_use or not _HAS_STL or cfg.stl_weight <= 0.0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    # Determine region and grid
    x0 = cfg.stl_x_min if cfg.stl_x_min is not None else cfg.x_min
    x1 = cfg.stl_x_max if cfg.stl_x_max is not None else cfg.x_max
    y0 = cfg.stl_y_min if cfg.stl_y_min is not None else cfg.y_min
    y1 = cfg.stl_y_max if cfg.stl_y_max is not None else cfg.y_max
    t0 = cfg.stl_t_min if cfg.stl_t_min is not None else cfg.t_min
    t1 = cfg.stl_t_max if cfg.stl_t_max is not None else cfg.t_max

    X, Y, T, XYT = grid2d(
        n_x=cfg.stl_nx, n_y=cfg.stl_ny, n_t=cfg.stl_nt,
        x_min=x0, x_max=x1, y_min=y0, y_max=y1, t_min=t0, t_max=t1,
        device=device, dtype=dtype
    )
    u = model(XYT).reshape(cfg.stl_nx * cfg.stl_ny, cfg.stl_nt)  # (Nxy, Nt)

    # Build predicates
    preds = []
    if cfg.stl_u_max is not None:
        preds.append(always(pred_leq(u, cfg.stl_u_max), temp=cfg.stl_temp, time_dim=-1))  # shape (Nxy,)
    if cfg.stl_u_min is not None:
        preds.append(always(pred_leq(-u, -cfg.stl_u_min), temp=cfg.stl_temp, time_dim=-1))  # u ≥ u_min

    if not preds:
        return torch.tensor(0.0, device=device, dtype=dtype)

    # Combine constraints with soft-min across predicates then mean over space
    r_space = preds[0] if len(preds) == 1 else softmin(torch.stack(preds, dim=0), temp=cfg.stl_temp, dim=0)
    r_scalar = r_space.mean()

    pen = STLPenalty(weight=cfg.stl_weight, margin=cfg.stl_margin, kind="softplus", beta=cfg.stl_beta)
    return pen(r_scalar)

# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def run_heat2d(cfg_dict: dict[str, Any]) -> list[str]:
    # --- parse config dict robustly (with sensible defaults) ---------------
    cfg = Heat2DConfig(
        hidden=tuple(cfg_dict.get("model", {}).get("hidden", (64, 64, 64))),
        activation=str(cfg_dict.get("model", {}).get("activation", "tanh")),

        n_x=int(cfg_dict.get("grid", {}).get("n_x", 64)),
        n_y=int(cfg_dict.get("grid", {}).get("n_y", 64)),
        n_t=int(cfg_dict.get("grid", {}).get("n_t", 16)),
        x_min=float(cfg_dict.get("grid", {}).get("x_min", 0.0)),
        x_max=float(cfg_dict.get("grid", {}).get("x_max", 1.0)),
        y_min=float(cfg_dict.get("grid", {}).get("y_min", 0.0)),
        y_max=float(cfg_dict.get("grid", {}).get("y_max", 1.0)),
        t_min=float(cfg_dict.get("grid", {}).get("t_min", 0.0)),
        t_max=float(cfg_dict.get("grid", {}).get("t_max", 1.0)),

        lr=float(cfg_dict.get("optim", {}).get("lr", 2e-3)),
        epochs=int(cfg_dict.get("optim", {}).get("epochs", 200)),
        batch=int(cfg_dict.get("optim", {}).get("batch", 4096)),
        weight_decay=float(cfg_dict.get("optim", {}).get("weight_decay", 0.0)),
        amsgrad=bool(cfg_dict.get("optim", {}).get("amsgrad", False)),
        scheduler=str(cfg_dict.get("optim", {}).get("scheduler", "none")),
        step_size=int(cfg_dict.get("optim", {}).get("step_size", 100)),
        gamma=float(cfg_dict.get("optim", {}).get("gamma", 0.5)),
        grad_clip=cfg_dict.get("optim", {}).get("grad_clip", None),
        compile=bool(cfg_dict.get("optim", {}).get("compile", False)),
        amp=bool(cfg_dict.get("optim", {}).get("amp", True)),
        device=str(cfg_dict.get("optim", {}).get("device", "auto")),

        alpha=float(cfg_dict.get("physics", {}).get("alpha", 0.1)),
        bcic_weight=float(cfg_dict.get("physics", {}).get("bcic_weight", 1.0)),

        rar_pool=int(cfg_dict.get("rar", {}).get("pool", 0)),
        rar_hard_frac=float(cfg_dict.get("rar", {}).get("hard_frac", 0.5)),
        rar_every=int(cfg_dict.get("rar", {}).get("every", 10)),

        stl_use=bool(cfg_dict.get("stl", {}).get("use", False)),
        stl_weight=float(cfg_dict.get("stl", {}).get("weight", 0.0)),
        stl_u_min=cfg_dict.get("stl", {}).get("u_min", None),
        stl_u_max=cfg_dict.get("stl", {}).get("u_max", None),
        stl_margin=float(cfg_dict.get("stl", {}).get("margin", 0.0)),
        stl_beta=float(cfg_dict.get("stl", {}).get("beta", 10.0)),
        stl_temp=float(cfg_dict.get("stl", {}).get("temp", 0.1)),
        stl_nx=int(cfg_dict.get("stl", {}).get("n_x", 32)),
        stl_ny=int(cfg_dict.get("stl", {}).get("n_y", 32)),
        stl_nt=int(cfg_dict.get("stl", {}).get("n_t", 16)),
        stl_every=int(cfg_dict.get("stl", {}).get("every", 10)),
        stl_x_min=cfg_dict.get("stl", {}).get("x_min", None),
        stl_x_max=cfg_dict.get("stl", {}).get("x_max", None),
        stl_y_min=cfg_dict.get("stl", {}).get("y_min", None),
        stl_y_max=cfg_dict.get("stl", {}).get("y_max", None),
        stl_t_min=cfg_dict.get("stl", {}).get("t_min", None),
        stl_t_max=cfg_dict.get("stl", {}).get("t_max", None),

        results_dir=str(cfg_dict.get("io", {}).get("results_dir", "results")),
        tag=str(cfg_dict.get("tag", "run")),
        save_ckpt=bool(cfg_dict.get("io", {}).get("save_ckpt", True)),
        save_frames=bool(cfg_dict.get("io", {}).get("save_frames", True)),
        frames_idx=tuple(cfg_dict.get("io", {}).get("frames_idx", (0, 8, 15))),
        save_figs=bool(cfg_dict.get("io", {}).get("save_figs", True)),
        print_every=int(cfg_dict.get("io", {}).get("print_every", 10)),
        seed=int(cfg_dict.get("seed", 0)),
    )

    # --- setup ----------------------------------------------------------------
    seed_everything(cfg.seed)
    device = _select_device(cfg.device)
    dtype = torch.get_default_dtype()

    # Precompute dense grid for sampling & for frame export
    X, Y, T, XYT = grid2d(
        n_x=cfg.n_x, n_y=cfg.n_y, n_t=cfg.n_t,
        x_min=cfg.x_min, x_max=cfg.x_max, y_min=cfg.y_min, y_max=cfg.y_max,
        t_min=cfg.t_min, t_max=cfg.t_max, device=device, dtype=dtype
    )

    model = MLP(in_dim=3, out_dim=1, hidden=cfg.hidden, activation=cfg.activation).to(device)
    model = _maybe_compile(model, cfg.compile)

    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, amsgrad=cfg.amsgrad)

    if cfg.scheduler == "cosine":
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))
    elif cfg.scheduler == "step":
        sched = optim.lr_scheduler.StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma)
    else:
        sched = None

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    # Prepare logging
    _ensure_dir(cfg.results_dir)
    csv_path = Path(cfg.results_dir) / f"heat2d_{cfg.tag}.csv"
    logger = CSVLogger(csv_path, header=["epoch", "lr", "loss", "loss_pde", "loss_bcic", "loss_stl"])

    # --- training loop --------------------------------------------------------
    saved: list[str] = []
    n_total = XYT.shape[0]
    batch = min(cfg.batch, n_total)
    hard_k = max(0, int(cfg.rar_hard_frac * batch)) if (cfg.rar_pool > 0 and cfg.rar_every > 0) else 0

    for epoch in range(cfg.epochs):
        model.train()
        # RAR: sample a pool, select hardest by |residual|
        if hard_k > 0 and cfg.rar_pool > 0 and (epoch % cfg.rar_every == 0):
            pool_idx = torch.randint(0, n_total, (min(cfg.rar_pool, n_total),), device=device)
            pool = XYT[pool_idx]
            pool.requires_grad_(True)
            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                res_pool = residual_heat2d(model, pool, alpha=cfg.alpha).squeeze(-1)
                scores = res_pool.abs()
            hard_idx_rel = torch.topk(scores, k=min(hard_k, scores.numel()), largest=True).indices
            hard_coords = pool[hard_idx_rel].detach()  # detach, will re‑enable grad below
            # Fill the rest randomly
            rand_k = batch - hard_coords.shape[0]
            rand_coords = XYT[torch.randint(0, n_total, (rand_k,), device=device)]
            coords = torch.cat([hard_coords, rand_coords], dim=0)
        else:
            idx = torch.randint(0, n_total, (batch,), device=device)
            coords = XYT[idx]

        coords = coords.requires_grad_(True)

        with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
            res = residual_heat2d(model, coords, alpha=cfg.alpha)
            loss_pde = (res.square()).mean()

            loss_bcic = bc_ic_heat2d(
                model,
                x_min=cfg.x_min, x_max=cfg.x_max, y_min=cfg.y_min, y_max=cfg.y_max,
                t_min=cfg.t_min, t_max=cfg.t_max,
                device=device, dtype=dtype,
            )

            # Optional STL penalty (computed every stl_every epochs to save time)
            if cfg.stl_use and _HAS_STL and cfg.stl_weight > 0.0 and (epoch % max(1, cfg.stl_every) == 0):
                loss_stl = _stl_penalty(model, cfg, device, dtype=dtype)
            else:
                loss_stl = torch.tensor(0.0, device=device, dtype=dtype)

            loss = loss_pde + cfg.bcic_weight * loss_bcic + loss_stl

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.grad_clip is not None:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
        scaler.step(opt)
        scaler.update()
        if sched is not None:
            sched.step()

        # Logging & progress
        lr_now = next(iter(opt.param_groups))['lr']
        logger.append([epoch, float(lr_now), float(loss), float(loss_pde), float(loss_bcic), float(loss_stl)])

        if (epoch % max(1, cfg.print_every) == 0) or (epoch == cfg.epochs - 1):
            # Lightweight print – avoids tqdm dependency
            print(f"[heat2d] epoch={epoch:04d} lr={lr_now:.2e} loss={float(loss):.4e} pde={float(loss_pde):.4e} bcic={float(loss_bcic):.4e} stl={float(loss_stl):.4e}")

    # --- artifacts ------------------------------------------------------------
    if cfg.save_ckpt:
        ckpt_path = Path(cfg.results_dir) / f"heat2d_{cfg.tag}.pt"
        torch.save({"model": model.state_dict(), "config": cfg.__dict__}, ckpt_path)
        saved.append(str(ckpt_path))

    if cfg.save_frames:
        with torch.no_grad():
            for raw_k in cfg.frames_idx:
                k = int(raw_k)
                k = max(0, min(k, cfg.n_t - 1))  # clamp
                # Build input for the k‑th time slice
                inp = torch.stack([
                    X[:, :, k].reshape(-1),
                    Y[:, :, k].reshape(-1),
                    T[:, :, k].reshape(-1)
                ], dim=-1).to(device)
                u = model(inp).reshape(cfg.n_x, cfg.n_y).detach().cpu().numpy()

                npy = Path(cfg.results_dir) / f"heat2d_{cfg.tag}_t{k}.npy"
                np.save(npy, u)
                saved.append(str(npy))

                if cfg.save_figs:
                    import matplotlib.pyplot as plt  # lazy import
                    gradmag = _gradmag_numpy(u)
                    plt.figure()
                    plt.imshow(gradmag.T, origin="lower", aspect="auto")  # transpose for (x,y) orientation
                    plt.colorbar(label="|∇u|")
                    plt.xlabel("x‑index"); plt.ylabel("y‑index")
                    plt.title(f"2‑D Heat |∇u|, frame t[{k}]")
                    figp = Path(cfg.results_dir) / f"heat2d_{cfg.tag}_gradmag_t{k}.png"
                    plt.tight_layout()
                    plt.savefig(figp, dpi=150)
                    plt.close()
                    saved.append(str(figp))

    return saved
