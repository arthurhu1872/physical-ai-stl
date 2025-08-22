from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import argparse
import math
import os

import torch
from torch import nn, optim

from physical_ai_stl.models.mlp import MLP
from physical_ai_stl.physics.diffusion1d import residual_loss, boundary_loss
from physical_ai_stl.training.grids import grid1d, sample_interior_1d
from physical_ai_stl.monitoring.stl_soft import (
    pred_leq, always, softmax as stl_softmax, STLPenalty
)
from physical_ai_stl.utils.seed import seed_everything
from physical_ai_stl.utils.logger import CSVLogger


# -----------------------------------------------------------------------------


def _auto_device(user_choice: Optional[str] = None) -> torch.device:
    if user_choice:
        return torch.device(user_choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


def _maybe_compile(module: nn.Module, do_compile: bool) -> nn.Module:
    if not do_compile:
        return module
    compile_fn = getattr(torch, "compile", None)
    if callable(compile_fn):  # pragma: no cover - depends on torch version
        return compile_fn(module)  # type: ignore[misc]
    return module


def _stl_spatial_reduce(u_xt: torch.Tensor, mode: str, temp: float) -> torch.Tensor:
    mode = str(mode).lower()
    if mode == "mean":
        return u_xt.mean(dim=0)
    if mode == "softmax":
        return stl_softmax(u_xt, temp=float(temp), dim=0, keepdim=False)  # type: ignore[arg-type]
    if mode == "amax":
        return u_xt.amax(dim=0)
    raise ValueError(f"Unknown stl_spatial mode: {mode!r}")


# -----------------------------------------------------------------------------


@dataclass
class Args:
    # model
    hidden: tuple[int, ...] = (64, 64, 64)
    activation: str = "tanh"
    out_act: Optional[str] = None

    # grid/domain
    nx: int = 128
    nt: int = 64
    x_min: float = 0.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    # optimization
    lr: float = 2e-3
    epochs: int = 200
    batch: int = 4096
    opt: str = "adam"               # adam | adamw
    weight_decay: float = 0.0
    sched: str = "none"             # none | onecycle | cosine
    grad_clip: float = 0.0

    # physics
    alpha: float = 0.1
    n_boundary: int = 256
    n_initial: int = 512
    sample_method: str = "sobol"    # sobol | uniform
    w_boundary: float = 1.0
    w_initial: float = 1.0

    # STL penalty (optional)
    stl_use: bool = False
    stl_weight: float = 0.0
    stl_u_max: float = 1.0
    stl_temp: float = 0.1
    stl_spatial: str = "mean"       # mean | softmax | amax
    stl_every: int = 1              # compute STL every k steps (k>=1)
    stl_nx: int = 64                # coarse grid for STL monitoring
    stl_nt: int = 64

    # system / numerics
    device: Optional[str] = None
    dtype: str = "float32"          # float32 | float64
    amp: bool = False               # AMP is conservative here (higher‑order grads)
    compile: bool = False
    seed: int = 0
    print_every: int = 25

    # I/O
    results_dir: str = "results"
    tag: str = "run"
    save_ckpt: bool = True
    resume: Optional[str] = None    # path to a saved checkpoint to resume from


def _parse() -> Args:
    p = argparse.ArgumentParser(description="Train 1‑D diffusion PINN with optional STL penalty.")
    # model
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64, 64], help="hidden layer sizes")
    p.add_argument("--activation", type=str, default="tanh", help="MLP hidden activation (tanh/relu/sine/...)")
    p.add_argument("--out-act", type=str, default=None, help="optional output activation (e.g., tanh)")

    # grid/domain
    p.add_argument("--nx", type=int, default=128)
    p.add_argument("--nt", type=int, default=64)
    p.add_argument("--x-min", type=float, default=0.0)
    p.add_argument("--x-max", type=float, default=1.0)
    p.add_argument("--t-min", type=float, default=0.0)
    p.add_argument("--t-max", type=float, default=1.0)

    # optimization
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--opt", type=str, default="adam", choices=["adam", "adamw"])
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--sched", type=str, default="none", choices=["none", "onecycle", "cosine"])
    p.add_argument("--grad-clip", type=float, default=0.0)

    # physics
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--n-boundary", type=int, default=256)
    p.add_argument("--n-initial", type=int, default=512)
    p.add_argument("--sample-method", type=str, default="sobol", choices=["sobol", "uniform"])
    p.add_argument("--w-boundary", type=float, default=1.0)
    p.add_argument("--w-initial", type=float, default=1.0)

    # STL
    p.add_argument("--stl-use", action="store_true", help="enable STL penalty")
    p.add_argument("--stl-weight", type=float, default=0.0)
    p.add_argument("--stl-u-max", type=float, default=1.0)
    p.add_argument("--stl-temp", type=float, default=0.1)
    p.add_argument("--stl-spatial", type=str, default="mean", choices=["mean", "softmax", "amax"])
    p.add_argument("--stl-every", type=int, default=1)
    p.add_argument("--stl-nx", type=int, default=64)
    p.add_argument("--stl-nt", type=int, default=64)

    # system / numerics
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--amp", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--print-every", type=int, default=25)

    # I/O
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--tag", type=str, default="run")
    p.add_argument("--save-ckpt", action="store_true")
    p.add_argument("--resume", type=str, default=None)

    a = p.parse_args()
    return Args(
        hidden=tuple(a.hidden), activation=a.activation, out_act=a.out_act,
        nx=a.nx, nt=a.nt, x_min=a.x_min, x_max=a.x_max, t_min=a.t_min, t_max=a.t_max,
        lr=a.lr, epochs=a.epochs, batch=a.batch, opt=a.opt, weight_decay=a.weight_decay,
        sched=a.sched, grad_clip=a.grad_clip,
        alpha=a.alpha, n_boundary=a.n_boundary, n_initial=a.n_initial, sample_method=a.sample_method,
        w_boundary=a.w_boundary, w_initial=a.w_initial,
        stl_use=bool(a.stl_use), stl_weight=a.stl_weight, stl_u_max=a.stl_u_max, stl_temp=a.stl_temp,
        stl_spatial=a.stl_spatial, stl_every=max(1, int(a.stl_every)), stl_nx=a.stl_nx, stl_nt=a.stl_nt,
        device=a.device, dtype=a.dtype, amp=bool(a.amp), compile=bool(a.compile),
        seed=a.seed, print_every=a.print_every,
        results_dir=a.results_dir, tag=a.tag, save_ckpt=bool(a.save_ckpt), resume=a.resume,
    )


# -----------------------------------------------------------------------------


def main() -> None:
    cfg = _parse()
    seed_everything(int(cfg.seed))

    # Device/dtype
    device = _auto_device(cfg.device)
    torch.set_default_dtype(getattr(torch, cfg.dtype))  # affects newly created tensors

    # Grid for training and coarse grid for STL monitoring
    X, T, XT = grid1d(
        n_x=cfg.nx, n_t=cfg.nt,
        x_min=cfg.x_min, x_max=cfg.x_max,
        t_min=cfg.t_min, t_max=cfg.t_max,
        device=device, dtype=getattr(torch, cfg.dtype)
    )
    Xs, Ts, XTs = grid1d(
        n_x=max(8, int(cfg.stl_nx)), n_t=max(4, int(cfg.stl_nt)),
        x_min=cfg.x_min, x_max=cfg.x_max,
        t_min=cfg.t_min, t_max=cfg.t_max,
        device=device, dtype=getattr(torch, cfg.dtype)
    )

    # Model
    model = MLP(
        in_dim=2, out_dim=1, hidden=cfg.hidden, activation=cfg.activation,
        out_activation=cfg.out_act, last_layer_scale=0.01,
        device=device, dtype=getattr(torch, cfg.dtype),
    )
    model = _maybe_compile(model, cfg.compile)

    # Optimizer & scheduler
    if cfg.opt == "adamw":
        opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.sched == "onecycle":
        sched = optim.lr_scheduler.OneCycleLR(
            opt, max_lr=float(cfg.lr), total_steps=int(cfg.epochs),
            pct_start=0.1, anneal_strategy="cos", div_factor=25.0, final_div_factor=1e4
        )
    elif cfg.sched == "cosine":
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(cfg.epochs))
    else:
        sched = None

    # AMP scaler
    use_autocast = (cfg.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_autocast)

    # Optional STL penalty
    penalty = STLPenalty(weight=float(cfg.stl_weight), margin=0.0, kind="softplus", beta=10.0)

    # I/O
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    log = CSVLogger(
        results_dir / f"diffusion1d_{cfg.tag}.csv",
        header=["epoch", "lr", "loss", "loss_pde", "loss_bcic", "loss_stl", "robustness"],
    )

    # (Optional) resume
    if cfg.resume and os.path.isfile(cfg.resume):
        ckpt = torch.load(cfg.resume, map_location="cpu")
        sd = ckpt.get("model")
        if sd:
            model.load_state_dict(sd, strict=True)
            print(f"[resume] Loaded model state from {cfg.resume}")

    # Training loop
    model.train()
    for epoch in range(int(cfg.epochs)):
        # Interior collocation points
        coords = sample_interior_1d(
            int(cfg.batch),
            x_min=cfg.x_min, x_max=cfg.x_max,
            t_min=cfg.t_min, t_max=cfg.t_max,
            method=cfg.sample_method, device=device, dtype=getattr(torch, cfg.dtype),
            seed=int(cfg.seed) + epoch,
        )
        coords.requires_grad_(True)

        with torch.cuda.amp.autocast(enabled=use_autocast):
            # PDE residual at collocation points (MSE)
            loss_pde = residual_loss(model, coords, alpha=cfg.alpha, reduction="mean")

            # Soft BC/IC penalties (new samples every epoch)
            loss_bcic = boundary_loss(
                model, x_left=cfg.x_min, x_right=cfg.x_max, t_min=cfg.t_min, t_max=cfg.t_max,
                device=device, dtype=getattr(torch, cfg.dtype), method=cfg.sample_method,
                n_boundary=cfg.n_boundary, n_initial=cfg.n_initial, seed=int(cfg.seed) + 13 * epoch,
                w_boundary=cfg.w_boundary, w_initial=cfg.w_initial,
            )

            # STL robustness: G_t ( reduce_x u(x,t) <= u_max )
            if cfg.stl_use and float(cfg.stl_weight) > 0.0 and (epoch % int(cfg.stl_every) == 0):
                u_xt = model(XTs).reshape(Xs.shape[0], Xs.shape[1])
                signal_t = _stl_spatial_reduce(u_xt, cfg.stl_spatial, cfg.stl_temp)
                margins = pred_leq(signal_t, cfg.stl_u_max)  # positive if satisfied
                rob = always(margins, temp=float(cfg.stl_temp), time_dim=0)
                loss_stl = penalty(rob)
            else:
                rob = torch.zeros((), device=device, dtype=getattr(torch, cfg.dtype))
                loss_stl = torch.zeros((), device=device, dtype=getattr(torch, cfg.dtype))

            loss = loss_pde + loss_bcic + loss_stl

        opt.zero_grad(set_to_none=True)
        if use_autocast:
            scaler.scale(loss).backward()
            # Gradient clipping (unscale first)
            if float(cfg.grad_clip) > 0.0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.grad_clip))
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if float(cfg.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.grad_clip))
            opt.step()

        if sched is not None:
            sched.step()

        # Logging/print
        lr_now = opt.param_groups[0]["lr"]
        log.append([epoch, lr_now, float(loss), float(loss_pde), float(loss_bcic), float(loss_stl), float(rob)])
        if (epoch % max(1, int(cfg.print_every)) == 0) or (epoch == int(cfg.epochs) - 1):
            print(
                f"[diffusion1d] epoch={epoch:04d} lr={lr_now:.2e} "
                f"loss={float(loss):.4e} pde={float(loss_pde):.4e} bcic={float(loss_bcic):.4e} stl={float(loss_stl):.4e}"
            )

    # --- artifacts ------------------------------------------------------------
    saved: list[str] = []

    if cfg.save_ckpt:
        ckpt_path = results_dir / f"diffusion1d_{cfg.tag}.pt"
        torch.save({"model": model.state_dict(), "config": vars(cfg)}, ckpt_path)
        saved.append(str(ckpt_path))

    # Save final field on the full grid
    model.eval()
    with torch.no_grad():
        U = model(XT).reshape(cfg.nx, cfg.nt).detach().to("cpu")
        X_cpu, T_cpu = X.detach().to("cpu"), T.detach().to("cpu")
    field_path = results_dir / f"diffusion1d_{cfg.tag}_field.pt"
    torch.save(
        {
            "u": U, "X": X_cpu, "T": T_cpu,
            "u_max": float(cfg.stl_u_max), "alpha": float(cfg.alpha),
            "config": vars(cfg),
        },
        field_path,
    )
    saved.append(str(field_path))

    print(f"[diffusion1d] done → {field_path} (and {len(saved)-1} other artifact(s))")


if __name__ == "__main__":
    main()
