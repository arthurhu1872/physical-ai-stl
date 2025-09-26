# scripts/train_burgers_torchphysics.py
from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import sys
import time


# ---------------------------------------------------------------------------


@dataclass
class Args:
    # PDE / domain
    x_min: float = -1.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0
    nu: float = 0.01

    # Model
    hidden: Sequence[int] = (64, 64, 64, 64)
    activation: str = "tanh"  # TorchPhysics FCN default is Tanh()

    # Sampling
    n_pde: int = 4096  # points in interior per step
    n_ic: int = 1024  # points on t = t_min
    n_bc: int = 512  # points on x = {x_min, x_max}
    seed: int = 7

    # STL penalty (|u| <= u_max globally in time and space samples)
    lambda_stl: float = 0.0
    u_max: float = 1.0

    # Optimization
    lr: float = 1e-3
    max_steps: int = 5000
    device: str = "auto"  # "auto" | "cpu" | "gpu"
    precision: int = 32  # 16 | 32 | 64 (PyTorch Lightning precision)
    log_every_n_steps: int = 100

    # Export
    n_x: int = 201
    n_t: int = 201
    results: Path = Path("results")
    tag: str = "run"

    # Fallback / CI
    dryrun: bool = False


def _parse_args() -> Args:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Domain & PDE
    p.add_argument("--x-min", type=float, default=Args.x_min)
    p.add_argument("--x-max", type=float, default=Args.x_max)
    p.add_argument("--t-min", type=float, default=Args.t_min)
    p.add_argument("--t-max", type=float, default=Args.t_max)
    p.add_argument("--nu", type=float, default=Args.nu, help="Viscosity ν")
    # Model
    p.add_argument("--hidden", type=int, nargs="+", default=list(Args.hidden))
    p.add_argument("--activation", type=str, default=Args.activation)
    # Sampling
    p.add_argument("--n-pde", type=int, default=Args.n_pde)
    p.add_argument("--n-ic", type=int, default=Args.n_ic)
    p.add_argument("--n-bc", type=int, default=Args.n_bc)
    p.add_argument("--seed", type=int, default=Args.seed)
    # STL
    p.add_argument(
        "--lambda-stl",
        type=float,
        default=Args.lambda_stl,
        dest="lambda_stl",
        help="Weight for STL penalty (0 disables).",
    )
    p.add_argument(
        "--u-max",
        type=float,
        default=Args.u_max,
        dest="u_max",
        help="Max |u| in G[t_min,t_max] |u| <= u_max.",
    )
    # Optimization
    p.add_argument("--lr", type=float, default=Args.lr)
    p.add_argument("--max-steps", type=int, default=Args.max_steps)
    p.add_argument(
        "--device", type=str, choices=("auto", "cpu", "gpu"), default=Args.device
    )
    p.add_argument("--precision", type=int, choices=(16, 32, 64), default=Args.precision)
    p.add_argument("--log-every-n-steps", type=int, default=Args.log_every_n_steps)
    # Export
    p.add_argument("--n-x", type=int, default=Args.n_x)
    p.add_argument("--n-t", type=int, default=Args.n_t)
    p.add_argument("--results", type=Path, default=Args.results)
    p.add_argument("--tag", type=str, default=Args.tag)
    # Fallback / CI
    p.add_argument(
        "--dryrun", action="store_true", help="Skip training; write a tiny placeholder artifact."
    )
    args = Args(**vars(p.parse_args()))
    return args


# ---------------------------------------------------------------------------


def _maybe_placeholder(args: Args) -> Path | None:
    if args.dryrun:
        try:
            import torch  # noqa: F401
        except Exception as exc:  # pragma: no cover
            print(f"Torch not available: {exc}")
            return None
        args.results.mkdir(parents=True, exist_ok=True)
        out = args.results / f"burgers_{args.tag}.pt"
        import torch

        ckpt = {
            "u": torch.zeros(4, 4),
            "X": torch.linspace(0, 1, 4),
            "T": torch.linspace(0, 1, 4),
            "u_max": 0.0,
            "args": asdict(args),
            "loss": None,
            "meta": {"mode": "dryrun"},
        }
        torch.save(ckpt, out)
        print(f"[DRYRUN] wrote placeholder to {out}")
        return out
    return None


# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    # Fast path if user requested a dry run (CI/quick smoke).
    ph = _maybe_placeholder(args)
    if ph is not None:
        return

    # Import heavy deps lazily and fail back to placeholder if missing.
    try:
        import pytorch_lightning as pl  # type: ignore
        import torch
        import torchphysics as tp  # type: ignore
    except Exception as exc:
        print(
            f"[WARN] TorchPhysics stack not available ({exc!s}). Falling back to --dryrun artifact."
        )
        args.dryrun = True
        _maybe_placeholder(args)
        return

    # Reproducibility
    torch.manual_seed(args.seed)

    # --- Spaces & domains ---------------------------------------------------
    X = tp.spaces.R1("x")
    T = tp.spaces.R1("t")
    U = tp.spaces.R1("u")

    Omega = tp.domains.Interval(X, lower_bound=args.x_min, upper_bound=args.x_max)
    time_interval = tp.domains.Interval(
        T, lower_bound=args.t_min, upper_bound=args.t_max
    )

    # --- Samplers -----------------------------------------------------------
    # Interior Ω×I  (for PDE residual)
    sampler_pde = tp.samplers.RandomUniformSampler(
        Omega * time_interval, n_points=args.n_pde
    )
    # Initial condition Ω×{t_min}
    sampler_ic = tp.samplers.RandomUniformSampler(
        Omega * time_interval.boundary_left, n_points=args.n_ic
    )
    # Dirichlet boundary ∂Ω×I (both ends)
    sampler_bc = tp.samplers.RandomUniformSampler(
        Omega.boundary * time_interval, n_points=args.n_bc
    )

    # --- Residuals ----------------------------------------------------------
    # Burgers PDE: u_t + u*u_x − nu*u_xx = 0
    def residual_pde(u, x, t):
        u_t = tp.utils.grad(u, t)
        u_x = tp.utils.grad(u, x)
        u_xx = tp.utils.laplacian(u, x)  # in 1‑D, Laplacian is ∂²/∂x²
        return u_t + u * u_x - args.nu * u_xx

    # Initial condition at t = t_min: u(x, t_min) = -sin(pi * (scaled x))
    # We scale x from [x_min,x_max] to [0,1] to get a single‑period sine.
    x_span = max(args.x_max - args.x_min, 1e-9)

    def u0(xd):
        x01 = (xd[:, 0] - args.x_min) / x_span
        return -torch.sin(math.pi * x01).unsqueeze(-1)

    def residual_ic(u, x, t):
        return u - u0(x)

    # Zero Dirichlet at x boundaries
    def residual_bc(u):
        return u  # target is 0 on both boundaries

    # Optional STL: penalize violations of |u| ≤ u_max at sampled interior points.
    # Use sqrt(lambda) so the squared loss scales by lambda.
    sqrt_lambda = math.sqrt(max(args.lambda_stl, 0.0))

    def residual_stl(u):
        if sqrt_lambda == 0.0:
            # Returning a zero residual of correct shape avoids branching in the Condition.
            return u * 0.0
        return sqrt_lambda * torch.nn.functional.relu(torch.abs(u) - args.u_max)

    # --- Model --------------------------------------------------------------
    norm = tp.models.NormalizationLayer(Omega * time_interval)
    fcn = tp.models.FCN(input_space=X * T, output_space=U, hidden=tuple(args.hidden))
    model = tp.models.Sequential(norm, fcn)

    # --- Conditions ---------------------------------------------------------
    cond_pde = tp.conditions.PINNCondition(
        module=model, sampler=sampler_pde, residual_fn=residual_pde
    )
    cond_ic = tp.conditions.PINNCondition(
        module=model, sampler=sampler_ic, residual_fn=residual_ic
    )
    cond_bc = tp.conditions.PINNCondition(
        module=model, sampler=sampler_bc, residual_fn=residual_bc
    )
    # STL penalty operates on the same interior sampler (can be different); it is a soft safety loss.
    cond_stl = tp.conditions.PINNCondition(
        module=model, sampler=sampler_pde, residual_fn=lambda u, *_: residual_stl(u)
    )

    train_conditions = [cond_pde, cond_ic, cond_bc]
    if args.lambda_stl > 0:
        train_conditions.append(cond_stl)

    # --- Optimizer & solver -------------------------------------------------
    optim = tp.OptimizerSetting(optimizer_class=torch.optim.Adam, lr=args.lr)
    solver = tp.solver.Solver(train_conditions=train_conditions, optimizer_setting=optim)

    # --- Trainer ------------------------------------------------------------
    use_gpu = (args.device == "gpu") or (
        args.device == "auto" and torch.cuda.is_available()
    )
    accelerator = "gpu" if use_gpu else "cpu"
    trainer = pl.Trainer(
        devices=1,
        accelerator=accelerator,
        precision=args.precision,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        max_steps=args.max_steps,
        log_every_n_steps=args.log_every_n_steps,
        benchmark=True,
    )

    print(
        f"[INFO] Training on {accelerator.upper()} for {args.max_steps} steps "
        f"(ν={args.nu}, hidden={tuple(args.hidden)}, λ_STL={args.lambda_stl})"
    )

    t0 = time.time()
    trainer.fit(solver)
    train_time = time.time() - t0
    final_loss = None  # Lightning hides the running loss here; keep None for compact artifact

    # --- Export a compact artifact -----------------------------------------
    # Evaluate on a dense Cartesian grid to save results in a framework‑agnostic format.
    n_x, n_t = int(args.n_x), int(args.n_t)
    Xg = torch.linspace(args.x_min, args.x_max, n_x)
    Tg = torch.linspace(args.t_min, args.t_max, n_t)
    xs, ts = torch.meshgrid(Xg, Tg, indexing="ij")  # shapes (n_x, n_t)
    pts = torch.stack([xs.flatten(), ts.flatten()], dim=1)
    points = tp.spaces.Points(pts, space=X * T)
    with torch.no_grad():
        ug = model.forward(points).as_tensor.reshape(n_x, n_t)  # (x,t)
    u_grid = ug.T.contiguous()  # (t,x) for convenient heatmaps

    out_dir = args.results
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"burgers_{args.tag}.pt"
    meta = {
        "torchphysics": getattr(sys.modules.get("torchphysics"), "__version__", "unknown"),
        "torch": torch.__version__,
        "device": accelerator,
        "train_time_s": round(train_time, 3),
        "nu": args.nu,
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    torch.save(
        {
            "u": u_grid,
            "X": Xg,
            "T": Tg,
            "u_max": float(torch.max(torch.abs(u_grid))),
            "args": asdict(args),
            "loss": final_loss,
            "meta": meta,
        },
        out_path,
    )
    umax = float(torch.max(torch.abs(u_grid)))
    print(
        f"[OK] Saved results to {out_path.resolve()} (grid: {n_t}×{n_x}, u_max={umax:.4f})"
    )


if __name__ == "__main__":
    main()
