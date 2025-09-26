from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

import numpy as np

# Optional torch import kept inside a guard so this script remains importable
# even in environments without torch.
try:  # pragma: no cover
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = object  # type: ignore


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def _seed_everything(seed: int = 1337) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # no‑op on CPU
        # Best‑effort determinism; safe on CPU as well.
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]


def _device(requested: str | None = None) -> str:
    if torch is None:  # pragma: no cover
        return "cpu"
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _chunked_forward(model: nn.Module, coords: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
    if coords.shape[0] <= chunk:
        return model(coords)
    outs = []
    for i in range(0, coords.shape[0], chunk):
        outs.append(model(coords[i:i + chunk]))
    return torch.cat(outs, dim=0)


# --------------------------------------------------------------------------------------
# Minimal MLP with sane defaults (activation‑aware init)
# --------------------------------------------------------------------------------------

class MLP(nn.Module):  # type: ignore[misc]

    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 1,
        hidden: tuple[int, ...] = (64, 64, 64),
        activation: nn.Module | None = None,
        last_layer_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if activation is None:
            activation = nn.Tanh()  # good default for PINNs

        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(_fresh_activation(activation))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

        # Initialization: Xavier for tanh/sigmoid; He for ReLU‑like; keep last layer small.
        for m in self.net:
            if isinstance(m, nn.Linear):
                act = activation.__class__.__name__.lower()
                if "tanh" in act or "sigmoid" in act:
                    nn.init.xavier_uniform_(m.weight)
                elif "relu" in act or "silu" in act or "gelu" in act:
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # small last layer
        last = [m for m in self.net if isinstance(m, nn.Linear)][-1]
        last.weight.data.mul_(last_layer_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, in_dim) -> (N, out_dim)
        return self.net(x)


def _fresh_activation(proto: nn.Module) -> nn.Module:
    # Return a fresh instance of the given activation prototype.
    cls = proto.__class__
    try:
        return cls(**{k: v for k, v in vars(proto).items() if not k.startswith("_")})
    except TypeError:
        return cls()  # fallback


# --------------------------------------------------------------------------------------
# PDE helpers – 2‑D heat equation
# --------------------------------------------------------------------------------------

@dataclass
class SquareDomain:
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    def sample_interior(self, n: int, sobol: bool = True, device: str = "cpu") -> torch.Tensor:
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required for training.")
        if sobol:
            # Sobol on [0,1]^3 then scale
            eng = torch.quasirandom.SobolEngine(dimension=3)
            u = eng.draw(n)
        else:
            u = torch.rand(n, 3)
        x = self.x_min + (self.x_max - self.x_min) * u[:, 0:1]
        y = self.y_min + (self.y_max - self.y_min) * u[:, 1:2]
        t = self.t_min + (self.t_max - self.t_min) * u[:, 2:3]
        out = torch.cat([x, y, t], dim=1)
        return out.to(device)

    def sample_boundary(self, n: int, device: str = "cpu") -> torch.Tensor:
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required for training.")
        # choose one of 4 sides uniformly
        side = torch.randint(0, 4, (n, 1))
        u = torch.rand(n, 2)  # for the two free coordinates
        # map to coordinates depending on side
        x = torch.where(
            side == 0,
            torch.full((n, 1), self.x_min),  # x=0 edge
            torch.where(side == 1, torch.full((n, 1), self.x_max), torch.nan),  # x=1
        )
        y = torch.where(
            side == 2,
            torch.full((n, 1), self.y_min),  # y=0
            torch.where(side == 3, torch.full((n, 1), self.y_max), torch.nan),  # y=1
        )
        # fill NaNs with random values in the interior for the free coordinate(s)
        x = torch.where(torch.isnan(x), self.x_min + (self.x_max - self.x_min) * u[:, 0:1], x)
        y = torch.where(torch.isnan(y), self.y_min + (self.y_max - self.y_min) * u[:, 1:2], y)
        t = self.t_min + (self.t_max - self.t_min) * torch.rand(n, 1)
        out = torch.cat([x, y, t], dim=1)
        return out.to(device)

    def sample_initial(self, n: int, device: str = "cpu") -> torch.Tensor:
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required for training.")
        x = self.x_min + (self.x_max - self.x_min) * torch.rand(n, 1)
        y = self.y_min + (self.y_max - self.y_min) * torch.rand(n, 1)
        t = torch.full((n, 1), self.t_min)
        return torch.cat([x, y, t], dim=1).to(device)


def gaussian_ic(
    x: torch.Tensor,
    y: torch.Tensor,
    cx: float = 0.5,
    cy: float = 0.5,
    sigma: float = 0.08,
) -> torch.Tensor:
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    return torch.exp(-r2 / (2.0 * sigma ** 2))


def heat_residual(model: nn.Module, coords: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    coords.requires_grad_(True)
    u = model(coords)  # (N,1)
    # first derivatives
    grads = torch.autograd.grad(u, coords, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]
    u_t = grads[:, 2:3]
    # second derivatives
    u_xx = torch.autograd.grad(u_x, coords, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, coords, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    r = u_t - alpha * (u_xx + u_yy)
    return r


def boundary_dirichlet_loss(model: nn.Module, coords: torch.Tensor, value: float = 0.0) -> torch.Tensor:
    u = model(coords)
    return torch.mean((u - value) ** 2)


def initial_condition_loss(
    model: nn.Module,
    coords: torch.Tensor,
    ic_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    x, y = coords[:, 0:1], coords[:, 1:2]
    target = ic_fn(x, y).detach()
    pred = model(coords)
    return torch.mean((pred - target) ** 2)


# --------------------------------------------------------------------------------------
# Evaluation / MoonLight helpers
# --------------------------------------------------------------------------------------

def eval_on_grid(
    model: nn.Module,
    nx: int,
    ny: int,
    nt: int,
    device: str = "cpu",
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for evaluation.")
    model.eval()
    with torch.no_grad():
        xs = torch.linspace(0.0, 1.0, nx, device=device).view(-1, 1)
        ys = torch.linspace(0.0, 1.0, ny, device=device).view(-1, 1)
        ts = torch.linspace(t_min, t_max, nt, device=device).view(-1, 1)
        # Build full set of (x,y,t) points in a streaming fashion over time to limit memory.
        u_frames: list[torch.Tensor] = []
        for k in range(nt):
            t = ts[k].expand(nx * ny, 1)
            x = xs.repeat_interleave(ny, dim=0)
            y = ys.repeat(nx, 1)
            coords = torch.cat([x, y, t], dim=1)
            vals = _chunked_forward(model, coords, chunk=nx * ny).view(nx, ny, 1)  # (nx,ny,1)
            u_frames.append(vals.cpu())
        u = torch.cat(u_frames, dim=2).numpy().astype(np.float32)
        return (
            u,
            xs.squeeze(1).cpu().numpy(),
            ys.squeeze(1).cpu().numpy(),
            ts.squeeze(1).cpu().numpy(),
        )


def try_moonlight_audit(
    u: np.ndarray,
    nx: int,
    ny: int,
    mls_path: Path,
    formula: str = "contain",
    threshold: float | None = None,
) -> None:
    # lazy import
    try:
        from physical_ai_stl.monitoring.moonlight_helper import (  # type: ignore
            build_grid_graph,
            field_to_signal,
            get_monitor,
            load_script_from_file,
        )
    except Exception as exc:  # pragma: no cover
        print(f"[MoonLight] Skipping audit (helper not available): {exc}")
        return

    try:
        mls = load_script_from_file(str(mls_path))
        mon = get_monitor(mls, formula)
        graph = build_grid_graph(nx, ny)
        # Choose a simple, data‑driven threshold if none is provided
        if threshold is None:
            mu, sigma = float(u.mean()), float(u.std())
            threshold = mu + 0.5 * sigma
        sig = field_to_signal(u, threshold=threshold)
        out = mon.monitor_graph_time_series(graph, sig)  # type: ignore[attr-defined]
        out_arr = np.array(out, dtype=float)
        print(f"[MoonLight] monitor '{formula}' – first 5 values: {np.round(out_arr[:5], 4)}")
        print(
            f"[MoonLight] summary – "
            f"min={out_arr.min():.4f}, mean={out_arr.mean():.4f}, max={out_arr.max():.4f}"
        )
    except Exception as exc:  # pragma: no cover
        print(f"[MoonLight] Skipping audit (monitor failed): {exc}")


# --------------------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # domain/physics
    alpha: float = 0.1
    # sampling
    n_interior: int = 4096
    n_boundary: int = 1024
    n_initial: int = 1024
    sobol: bool = True
    # model/opt
    hidden: tuple[int, ...] = (64, 64, 64)
    lr: float = 3e-3
    epochs: int = 200
    w_pde: float = 1.0
    w_bc: float = 1.0
    w_ic: float = 1.0
    # eval grid
    nx: int = 48
    ny: int = 48
    nt: int = 33
    # misc
    seed: int = 1337
    device: str | None = None


def train_heat2d(cfg: TrainConfig, out_dir: Path, audit: bool, mls: Path) -> Path:
    if torch is None:  # pragma: no cover
        raise SystemExit("PyTorch is required. Install torch to train this model.")

    _seed_everything(cfg.seed)
    device = _device(cfg.device)
    print(f"[setup] device={device}  alpha={cfg.alpha}  epochs={cfg.epochs}")

    model = MLP(in_dim=3, out_dim=1, hidden=cfg.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    dom = SquareDomain()

    def ic_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return gaussian_ic(x, y, cx=0.5, cy=0.5, sigma=0.08)

    # Light progress meter without hard dependency on tqdm
    try:
        from tqdm.auto import trange  # type: ignore
    except Exception:  # pragma: no cover

        def trange(n: int, **kw: object) -> range:  # type: ignore
            return range(n)

    for step in trange(cfg.epochs, desc="train", leave=False):
        model.train()
        # Sample fresh collocation points each epoch (better generalization)
        X_int = dom.sample_interior(cfg.n_interior, sobol=cfg.sobol, device=device)
        X_bc = dom.sample_boundary(cfg.n_boundary, device=device)
        X_ic = dom.sample_initial(cfg.n_initial, device=device)

        # PDE residual
        r = heat_residual(model, X_int, alpha=cfg.alpha)
        loss_pde = torch.mean(r ** 2)

        # Boundary and initial condition losses
        loss_bc = boundary_dirichlet_loss(model, X_bc, value=0.0)
        loss_ic = initial_condition_loss(model, X_ic, ic_fn)

        loss = cfg.w_pde * loss_pde + cfg.w_bc * loss_bc + cfg.w_ic * loss_ic

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (step + 1) % max(1, cfg.epochs // 10) == 0 or step == 0:
            print(
                f"[{step + 1:04d}/{cfg.epochs}] loss={float(loss):.4e}  "
                f"(pde={float(loss_pde):.3e}, bc={float(loss_bc):.3e}, ic={float(loss_ic):.3e})"
            )

    # Evaluate on a structured grid and save artifact
    u, xs, ys, ts = eval_on_grid(model, cfg.nx, cfg.ny, cfg.nt, device=device)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "heat2d_run.npz"
    np.savez_compressed(
        out_path,
        u=u,
        x=xs,
        y=ys,
        t=ts,
        alpha=np.array(cfg.alpha, dtype=np.float32),
    )
    print(f"[save] wrote {out_path}  (u shape = {u.shape}, dtype = {u.dtype})")

    if audit:
        try_moonlight_audit(u, cfg.nx, cfg.ny, mls_path=mls, formula="contain")

    return out_path


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Train a 2‑D heat‑equation PINN and optionally audit with MoonLight STREL."
        )
    )
    # training
    ap.add_argument("--epochs", type=int, default=200, help="Training steps (default: 200).")
    ap.add_argument("--alpha", type=float, default=0.1, help="Diffusivity alpha (default: 0.1).")
    ap.add_argument("--lr", type=float, default=3e-3, help="Learning rate (default: 3e-3).")
    ap.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        default=(64, 64, 64),
        help="Hidden widths, e.g. --hidden 64 64 64.",
    )
    ap.add_argument("--n-interior", type=int, default=4096, help="Interior collocation points per epoch.")
    ap.add_argument("--n-boundary", type=int, default=1024, help="Boundary points per epoch.")
    ap.add_argument("--n-initial", type=int, default=1024, help="Initial condition points per epoch.")
    ap.add_argument("--no-sobol", action="store_true", help="Disable Sobol quasi‑random sampling.")
    # eval grid
    ap.add_argument("--nx", type=int, default=48)
    ap.add_argument("--ny", type=int, default=48)
    ap.add_argument("--nt", type=int, default=33)
    # audit
    ap.add_argument("--audit", action="store_true", help="Run MoonLight STREL audit after training.")
    ap.add_argument(
        "--mls",
        type=Path,
        default=Path("scripts/specs/contain_hotspot.mls"),
        help="Path to .mls script.",
    )
    # misc
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default=None, help="torch device (default: auto).")
    ap.add_argument("--out", type=Path, default=Path("results/heat2d"), help="Output directory.")
    args = ap.parse_args()

    # Build config
    cfg = TrainConfig(
        alpha=float(args.alpha),
        n_interior=int(args.n_interor) if hasattr(args, "n_interor") else int(args.n_interior),
        n_boundary=int(args.n_boundary),
        n_initial=int(args.n_initial),
        sobol=not args.no_sobol,
        hidden=tuple(int(h) for h in args.hidden),
        lr=float(args.lr),
        epochs=int(args.epochs),
        nx=int(args.nx),
        ny=int(args.ny),
        nt=int(args.nt),
        seed=int(args.seed),
        device=args.device,
    )

    # Ensure default spec exists (convenience for fresh clones)
    if args.audit and not args.mls.exists():
        args.mls.parent.mkdir(parents=True, exist_ok=True)
        args.mls.write_text(
            "signal { bool hot; }\n"
            "domain boolean;\n"
            "formula contain = eventually (!(somewhere (hot)));\n"
        )
        print(f"[spec] created default MoonLight spec at {args.mls}")

    _ = train_heat2d(cfg, out_dir=args.out, audit=bool(args.audit), mls=args.mls)


if __name__ == "__main__":
    main()
