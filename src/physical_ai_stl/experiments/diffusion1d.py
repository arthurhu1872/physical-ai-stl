"""Training loop for a 1D diffusion PINN with optional STL penalty."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from torch import nn, optim
import torch

from ..models.mlp import MLP
from ..monitoring.stl_soft import pred_leq, always, STLPenalty
from ..physics.diffusion1d import pde_residual, boundary_loss
from ..training.grids import grid1d
from ..utils.logger import CSVLogger
from ..utils.seed import seed_everything
__all__ = ["Diffusion1DConfig", "run_diffusion1d"]

@dataclass
class Diffusion1DConfig:
    hidden: tuple[int, int, int] = (64, 64, 64)
    activation: str = "tanh"
    n_x: int = 128
    n_t: int = 64
    lr: float = 2e-3
    epochs: int = 200
    batch: int = 4096
    alpha: float = 0.1
    stl_use: bool = False
    stl_weight: float = 0.0
    stl_u_max: float = 1.0
    stl_temp: float = 0.1
    results_dir: str = "results"
    tag: str = "run"
    seed: int = 0

def _activation(name: str) -> nn.Module:
    name = name.lower()
    return {"tanh": nn.Tanh(), "relu": nn.ReLU(), "gelu": nn.GELU()}.get(name, nn.Tanh())

def run_diffusion1d(cfg_dict: dict[str, Any]) -> str:
    cfg = Diffusion1DConfig(
        hidden=tuple(cfg_dict.get("model", {}).get("hidden", (64, 64, 64))),
        activation=cfg_dict.get("model", {}).get("activation", "tanh"),
        n_x=cfg_dict.get("grid", {}).get("n_x", 128),
        n_t=cfg_dict.get("grid", {}).get("n_t", 64),
        lr=cfg_dict.get("optim", {}).get("lr", 2e-3),
        epochs=cfg_dict.get("optim", {}).get("epochs", 200),
        batch=cfg_dict.get("optim", {}).get("batch", 4096),
        alpha=cfg_dict.get("physics", {}).get("alpha", 0.1),
        stl_use=cfg_dict.get("stl", {}).get("use", False),
        stl_weight=cfg_dict.get("stl", {}).get("weight", 0.0),
        stl_u_max=cfg_dict.get("stl", {}).get("u_max", 1.0),
        stl_temp=cfg_dict.get("stl", {}).get("temp", 0.1),
        results_dir=cfg_dict.get("io", {}).get("results_dir", "results"),
        tag=cfg_dict.get("tag", "run"),
        seed=cfg_dict.get("seed", 0),
    )
    seed_everything(cfg.seed)
    device = "cpu"
    X, T, XT = grid1d(n_x=cfg.n_x, n_t=cfg.n_t, device=device)
    model = MLP(in_dim=2, out_dim=1, hidden=cfg.hidden, activation=_activation(cfg.activation)).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    penalty = STLPenalty(weight=cfg.stl_weight, margin=0.0) if cfg.stl_use else None
    logger = CSVLogger(f"{cfg.results_dir}/diffusion1d_{cfg.tag}.csv",
                       header=["epoch", "loss", "loss_pde", "loss_bcic", "loss_stl", "rob"])
    for epoch in range(cfg.epochs):
        idx = torch.randint(0, XT.shape[0], (cfg.batch,), device=device)
        coords = XT[idx]
        res = pde_residual(model, coords, alpha=cfg.alpha)
        loss_pde = res.square().mean()
        loss_bcic = boundary_loss(model, device=device)
        loss_stl = torch.tensor(0.0)
        rob = torch.tensor(0.0)
        if penalty is not None:
            # Compute STL robustness on the entire space-time grid
            with torch.no_grad():
                inp = XT
            u = model(inp).reshape(cfg.n_x, cfg.n_t)
            u_mean = u.mean(dim=0)
            margins = pred_leq(u_mean, cfg.stl_u_max)
            rob = always(margins, temp=cfg.stl_temp, time_dim=0)
            loss_stl = penalty(rob)
        loss = loss_pde + loss_bcic + loss_stl
        opt.zero_grad()
        loss.backward()
        opt.step()
        logger.append([epoch, float(loss), float(loss_pde), float(loss_bcic),
                       float(loss_stl), float(rob)])
    # After training, evaluate model on full grid and save results
    with torch.no_grad():
        u = model(XT).reshape(cfg.n_x, cfg.n_t)
    out_path = f"{cfg.results_dir}/diffusion_{cfg.tag}.pt"
    torch.save({"u": u.cpu(), "X": X.cpu(), "T": T.cpu(), "u_max": float(cfg.stl_u_max)}, out_path)
    return out_path