from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Iterable, List
import numpy as np, torch
from torch import optim
from ..models.mlp import MLP
from ..training.grids import grid2d
from ..physics.heat2d import residual_heat2d, bc_ic_heat2d
from ..utils.seed import seed_everything
from ..utils.logger import CSVLogger
import torch.nn as nn

@dataclass
class Heat2DConfig:
    hidden: tuple[int, int, int] = (64, 64, 64)
    activation: str = "tanh"
    n_x: int = 64; n_y: int = 64; n_t: int = 16
    lr: float = 2e-3; epochs: int = 200; batch: int = 4096
    alpha: float = 0.1
    results_dir: str = "results"
    save_frames: bool = True
    frames_idx: Iterable[int] = (0, 8, 15)
    save_figs: bool = True
    tag: str = "run"; seed: int = 0

def _activation(name: str): 
    return {"tanh": nn.Tanh(), "relu": nn.ReLU(), "gelu": nn.GELU()}.get(name.lower(), nn.Tanh())

def _gradmag(u: torch.Tensor) -> np.ndarray:
    u_np = u.detach().cpu().numpy()
    gx = np.zeros_like(u_np); gy = np.zeros_like(u_np)
    gx[1:-1, :] = 0.5 * (u_np[2:, :] - u_np[:-2, :])
    gy[:, 1:-1] = 0.5 * (u_np[:, 2:] - u_np[:, :-2])
    return np.sqrt(gx * gx + gy * gy)

def run_heat2d(cfg_dict: Dict[str, Any]) -> List[str]:
    cfg = Heat2DConfig(
        hidden=tuple(cfg_dict.get("model", {}).get("hidden", (64,64,64))),
        activation=cfg_dict.get("model", {}).get("activation", "tanh"),
        n_x=cfg_dict.get("grid", {}).get("n_x", 64),
        n_y=cfg_dict.get("grid", {}).get("n_y", 64),
        n_t=cfg_dict.get("grid", {}).get("n_t", 16),
        lr=cfg_dict.get("optim", {}).get("lr", 2e-3),
        epochs=cfg_dict.get("optim", {}).get("epochs", 200),
        batch=cfg_dict.get("optim", {}).get("batch", 4096),
        alpha=cfg_dict.get("physics", {}).get("alpha", 0.1),
        results_dir=cfg_dict.get("io", {}).get("results_dir", "results"),
        save_frames=cfg_dict.get("io", {}).get("save_frames", True),
        frames_idx=tuple(cfg_dict.get("io", {}).get("frames_idx", (0,8,15))),
        save_figs=cfg_dict.get("io", {}).get("save_figs", True),
        tag=cfg_dict.get("tag", "run"),
        seed=cfg_dict.get("seed", 0),
    )
    seed_everything(cfg.seed); device = "cpu"
    X, Y, T, XYT = grid2d(n_x=cfg.n_x, n_y=cfg.n_y, n_t=cfg.n_t, device=device)
    model = MLP(in_dim=3, out_dim=1, hidden=cfg.hidden, activation=_activation(cfg.activation)).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    logger = CSVLogger(f"{cfg.results_dir}/heat2d_{cfg.tag}.csv", header=["epoch","loss","loss_pde","loss_bcic"])
    for epoch in range(cfg.epochs):
        idx = torch.randint(0, XYT.shape[0], (cfg.batch,), device=device)
        coords = XYT[idx]
        res = residual_heat2d(model, coords, alpha=cfg.alpha)
        loss_pde = res.square().mean()
        loss_bcic = bc_ic_heat2d(model, device=device)
        loss = loss_pde + loss_bcic
        opt.zero_grad(); loss.backward(); opt.step()
        logger.append([epoch, float(loss), float(loss_pde), float(loss_bcic)])
    saved: List[str] = []
    if cfg.save_frames:
        with torch.no_grad():
            for k in cfg.frames_idx:
                k = int(k)
                inp = torch.stack([X[:,:,k].reshape(-1), Y[:,:,k].reshape(-1), T[:,:,k].reshape(-1)], dim=-1)
                u = model(inp).reshape(cfg.n_x, cfg.n_y)
                npy = f"{cfg.results_dir}/heat2d_{cfg.tag}_t{k}.npy"; np.save(npy, u.numpy()); saved.append(npy)
                if cfg.save_figs:
                    import matplotlib.pyplot as plt
                    gradmag = _gradmag(u)
                    plt.figure(); plt.imshow(gradmag, origin="lower"); plt.colorbar(label="|∇u|")
                    plt.title(f"2D Heat |∇u|, frame {k}")
                    figp = f"{cfg.results_dir}/heat2d_{cfg.tag}_gradmag_t{k}.png"
                    plt.savefig(figp, dpi=150); plt.close(); saved.append(figp)
    return saved
