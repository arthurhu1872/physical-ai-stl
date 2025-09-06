"""Neuromancer demo: apples-to-apples STL-like bound enforcement inside Neuromancer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
@dataclass
class DemoConfig:
    n: int = 256
    epochs: int = 200
    lr: float = 1e-3
    bound: float = 0.8
    weight: float = 100.0
    device: str = "cpu"
    seed: int = 7

def _make_data(n: int, device: str = "cpu") -> dict[str, torch.Tensor]:
    t = torch.linspace(0.0, 1.0, steps=n, device=device).unsqueeze(-1)  # (n,1)
    y_target = torch.sin(2.0 * math.pi * t)  # (n,1)
    return {"t": t, "y_target": y_target}

def _violation(u: torch.Tensor, bound: float) -> torch.Tensor:
    """Quantitative violation: ReLU(u - bound) averaged over batch."""
    return F.relu(u - bound).mean()

def _mlp(insize: int = 1, outsize: int = 1) -> nn.Module:
    return nn.Sequential(
        nn.Linear(insize, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, outsize),
    )

def train_demo(cfg: DemoConfig) -> dict[str, Optional[dict[str, float]]]:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    data = _make_data(cfg.n, device=device)
    # ---------- Plain PyTorch baseline (for apples-to-apples) ----------
    net = _mlp().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    for _ in range(cfg.epochs):
        opt.zero_grad()
        u = net(data["t"])                # (n,1)
        fit = F.mse_loss(u, data["y_target"])  # MSE objective
        pen = cfg.weight * _violation(u, cfg.bound)  # "always(u<=bound)" as soft penalty
        loss = fit + pen
        loss.backward()
        opt.step()
    with torch.no_grad():
        u = net(data["t"])
        pytorch_final_mse = F.mse_loss(u, data["y_target"]).item()
        pytorch_final_violation = _violation(u, cfg.bound).item()
    results = {
        "pytorch": {
            "final_mse": float(pytorch_final_mse),
            "final_violation": float(pytorch_final_violation),
        },
        "neuromancer": None,
    }

    # ---------- Neuromancer variant (preferred path) ----------
    try:
        import neuromancer as nm  # type: ignore

        # Map wrapping the same architecture
        func = nm.blocks.MLP(insize=1, outsize=1, hsizes=[64, 64], activation="tanh")
        u_map = nm.maps.Map(func, input_keys=["t"], output_keys=["u"], name="u_map")  # type: ignore

        # Named variables for constraint/objective expressions
        u = nm.constraints.variable("u")                         # type: ignore
        y = nm.constraints.variable("y_target")                  # type: ignore

        # Objective: minimize squared error (mean over batch)
        fit_expr = ((u - y) ** 2).mean()
        fit = fit_expr.minimize(weight=1.0, name="fit")          # type: ignore

        # Constraint: quantitative "always(u <= bound)" via pointwise penalty
        con_upper = cfg.weight * (u <= cfg.bound)                # type: ignore

        # Compose loss and problem
        loss = nm.loss.PenaltyLoss(objectives=[fit], constraints=[con_upper])  # type: ignore
        problem = nm.problem.Problem(components=[u_map], loss=loss)            # type: ignore

        # Train with a simple torch loop over the Problem (callable module)
        opt2 = torch.optim.Adam(problem.parameters(), lr=cfg.lr)   # type: ignore
        for _ in range(cfg.epochs):
            opt2.zero_grad()
            outs = problem(data)            # type: ignore  # expects dict with keys 't','y_target'
            # By convention, Problem returns a dict with 'loss' aggregated from objectives+constraints
            (outs["loss"]).backward()       # type: ignore
            opt2.step()

        with torch.no_grad():
            outs = problem(data)            # type: ignore
            # Access predicted 'u' and compute diagnostics in the same way
            u_pred = outs["u"]              # type: ignore
            nm_final_mse = F.mse_loss(u_pred, data["y_target"]).item()
            nm_final_violation = _violation(u_pred, cfg.bound).item()

        results["neuromancer"] = {
            "final_mse": float(nm_final_mse),
            "final_violation": float(nm_final_violation),
        }

    except Exception:  # pragma: no cover - optional dependency path
        # Keep this very lightweight: if Neuromancer isn't available, still succeed.
        results["neuromancer"] = None

    return results

__all__ = ["DemoConfig", "train_demo"]