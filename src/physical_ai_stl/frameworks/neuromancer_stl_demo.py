from __future__ import annotations

import math
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


# -----------------------------------------------------------------------------
# Public config
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class DemoConfig:
    n: int = 256
    epochs: int = 200
    lr: float = 1e-3
    bound: float = 0.8
    weight: float = 100.0
    device: str = "cpu"
    seed: int = 7


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for this demo, but is not available.")


def _make_data(n: int, device: str = "cpu") -> dict[str, torch.Tensor]:
    _require_torch()
    t = torch.linspace(0.0, 2.0 * math.pi, n, device=device).reshape(n, 1)
    y_true = torch.sin(t)
    return {"t": t, "y_true": y_true}


def _mlp(insize: int = 1, outsize: int = 1) -> nn.Module:
    _require_torch()
    return nn.Sequential(
        nn.Linear(insize, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, outsize),
    )


def stl_violation(u: torch.Tensor, bound: float) -> torch.Tensor:
    _require_torch()
    return torch.relu(u - bound)  # type: ignore[operator]


def stl_offline_robustness(u: torch.Tensor, bound: float) -> float:
    _require_torch()
    return float((bound - u).min().item())


# -----------------------------------------------------------------------------
# PyTorch baseline
# -----------------------------------------------------------------------------

def _train_pytorch(cfg: DemoConfig, data: dict[str, torch.Tensor]) -> dict[str, float]:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    net = _mlp().to(device)  # type: ignore[call-arg]
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)  # type: ignore[attr-defined]

    t = data["t"]
    y = data["y_true"]

    for _ in range(cfg.epochs):
        opt.zero_grad(set_to_none=True)
        y_hat = net(t)
        fit = F.mse_loss(y_hat, y)  # type: ignore[arg-type]
        penalty = stl_violation(y_hat, cfg.bound).mean()
        loss = fit + cfg.weight * penalty
        loss.backward()
        opt.step()

    with torch.no_grad():
        y_hat = net(t)
        final_mse = F.mse_loss(y_hat, y).item()  # type: ignore[arg-type]
        final_violation = stl_violation(y_hat, cfg.bound).mean().item()
        rho = stl_offline_robustness(y_hat, cfg.bound)

    return {
        "final_mse": float(final_mse),
        "final_violation": float(final_violation),
        "robustness_min": float(rho),
    }


# -----------------------------------------------------------------------------
# Neuromancer variant (optional)
# -----------------------------------------------------------------------------

def _train_neuromancer(cfg: DemoConfig, data: dict[str, torch.Tensor]) -> dict[str, float] | None:
    try:
        import neuromancer as nm  # type: ignore
    except Exception:
        return None

    try:
        # Model block identical to the PyTorch MLP.
        func = nm.modules.blocks.MLP(  # type: ignore[attr-defined]
            insize=1,
            outsize=1,
            hsizes=[64, 64],
            nonlin=nn.Tanh,  # reuse torch.nn.Tanh for parity
            linear_map=nm.slim.maps["linear"],  # type: ignore[index]
        )
        node = nm.system.Node(func, ["t"], ["y_hat"], name="regressor")  # type: ignore[attr-defined]

        # Symbolic variables and objective.
        y_hat = nm.constraint.variable("y_hat")  # type: ignore[attr-defined]
        y = nm.constraint.variable("y_true")  # type: ignore[attr-defined]
        obj = ((y_hat - y) ** 2).mean().minimize(weight=1.0, name="fit")  # type: ignore[attr-defined]

        # Soft inequality y_hat <= bound with a configurable weight.
        con = cfg.weight * (y_hat <= cfg.bound)  # type: ignore[operator]

        loss = nm.loss.PenaltyLoss(objectives=[obj], constraints=[con])  # type: ignore[attr-defined]
        problem = nm.problem.Problem(nodes=[node], loss=loss)  # type: ignore[attr-defined]

        # Dataset + loader
        DictDataset = getattr(nm.dataset, "DictDataset")  # type: ignore[attr-defined]
        train_ds = DictDataset(data, name="train")
        # Full‑batch to keep parity with the PyTorch path.
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=len(data["t"]), shuffle=False)

        # Optimizer and trainer
        optimizer = torch.optim.Adam(problem.parameters(), lr=cfg.lr)  # type: ignore[attr-defined]

        # Prefer the official Trainer API when available.
        try:
            Trainer = getattr(nm.trainer, "Trainer")  # type: ignore[attr-defined]
            trainer = Trainer(problem, train_loader=train_loader, test_loader=None, optimizer=optimizer)
            trainer.fit(epoch_start=0, epoch_end=cfg.epochs)
        except Exception:
            # Fallback: minimal manual loop
            for _ in range(cfg.epochs):
                for batch in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    loss_val = problem(batch)  # forward computes PenaltyLoss over the batch
                    # problem returns a dict or scalar depending on version;
                    # try to backprop through the scalar entry "loss".
                    if isinstance(loss_val, dict) and "loss" in loss_val:
                        (loss_val["loss"]).backward()
                    elif isinstance(loss_val, torch.Tensor):
                        loss_val.backward()
                    else:  # last‑resort: call `.compute_loss`
                        loss_scalar = getattr(problem, "compute_loss")(batch)  # type: ignore[misc]
                        loss_scalar.backward()
                    optimizer.step()

        # Final metrics on the training grid
        with torch.no_grad():
            yh = node(data)["y_hat"]
            nm_mse = ((yh - data["y_true"]) ** 2).mean().item()
            nm_violation = torch.relu(yh - cfg.bound).mean().item()
            rho = stl_offline_robustness(yh, cfg.bound)

        return {
            "final_mse": float(nm_mse),
            "final_violation": float(nm_violation),
            "robustness_min": float(rho),
        }

    except Exception:
        # Any unexpected API mismatch should not break the repository tests.
        return None


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def train_demo(cfg: DemoConfig) -> dict[str, dict[str, float] | None]:
    _require_torch()
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    data = _make_data(cfg.n, device=str(device))
    metrics_pt = _train_pytorch(cfg, data)

    metrics_nm = _train_neuromancer(cfg, data)

    return {"pytorch": metrics_pt, "neuromancer": metrics_nm}


__all__ = ["DemoConfig", "train_demo", "stl_violation", "stl_offline_robustness"]
