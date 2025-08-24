"""Simple MLP used in PINN examples."""
from __future__ import annotations

from typing import Sequence
import torch
import torch.nn as nn

__all__ = ["MLP"]

class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int] = (64, 64, 64),
        activation: nn.Module = nn.Tanh(),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        # Use a fresh activation instance for each layer to avoid shared state across layers
        if isinstance(activation, nn.Module):
            act_cls = activation.__class__
        else:
            raise TypeError("activation must be an instance of torch.nn.Module")
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(act_cls())
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
