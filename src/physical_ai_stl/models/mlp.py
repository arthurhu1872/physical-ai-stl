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
        for h in hidden:
            layers += [nn.Linear(last, h), activation]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
