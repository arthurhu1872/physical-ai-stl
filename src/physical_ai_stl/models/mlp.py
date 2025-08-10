"""Simple multi-layer perceptron used for PINN examples.

This module defines a generic fully connected neural network with
configurable input/output dimensions, hidden layer sizes and activation
function.  It is intentionally lightweight to minimise dependencies and
provide a clear template for students to extend.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Iterable, Type


class MLP(nn.Module):
    """A basic fully connected neural network.

    Parameters
    ----------
    in_dim : int
        Dimension of the input vector.
    out_dim : int
        Dimension of the output vector.
    hidden : Iterable[int], optional
        Sizes of hidden layers. Defaults to three layers of 128 neurons.
    act : Type[nn.Module], optional
        Activation class to instantiate after each hidden layer. Defaults to ``nn.Tanh``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Iterable[int] | None = None,
        act: Type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        if hidden is None:
            hidden = (128, 128, 128)
        layers: list[nn.Module] = []
        dim_prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(dim_prev, h))
            layers.append(act())
            dim_prev = h
        layers.append(nn.Linear(dim_prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, in_dim)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch, out_dim)``.
        """
        return self.net(x)
