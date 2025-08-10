"""Differentiable soft semantics for Signal Temporal Logic.

This module implements smoothed versions of the min/max operators and
predicates used to compute robustness for STL specifications.  Soft
approximations allow gradients to propagate through logic terms so
that STL penalties can be used directly in neural network training
loops.  We also include a simple hinge penalty to encourage positive
robustness with a safety margin.
"""

from __future__ import annotations

import torch
from torch import nn


def softmin(x: torch.Tensor, temp: float = 0.1, dim: int = -1) -> torch.Tensor:
    """Smooth approximation of the minimum operator.

    Uses the log-sum-exp trick: ``min(x) ≈ −temp * logsumexp(−x / temp)``.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    temp : float
        Temperature parameter controlling smoothness (smaller = sharper).
    dim : int
        Dimension over which to reduce.

    Returns
    -------
    torch.Tensor
        Approximated minimum along ``dim``.
    """
    return -torch.logsumexp(-x / temp, dim=dim) * temp


def softmax(x: torch.Tensor, temp: float = 0.1, dim: int = -1) -> torch.Tensor:
    """Smooth approximation of the maximum operator.

    Uses the log-sum-exp trick: ``max(x) ≈ temp * logsumexp(x / temp)``.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    temp : float
        Temperature parameter controlling smoothness (smaller = sharper).
    dim : int
        Dimension over which to reduce.

    Returns
    -------
    torch.Tensor
        Approximated maximum along ``dim``.
    """
    return torch.logsumexp(x / temp, dim=dim) * temp


def pred_leq(signal: torch.Tensor, threshold: float) -> torch.Tensor:
    """Robustness for predicate ``signal ≤ threshold``.

    Positive values indicate satisfaction with margin equal to ``threshold - signal``.
    """
    return threshold - signal


def pred_geq(signal: torch.Tensor, threshold: float) -> torch.Tensor:
    """Robustness for predicate ``signal ≥ threshold``.

    Positive values indicate satisfaction with margin equal to ``signal - threshold``.
    """
    return signal - threshold


def always(robustness: torch.Tensor, temp: float = 0.1, dim: int = -1) -> torch.Tensor:
    """Smooth approximation of the temporal ``always`` operator.

    Performs a soft minimum over the specified dimension.
    """
    return softmin(robustness, temp=temp, dim=dim)


def eventually(robustness: torch.Tensor, temp: float = 0.1, dim: int = -1) -> torch.Tensor:
    """Smooth approximation of the temporal ``eventually`` operator.

    Performs a soft maximum over the specified dimension.
    """
    return softmax(robustness, temp=temp, dim=dim)


class STLPenalty(nn.Module):
    """Hinge penalty encouraging positive robustness.

    Given a robustness scalar ``rho``, returns ``ReLU(margin - rho)``.  When
    robustness exceeds the margin, the penalty is zero.  Otherwise it
    penalises violations and near-violations of the specification.
    """

    def __init__(self, margin: float = 0.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, robustness: torch.Tensor) -> torch.Tensor:
        # If robustness is a vector, take mean penalty
        return torch.relu(self.margin - robustness).mean()
