"""Differentiable soft semantics for Signal Temporal Logic (temporal only).

We provide:
- softmin / softmax,
- predicate "u <= c" as a (signed) margin,
- temporal always/eventually over time axis,
- a simple penalty module that encourages positive robustness.
"""

from __future__ import annotations
import torch
from torch import nn


def softmin(x: torch.Tensor, temp: float = 0.1, dim: int = -1) -> torch.Tensor:
    """Smooth approximation of min using log-sum-exp."""
    return -(temp * torch.logsumexp(-x / temp, dim=dim))


def softmax(x: torch.Tensor, temp: float = 0.1, dim: int = -1) -> torch.Tensor:
    """Smooth approximation of max using log-sum-exp."""
    return temp * torch.logsumexp(x / temp, dim=dim)


def pred_leq(u: torch.Tensor, c: float) -> torch.Tensor:
    """Return per-time margin for (u <= c): margin = c - u (positive when satisfied)."""
    return (torch.as_tensor(c, dtype=u.dtype, device=u.device) - u)


def always(margins: torch.Tensor, temp: float = 0.1, time_dim: int = -1) -> torch.Tensor:
    """Soft-G over time: approximate min over the time dimension."""
    return softmin(margins, temp=temp, dim=time_dim)


def eventually(margins: torch.Tensor, temp: float = 0.1, time_dim: int = -1) -> torch.Tensor:
    """Soft-F over time: approximate max over the time dimension."""
    return softmax(margins, temp=temp, dim=time_dim)


class STLPenalty(nn.Module):
    """Hinge-like penalty that drives robustness positive with a margin."""

    def __init__(self, weight: float = 1.0, margin: float = 0.0) -> None:
        super().__init__()
        self.weight = float(weight)
        self.margin = float(margin)

    def forward(self, robustness: torch.Tensor) -> torch.Tensor:
        # Penalize negative robustness (violations) with a softplus
        return torch.nn.functional.softplus(self.margin - robustness).mean() * self.weight
