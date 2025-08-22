from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core smooth aggregations
# ---------------------------------------------------------------------------

def softmin(x: torch.Tensor, *, temp: float = 0.1, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    if temp <= 0:
        raise ValueError(f"temp must be > 0, got {temp}")
    return -(temp * torch.logsumexp(-x / temp, dim=dim, keepdim=keepdim))


def softmax(x: torch.Tensor, *, temp: float = 0.1, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    if temp <= 0:
        raise ValueError(f"temp must be > 0, got {temp}")
    return temp * torch.logsumexp(x / temp, dim=dim, keepdim=keepdim)


def soft_and(a: torch.Tensor, b: torch.Tensor, *, temp: float = 0.1) -> torch.Tensor:
    a, b = torch.broadcast_tensors(a, b)
    stacked = torch.stack([a, b], dim=-1)
    return softmin(stacked, temp=temp, dim=-1)


def soft_or(a: torch.Tensor, b: torch.Tensor, *, temp: float = 0.1) -> torch.Tensor:
    a, b = torch.broadcast_tensors(a, b)
    stacked = torch.stack([a, b], dim=-1)
    return softmax(stacked, temp=temp, dim=-1)


def soft_not(r: torch.Tensor) -> torch.Tensor:
    return -r


# ---------------------------------------------------------------------------
# Predicates → per-time robustness margins
# ---------------------------------------------------------------------------

def pred_leq(u: torch.Tensor, c: float | torch.Tensor) -> torch.Tensor:
    c = torch.as_tensor(c, dtype=u.dtype, device=u.device)
    return c - u


def pred_geq(u: torch.Tensor, c: float | torch.Tensor) -> torch.Tensor:
    c = torch.as_tensor(c, dtype=u.dtype, device=u.device)
    return u - c


def pred_abs_leq(u: torch.Tensor, c: float | torch.Tensor) -> torch.Tensor:
    c = torch.as_tensor(c, dtype=u.dtype, device=u.device)
    return c - u.abs()


def pred_linear_leq(x: torch.Tensor, a: torch.Tensor, b: float | torch.Tensor) -> torch.Tensor:
    b = torch.as_tensor(b, dtype=x.dtype, device=x.device)
    ax = (x * a).sum(dim=-1)
    return b - ax


# ---------------------------------------------------------------------------
# Temporal operators over a time axis
# ---------------------------------------------------------------------------

def _move_time_last(x: torch.Tensor, time_dim: int) -> Tuple[torch.Tensor, int]:
    time_dim = int(time_dim) % x.ndim
    if time_dim != x.ndim - 1:
        x = x.movedim(time_dim, -1)
    return x, time_dim


def always(margins: torch.Tensor, *, temp: float = 0.1, time_dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    return softmin(margins, temp=temp, dim=time_dim, keepdim=keepdim)


def eventually(margins: torch.Tensor, *, temp: float = 0.1, time_dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    return softmax(margins, temp=temp, dim=time_dim, keepdim=keepdim)


def _unfold_time(x: torch.Tensor, *, window: int, stride: int) -> torch.Tensor:
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    return x.unfold(dimension=-1, size=window, step=stride)


def _windowed_soft_agg(x: torch.Tensor, *, window: int, stride: int, temp: float, kind: str) -> torch.Tensor:
    if temp <= 0:
        raise ValueError(f"temp must be > 0, got {temp}")
    xw = _unfold_time(x, window=window, stride=stride)  # (..., L, W)
    if kind == "max":
        # temp * logsumexp(x / temp) over the window dimension
        y = temp * torch.logsumexp(xw / temp, dim=-1)
    elif kind == "min":
        y = -(temp * torch.logsumexp(-xw / temp, dim=-1))
    else:
        raise ValueError("kind must be 'min' or 'max'")
    return y  # (..., L)


def always_window(
    margins: torch.Tensor,
    *,
    window: int,
    stride: int = 1,
    temp: float = 0.1,
    time_dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:
    x, old_dim = _move_time_last(margins, time_dim)
    y = _windowed_soft_agg(x, window=window, stride=stride, temp=temp, kind="min")
    if keepdim:
        y = y.unsqueeze(-1)
    if old_dim != x.ndim - 1:
        y = y.movedim(-1, old_dim)
    return y


def eventually_window(
    margins: torch.Tensor,
    *,
    window: int,
    stride: int = 1,
    temp: float = 0.1,
    time_dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:
    x, old_dim = _move_time_last(margins, time_dim)
    y = _windowed_soft_agg(x, window=window, stride=stride, temp=temp, kind="max")
    if keepdim:
        y = y.unsqueeze(-1)
    if old_dim != x.ndim - 1:
        y = y.movedim(-1, old_dim)
    return y


def shift_left(x: torch.Tensor, *, steps: int, time_dim: int = -1, pad_value: float = float("nan")) -> torch.Tensor:
    if steps < 0:
        raise ValueError("steps must be non-negative")
    x_m, old_dim = _move_time_last(x, time_dim)
    T = x_m.shape[-1]
    steps = min(int(steps), T)
    if steps == 0:
        out = x_m
    else:
        tail = torch.full_like(x_m[..., :steps], fill_value=pad_value)
        out = torch.cat([x_m[..., steps:], tail], dim=-1)
    if old_dim != x_m.ndim - 1:
        out = out.movedim(-1, old_dim)
    return out


# ---------------------------------------------------------------------------
# Loss: drive robustness positive (violations → penalty)
# ---------------------------------------------------------------------------

@dataclass
class STLPenaltyConfig:
    weight: float = 1.0
    margin: float = 0.0            # desire robustness >= margin
    kind: str = "softplus"         # {'softplus', 'hinge', 'sqhinge', 'logistic'}
    beta: float = 10.0             # sharpness for 'softplus' / 'logistic'
    reduction: str = "mean"        # {'mean', 'sum', 'none'}


class STLPenalty(nn.Module):

    def __init__(
        self,
        weight: float = 1.0,
        margin: float = 0.0,
        *,
        kind: str = "softplus",
        beta: float = 10.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.register_buffer("margin", torch.as_tensor(float(margin)))
        self.weight = float(weight)
        self.kind = str(kind).lower()
        self.beta = float(beta)
        self.reduction = str(reduction).lower()
        if self.kind not in {"softplus", "hinge", "sqhinge", "logistic"}:
            raise ValueError(f"Unsupported kind: {kind}")
        if self.reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")

    def forward(self, robustness: torch.Tensor) -> torch.Tensor:
        delta = self.margin - robustness  # positive when violating desired margin
        if self.kind == "softplus":
            loss = F.softplus(self.beta * delta) / self.beta
        elif self.kind == "logistic":
            loss = torch.log1p(torch.exp(self.beta * delta)) / self.beta
        elif self.kind == "hinge":
            loss = torch.clamp(delta, min=0.0)
        elif self.kind == "sqhinge":
            loss = torch.clamp(delta, min=0.0).square()
        else:  # pragma: no cover
            raise AssertionError("unreachable")

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        # 'none' returns elementwise

        return loss * self.weight


__all__ = [
    # aggregations
    "softmin",
    "softmax",
    "soft_and",
    "soft_or",
    "soft_not",
    # predicates
    "pred_leq",
    "pred_geq",
    "pred_abs_leq",
    "pred_linear_leq",
    # temporal
    "always",
    "eventually",
    "always_window",
    "eventually_window",
    "shift_left",
    # penalty
    "STLPenalty",
    "STLPenaltyConfig",
]
