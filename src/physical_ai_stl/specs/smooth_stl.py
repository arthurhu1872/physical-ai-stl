"""
Differentiable smooth STL operators.

This module provides differentiable approximations to logical operators used in training neural networks with Signal Temporal Logic constraints. These functions implement smooth versions of min/max, conjunction, disjunction, temporal operators and a hinge penalty so that robustness signals can flow through gradient-based learning.
"""

from __future__ import annotations

import torch


def smooth_max(x: torch.Tensor, temp: float = 0.1, dim: int = -1) -> torch.Tensor:
    """Smooth approximation of the maximum operator.

    Uses the log-sum-exp trick: ``max(x) \approx temp * logsumexp(x / temp)``.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    temp : float, optional
        Temperature controlling the sharpness of the approximation. Smaller values
        make the approximation closer to the true max. Default is 0.1.
    dim : int, optional
        Dimension along which to compute the maximum. Default is the last dim.

    Returns
    -------
    torch.Tensor
        Approximated maximum values along ``dim``.
    """
    return torch.logsumexp(x / temp, dim=dim) * temp


def smooth_min(x: torch.Tensor, temp: float = 0.1, dim: int = -1) -> torch.Tensor:
    """Smooth approximation of the minimum operator.

    Uses the log-sum-exp trick: ``min(x) \approx -temp * logsumexp(-x / temp)``.

    Parameters and returns mirror those of :func:`smooth_max`.
    """
    return -torch.logsumexp(-x / temp, dim=dim) * temp


def robustness_bound(signal: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    """Compute robustness margins of a signal relative to bounds.

    The robustness margin is the signed distance to the allowed interval ``[lower, upper]``.
    Positive values indicate satisfaction and negative values violation. For each element
    of ``signal``, the margin is ``min(signal - lower, upper - signal)``.

    Parameters
    ----------
    signal : torch.Tensor
        Input signal values.
    lower : float
        Lower bound of the allowed interval.
    upper : float
        Upper bound of the allowed interval.

    Returns
    -------
    torch.Tensor
        Tensor of the same shape as ``signal`` containing the robustness margins.
    """
    return torch.min(signal - lower, upper - signal)


def stl_always(robustness: torch.Tensor, temp: float = 0.1, dim: int = -1) -> torch.Tensor:
    """Smooth approximation of the temporal "always" (globally) operator.

    This is computed as a soft minimum over the specified dimension.

    Parameters
    ----------
    robustness : torch.Tensor
        Robustness values over which to take the temporal minimum.
    temp : float, optional
        Temperature for the smooth minimum. Default is 0.1.
    dim : int, optional
        Dimension corresponding to time. Default is the last dimension.

    Returns
    -------
    torch.Tensor
        Soft minimum of the robustness values along ``dim``.
    """
    return smooth_min(robustness, temp=temp, dim=dim)


def stl_eventually(robustness: torch.Tensor, temp: float = 0.1, dim: int = -1) -> torch.Tensor:
    """Smooth approximation of the temporal "eventually" operator.

    This is computed as a soft maximum over the specified dimension.
    """
    return smooth_max(robustness, temp=temp, dim=dim)


def stl_and(rho1: torch.Tensor, rho2: torch.Tensor, temp: float = 0.1, dim: int = 0) -> torch.Tensor:
    """Smooth approximation of the conjunction of two robustness signals.

    Computes a soft minimum of ``rho1`` and ``rho2`` along a new dimension.
    """
    stacked = torch.stack((rho1, rho2), dim=dim)
    return smooth_min(stacked, temp=temp, dim=dim)


def stl_or(rho1: torch.Tensor, rho2: torch.Tensor, temp: float = 0.1, dim: int = 0) -> torch.Tensor:
    """Smooth approximation of the disjunction of two robustness signals.

    Computes a soft maximum of ``rho1`` and ``rho2`` along a new dimension.
    """
    stacked = torch.stack((rho1, rho2), dim=dim)
    return smooth_max(stacked, temp=temp, dim=dim)


def penalty_from_robustness(robustness: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """Convert robustness to a training penalty via a hinge loss.

    For a scalar or tensor robustness ``rho``, the penalty is ``ReLU(margin - rho)``.
    When ``rho >= margin``, the specification is satisfied and the penalty is zero.

    Parameters
    ----------
    robustness : torch.Tensor
        Robustness values to penalize.
    margin : float, optional
        Safety margin. Violations within this margin also incur penalty. Default is 0.0.

    Returns
    -------
    torch.Tensor
        Scalar penalty obtained by averaging over all elements of ``robustness``.
    """
    return torch.relu(margin - robustness).mean()
