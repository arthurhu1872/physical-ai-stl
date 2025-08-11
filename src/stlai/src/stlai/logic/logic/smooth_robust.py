import numpy as np


def softmin(values, dim=None, tau=0.1):
    """Smooth approximation of minimum using log-sum-exp.
    values: numpy array
    dim: axis along which to compute; if None flatten
    tau: temperature parameter (smaller -> closer to min)
    Returns a numpy array reduced along dim.
    """
    arr = np.asarray(values, dtype=float)
    if dim is None:
        arr = arr.ravel()
        return -tau * np.log(np.sum(np.exp(-arr / tau)))
    return -tau * np.log(np.sum(np.exp(-arr / tau), axis=dim))


def softmax(values, dim=None, tau=0.1):
    """Smooth approximation of maximum using log-sum-exp."""
    arr = np.asarray(values, dtype=float)
    if dim is None:
        arr = arr.ravel()
        return tau * np.log(np.sum(np.exp(arr / tau)))
    return tau * np.log(np.sum(np.exp(arr / tau), axis=dim))


def rho_bound_le(u, umax):
    """Robustness of the predicate u <= umax. Positive when satisfied."""
    return umax - u


def G_soft_over_time(rho_t, tau=0.1):
    """Smooth always (global) operator over the time dimension (axis=1)."""
    return softmin(rho_t, dim=1, tau=tau)


def Everywhere_soft_over_space(rho_xy, tau=0.1):
    """Smooth everywhere operator over spatial dimensions (last two dims)."""
    rho_xy = np.asarray(rho_xy)
    flat = rho_xy.reshape(rho_xy.shape[0], -1)
    return softmin(flat, dim=1, tau=tau)


def spec_loss_margin(rho_hat, margin=0.0):
    """Loss encouraging rho_hat >= margin."""
    diff = margin - rho_hat
    diff = np.maximum(0, diff)
    return diff.mean()
