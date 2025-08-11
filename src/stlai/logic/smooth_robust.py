import numpy as np


def softmin(values: np.ndarray, tau: float = 0.1, axis: int = -1) -> np.ndarray:
    """
    Smooth approximation of the minimum using the log-sum-exp trick.

    Args:
        values: Input array.
        tau: Temperature parameter controlling the smoothness.
        axis: Axis over which to reduce.

    Returns:
        An array of reduced dimension approximating the elementwise minimum.
    """
    values = np.asarray(values)
    return -tau * np.log(np.sum(np.exp(-values / tau), axis=axis))


def softmax(values: np.ndarray, tau: float = 0.1, axis: int = -1) -> np.ndarray:
    """
    Smooth approximation of the maximum using the log-sum-exp trick.

    Args:
        values: Input array.
        tau: Temperature parameter controlling the smoothness.
        axis: Axis over which to reduce.

    Returns:
        An array of reduced dimension approximating the elementwise maximum.
    """
    values = np.asarray(values)
    return tau * np.log(np.sum(np.exp(values / tau), axis=axis))


def rho_bound_le(u: np.ndarray, umax: float) -> np.ndarray:
    """
    Compute robustness for the predicate u <= umax.

    Positive values indicate satisfaction, negative indicate violation.

    Args:
        u: Array of signal values.
        umax: Upper bound value.

    Returns:
        Array of robustness values.
    """
    return umax - np.asarray(u)


def G_soft_over_time(rho_t: np.ndarray, tau: float = 0.1) -> np.ndarray:
    """
    Smoothly aggregate robustness over time for the always (G) operator.

    Takes the soft minimum over the last dimension (time) of the input.

    Args:
        rho_t: Array with time as the last axis.
        tau: Temperature parameter for smoothness.

    Returns:
        Array representing the smooth minimum robustness over time.
    """
    return softmin(rho_t, tau=tau, axis=-1)


def Everywhere_soft_over_space(rho_xy: np.ndarray, tau: float = 0.1) -> np.ndarray:
    """
    Smoothly aggregate robustness over space for the everywhere operator.

    Flattens the spatial dimensions and takes a soft minimum.

    Args:
        rho_xy: Array where the last two axes correspond to spatial dimensions.
        tau: Temperature parameter for smoothness.

    Returns:
        Array representing the smooth minimum robustness over space.
    """
    rho_flat = rho_xy.reshape(*rho_xy.shape[:-2], -1)
    return softmin(rho_flat, tau=tau, axis=-1)


def spec_loss_margin(rho_hat: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """
    Compute a non-negative loss encouraging robustness above a margin.

    Args:
        rho_hat: Robustness values.
        margin: Desired minimum robustness.

    Returns:
        Non-negative array where violations contribute positively to the loss.
    """
    return np.maximum(0.0, margin - np.asarray(rho_hat))
