"""
Minimal stub torch module for environments where PyTorch is unavailable.

This stub provides a subset of the torch API required by the physical-ai-stl package.
It wraps NumPy functions to mimic PyTorch behavior sufficiently for testing.
"""

import numpy as _np

# Expose ndarray as Tensor
Tensor = _np.ndarray


def stack(tensors, dim=0):
    """Stack a sequence of arrays along a new dimension."""
    return _np.stack(tensors, axis=dim)


def logsumexp(x, dim=None):
    """Compute log-sum-exp of input array along a given dimension."""
    return _np.log(_np.sum(_np.exp(x), axis=dim))


def minimum(a, b):
    """Elementwise minimum of two arrays."""
    return _np.minimum(a, b)


def maximum(a, b):
    """Elementwise maximum of two arrays."""
    return _np.maximum(a, b)


def relu(x):
    """Rectified Linear Unit (ReLU) function."""
    return _np.maximum(0, x)


def ones_like(x):
    """Return an array of ones with the same shape and type as x."""
    return _np.ones_like(x)


def zeros_like(x):
    """Return an array of zeros with the same shape and type as x."""
    return _np.zeros_like(x)


def cat(tensors, dim=0):
    """Concatenate a sequence of arrays along an existing dimension."""
    return _np.concatenate(tensors, axis=dim)


def tanh(x):
    """Hyperbolic tangent function."""
    return _np.tanh(x)


def abs(x):
    """Absolute value function."""
    return _np.abs(x)
