"""Minimal Neuromancer integration test."""
from __future__ import annotations

def neuromancer_version() -> str:
    """Return the installed Neuromancer version string.

    Raises
    ------
    ImportError
        If ''neuromancer'' is not installed.
    """
    try:
        import neuromancer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("neuromancer not installed") from e
    return getattr(neuromancer, "__version__", "unknown")

__all__ = ["neuromancer_version"]
