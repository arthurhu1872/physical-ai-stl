"""Minimal Neuromancer integration test.

This module exposes a function that returns the installed Neuromancer version string.  Importing the package is enough to verify that the dependency is present and working.  The goal is simply to ensure the project can access Neuromancer when available without pulling in heavy runtime dependencies during tests.
"""
from __future__ import annotations

def neuromancer_version() -> str:
    """Return the ``neuromancer`` package version.

    Raises
    ------
    ImportError
        If ``neuromancer`` is not installed.
    """
    try:
        import neuromancer  # type: ignore
    except Exception as e:  # pragma: no cover - handled by tests
        raise ImportError("neuromancer not installed") from e
    return getattr(neuromancer, "__version__", "unknown")

__all__ = ["neuromancer_version"]
