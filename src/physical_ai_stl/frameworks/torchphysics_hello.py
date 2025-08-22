"""Minimal TorchPhysics integration test."""
from __future__ import annotations

def torchphysics_version() -> str:
    """Return the installed TorchPhysics version string.

    Raises
    ------
    ImportError
        If ``torchphysics`` is not installed.
    """
    try:
        import torchphysics  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("torchphysics not installed") from e
    return getattr(torchphysics, "__version__", "unknown")

__all__ = ["torchphysics_version"]
