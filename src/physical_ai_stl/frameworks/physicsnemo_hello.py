"""Minimal NVIDIA PhysicsNeMo integration test."""
from __future__ import annotations

def physicsnemo_version() -> str:
    """Return the installed PhysicsNeMo version string.

    Raises
    ------
    ImportError
        If ``physicsnemo`` is not installed.
    """
    try:
        import physicsnemo  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("physicsnemo not installed") from e
    return getattr(physicsnemo, "__version__", "unknown")

__all__ = ["physicsnemo_version"]
