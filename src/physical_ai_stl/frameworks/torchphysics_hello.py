from __future__ import annotations

def torchphysics_version() -> str:
    """Return TorchPhysics version or raise if not installed.

    Mirrors the *hello* helpers for other frameworks so tests and scripts can
    quickly check availability without importing heavy submodules.
    """
    try:
        import torchphysics  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("TorchPhysics is not installed. `pip install torchphysics`") from e

    ver = getattr(torchphysics, "__version__", None)
    return ver if isinstance(ver, str) and ver else "unknown"


__all__ = ["torchphysics_version"]
