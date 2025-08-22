"""Minimal SpaTiaL (spatial-spec) integration test."""
from __future__ import annotations

def spatial_spec_version() -> str:
    """Return the installed ``spatial_spec`` package version.

    Raises
    ------
    ImportError
        If ``spatial_spec`` is not installed.
    """
    try:
        import spatial_spec  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("spatial-spec not installed") from e
    return getattr(spatial_spec, "__version__", "unknown")

__all__ = ["spatial_spec_version"]
