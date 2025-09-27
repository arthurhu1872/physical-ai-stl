from __future__ import annotations

from importlib import import_module, metadata as _metadata


# Public constants for clarity in messages and for downstream tooling.
SPATIAL_SPEC_DIST_NAME: str = "spatial-spec"
SPATIAL_SPEC_MODULE_NAME: str = "spatial_spec"


def spatial_spec_version() -> str:
    # Import the module first; tests rely on this raising ImportError
    # when SpaTiaL is unavailable.
    try:
        mod = import_module(SPATIAL_SPEC_MODULE_NAME)
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "spatial-spec (SpaTiaL Specifications) is not installed. "
            "Install with: pip install spatial-spec"
        ) from e

    # Prefer the module's __version__ attribute if present.
    ver = getattr(mod, "__version__", None)
    if isinstance(ver, str) and ver:
        return ver

    # Fall back to the distribution version metadata.
    try:
        return _metadata.version(SPATIAL_SPEC_DIST_NAME)
    except Exception:
        return "unknown"


def spatial_spec_available() -> bool:
    try:
        import_module(SPATIAL_SPEC_MODULE_NAME)
        return True
    except Exception:
        return False


__all__ = [
    "SPATIAL_SPEC_DIST_NAME",
    "SPATIAL_SPEC_MODULE_NAME",
    "spatial_spec_version",
    "spatial_spec_available",
]
