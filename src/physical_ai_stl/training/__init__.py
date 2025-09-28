from __future__ import annotations

import importlib
from typing import Any, TYPE_CHECKING

# Map of lazily importable submodules.
_LAZY_MODULES = {
    "grids": "physical_ai_stl.training.grids",
}

# Forwarded attributes: provide nice `from physical_ai_stl.training import grid1d`.
# These will be resolved from the `grids` module on first access.
_FORWARD_ATTRS: dict[str, tuple[str, str]] = {
    # original API
    "grid1d": ("grids", "grid1d"),
    "grid2d": ("grids", "grid2d"),
    # new generators / helpers
    "grid3d": ("grids", "grid3d"),
    "spacing1d": ("grids", "spacing1d"),
    "spacing2d": ("grids", "spacing2d"),
    # samplers
    "sample_interior_1d": ("grids", "sample_interior_1d"),
    "sample_interior_2d": ("grids", "sample_interior_2d"),
    "sample_boundary_1d": ("grids", "sample_boundary_1d"),
    "sample_boundary_2d": ("grids", "sample_boundary_2d"),
    # simple domains
    "Box1D": ("grids", "Box1D"),
    "Box2D": ("grids", "Box2D"),
}

# What `from physical_ai_stl.training import *` exposes.
# (Note: star-import will import `grids` once to retrieve forwarded names.)
__all__ = sorted({*list(_LAZY_MODULES.keys()), *list(_FORWARD_ATTRS.keys())})


def _import_module(mod_path: str):  # pragma: no cover - trivial wrapper
    try:
        return importlib.import_module(mod_path)
    except ImportError as e:  # Provide a friendlier nudge for torch missing.
        msg = str(e).lower()
        if "torch" in msg or "pytorch" in msg:
            raise ImportError(
                "The 'grids' utilities require PyTorch. "
                'Install the extra with: pip install "physical-ai-stl[torch]"'
            ) from e
        raise


def __getattr__(name: str) -> Any:  # pragma: no cover - exercised implicitly
    # Submodule?
    if name in _LAZY_MODULES:
        module = _import_module(_LAZY_MODULES[name])
        globals()[name] = module
        return module
    # Forwarded attribute?
    if name in _FORWARD_ATTRS:
        mod_name, attr = _FORWARD_ATTRS[name]
        module = __getattr__(mod_name)  # ensure submodule is imported & cached
        value = getattr(module, attr)
        globals()[name] = value  # cache for subsequent lookups
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - trivial
    # Expose both already-bound globals and the lazy attributes
    return sorted(list(globals().keys()) + list(__all__))


# Help IDEs/type-checkers without paying runtime import cost.
if TYPE_CHECKING:  # pragma: no cover
    from . import grids as grids  # noqa: F401
    from .grids import (  # noqa: F401
        Box1D,
        Box2D,
        grid1d,
        grid2d,
        grid3d,
        sample_boundary_1d,
        sample_boundary_2d,
        sample_interior_1d,
        sample_interior_2d,
        spacing1d,
        spacing2d,
    )
