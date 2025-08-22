# src/physical_ai_stl/datasets/__init__.py

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "SyntheticSTLNetDataset",
    "BoundedAtomicSpec",
    "stlnet_synthetic",
]

# Map public attribute -> "relative.module[:object]".
# Objects are resolved only when first accessed to keep imports snappy.
_LAZY: dict[str, str] = {
    # Submodule
    "stlnet_synthetic": ".stlnet_synthetic",
    # Objects re‑exported at package level
    "SyntheticSTLNetDataset": ".stlnet_synthetic:SyntheticSTLNetDataset",
    "BoundedAtomicSpec": ".stlnet_synthetic:BoundedAtomicSpec",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - tiny import shim
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'physical_ai_stl.datasets' has no attribute {name!r}")
    if ":" in target:
        mod_name, qual = target.split(":", 1)
        obj = getattr(import_module(mod_name, __name__), qual)
        globals()[name] = obj  # cache for future lookups
        return obj
    # Return the submodule itself (e.g., `stlnet_synthetic`)
    module = import_module(target, __name__)
    globals()[name] = module
    return module


def __dir__() -> list[str]:  # pragma: no cover - tiny shim
    # Expose both already‑bound globals and the lazy attributes
    return sorted(list(globals().keys()) + list(_LAZY.keys()))


# Help IDEs and type checkers with concrete symbols without runtime cost.
if TYPE_CHECKING:  # pragma: no cover
    from . import stlnet_synthetic as stlnet_synthetic  # noqa: F401
    from .stlnet_synthetic import BoundedAtomicSpec as BoundedAtomicSpec  # noqa: F401
    from .stlnet_synthetic import SyntheticSTLNetDataset as SyntheticSTLNetDataset  # noqa: F401
