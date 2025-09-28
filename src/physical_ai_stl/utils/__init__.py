from __future__ import annotations

from importlib import import_module
from typing import Any, TYPE_CHECKING


__all__ = [
    # Submodules
    "seed",
    "logger",
    # Primary re‑exports
    "seed_everything",
    "seed_worker",
    "torch_generator",
    "CSVLogger",
]

# Map attribute name -> "relative.module[:object]".
# Objects are imported only on first access to keep imports snappy.
_LAZY: dict[str, str] = {
    # Submodules
    "seed": ".seed",
    "logger": ".logger",
    # Helpers
    "seed_everything": ".seed:seed_everything",
    "seed_worker": ".seed:seed_worker",
    "torch_generator": ".seed:torch_generator",
    "CSVLogger": ".logger:CSVLogger",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - tiny import shim
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'physical_ai_stl.utils' has no attribute {name!r}")
    if ":" in target:
        mod_name, qual = target.split(":", 1)
        obj = getattr(import_module(mod_name, __name__), qual)
        globals()[name] = obj  # cache for future lookups
        return obj
    # Return the submodule itself (e.g., `seed`, `logger`)
    module = import_module(target, __name__)
    globals()[name] = module
    return module


def __dir__() -> list[str]:  # pragma: no cover - tiny shim
    # Expose both already‑bound globals and the lazy attributes
    return sorted(list(globals().keys()) + list(_LAZY.keys()))


# Help IDEs and type checkers with concrete symbols without runtime cost.
if TYPE_CHECKING:  # pragma: no cover
    from . import logger, seed  # noqa: F401
    from .logger import CSVLogger  # noqa: F401
    from .seed import (  # noqa: F401
        seed_everything,
        seed_worker,
        torch_generator,
    )
