from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Mapping
from typing import Any, TYPE_CHECKING

# ------------------------------------------------------------------------------
# Public surface
# ------------------------------------------------------------------------------

__all__ = [
    # Models (lazily loaded on first access)
    "MLP",
    # Registry & helpers
    "MODEL_REGISTRY",
    "register",
    "available",
    "build",
]

# ------------------------------------------------------------------------------
# Lazy exports (attribute -> "module:object")
# ------------------------------------------------------------------------------

_LAZY: Mapping[str, str] = {
    "MLP": "physical_ai_stl.models.mlp:MLP",
}


def _import_module(name: str):
    return importlib.import_module(name)


def __getattr__(name: str) -> Any:  # pragma: no cover - tiny shim
    qual = _LAZY.get(name)
    if qual is None:
        raise AttributeError(f"module 'physical_ai_stl.models' has no attribute {name!r}")
    mod_name, obj_name = qual.split(":")
    obj = getattr(_import_module(mod_name), obj_name)
    globals()[name] = obj  # cache for subsequent accesses
    return obj


def __dir__() -> list[str]:  # pragma: no cover - tiny shim
    return sorted(list(globals().keys()) + list(_LAZY.keys()))


# ------------------------------------------------------------------------------
# Lightweight model registry
# ------------------------------------------------------------------------------

# A builder takes (*args, **kwargs) and returns an initialized model instance.
ModelBuilder = Callable[..., Any]

MODEL_REGISTRY: dict[str, ModelBuilder] = {}


def _norm(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def register(name: str, builder: ModelBuilder, *, aliases: Iterable[str] = ()) -> ModelBuilder:
    key = _norm(name)
    if key in MODEL_REGISTRY and MODEL_REGISTRY[key] is not builder:
        raise KeyError(f"Model name '{name}' is already registered to a different builder.")
    MODEL_REGISTRY[key] = builder
    for alias in aliases:
        akey = _norm(alias)
        if akey in MODEL_REGISTRY and MODEL_REGISTRY[akey] is not builder:
            raise KeyError(f"Alias '{alias}' is already registered to a different builder.")
        MODEL_REGISTRY[akey] = builder
    return builder


def available() -> list[str]:
    return sorted(MODEL_REGISTRY.keys())


def build(name: str, *args: Any, **kwargs: Any) -> Any:
    key = _norm(name)
    try:
        builder = MODEL_REGISTRY[key]
    except KeyError as e:
        opts = ", ".join(available()) or "<empty>"
        raise KeyError(f"Unknown model '{name}'. Available: {opts}") from e
    return builder(*args, **kwargs)


# ------------------------------------------------------------------------------
# Default registrations (kept lazy and dependency‑light)
# ------------------------------------------------------------------------------

# Avoid importing torch/MLP at import time. The builder imports on demand.
def _build_mlp(*args: Any, **kwargs: Any) -> Any:
    from .mlp import MLP  # local import to keep top‑level import fast

    return MLP(*args, **kwargs)


# Common aliases for convenience in configs.
register("mlp", _build_mlp, aliases=("dense", "fc", "fully_connected"))


# ------------------------------------------------------------------------------
# Static imports for type checkers only (no runtime impact)
# ------------------------------------------------------------------------------

if TYPE_CHECKING:  # pragma: no cover
    from .mlp import MLP as MLP  # re‑export for IDEs / static analysis
