from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module, metadata as _metadata, util as _import_util
from typing import Any, TYPE_CHECKING

__all__ = [
    # Submodules (lazy)
    "rtamt_hello",
    "moonlight_hello",
    "moonlight_strel_hello",
    "spatial_demo",
    # Convenience re‑exports (lazy)
    "stl_hello_offline",
    "temporal_hello",
    "strel_hello",
    "spatial_run_demo",
    # Environment helpers
    "available_backends",
]

# ----- Lazy import shims -----------------------------------------------------

_SUBMODULES: Mapping[str, str] = {
    "rtamt_hello": "physical_ai_stl.monitors.rtamt_hello",
    "moonlight_hello": "physical_ai_stl.monitors.moonlight_hello",
    "moonlight_strel_hello": "physical_ai_stl.monitors.moonlight_strel_hello",
    "spatial_demo": "physical_ai_stl.monitors.spatial_demo",
}

# function_name -> "module_path:object"
_HELPERS: Mapping[str, str] = {
    "stl_hello_offline": "physical_ai_stl.monitors.rtamt_hello:stl_hello_offline",
    "temporal_hello": "physical_ai_stl.monitors.moonlight_hello:temporal_hello",
    "strel_hello": "physical_ai_stl.monitors.moonlight_strel_hello:strel_hello",
    "spatial_run_demo": "physical_ai_stl.monitors.spatial_demo:run_demo",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - tiny shim
    if name in _SUBMODULES:
        mod = import_module(_SUBMODULES[name])
        globals()[name] = mod
        return mod
    if name in _HELPERS:
        mod_name, obj_name = _HELPERS[name].split(":")
        obj = getattr(import_module(mod_name), obj_name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module 'physical_ai_stl.monitors' has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - tiny shim
    return sorted(list(globals().keys()) + list(_SUBMODULES.keys()) + list(_HELPERS.keys()))


# ----- Optional backend inspection ------------------------------------------

# Map import name -> distribution name on PyPI (when they differ)
_OPT_DEPS: Mapping[str, str] = {
    "rtamt": "rtamt",
    "moonlight": "moonlight",
    # SpaTiaL is published as 'spatial' on PyPI or via GitHub
    "spatial": "spatial",
}


def _probe(mod_name: str) -> tuple[bool, str | None]:
    if _import_util.find_spec(mod_name) is None:
        return False, None
    dist = _OPT_DEPS.get(mod_name) or mod_name
    try:
        ver = _metadata.version(dist)  # type: ignore[arg-type]
    except Exception:
        try:
            m = import_module(mod_name)
            ver = getattr(m, "__version__", None)
        except Exception:
            ver = None
    return True, ver


def available_backends() -> dict[str, dict[str, bool | str | None]]:
    report: dict[str, dict[str, bool | str | None]] = {}
    for name in _OPT_DEPS.keys():
        ok, ver = _probe(name)
        report[name] = {"available": ok, "version": ver}
    return report


# ----- Static imports for type checkers only --------------------------------

if TYPE_CHECKING:  # pragma: no cover
    from . import moonlight_hello as moonlight_hello  # noqa: F401
    from . import moonlight_strel_hello as moonlight_strel_hello  # noqa: F401
    from . import rtamt_hello as rtamt_hello  # noqa: F401
    from . import spatial_demo as spatial_demo  # noqa: F401
    from .moonlight_hello import temporal_hello as temporal_hello  # noqa: F401
    from .moonlight_strel_hello import strel_hello as strel_hello  # noqa: F401
    from .rtamt_hello import stl_hello_offline as stl_hello_offline  # noqa: F401
    from .spatial_demo import run_demo as spatial_run_demo  # noqa: F401
