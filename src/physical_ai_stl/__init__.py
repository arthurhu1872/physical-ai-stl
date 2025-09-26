# ruff: noqa: I001
# isort: skip_file
from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from importlib import metadata as _metadata
from importlib import util as _import_util
from typing import Any, TYPE_CHECKING

__all__ = [
    "__version__",
    # Lazy subpackages/modules
    "datasets",
    "experiments",
    "frameworks",
    "models",
    "monitoring",
    "monitors",
    "physics",
    "training",
    "utils",
    "pde_example",
    # Small re‑exports
    "seed_everything",
    "CSVLogger",
    # Helpers
    "about",
    "optional_dependencies",
]

# NOTE: Keep this as a literal string for Hatch (pyproject.toml -> [tool.hatch.version])
# to source the package version directly from this file.
__version__ = "0.1.0"

# ----- Lazy access to subpackages (PEP 562) ---------------------------------

# Map attribute name -> fully qualified module path
_SUBMODULES: Mapping[str, str] = {
    "datasets": "physical_ai_stl.datasets",
    "experiments": "physical_ai_stl.experiments",
    "frameworks": "physical_ai_stl.frameworks",  # namespace package (no heavy import)
    "models": "physical_ai_stl.models",
    "monitoring": "physical_ai_stl.monitoring",
    "monitors": "physical_ai_stl.monitors",
    "physics": "physical_ai_stl.physics",
    "training": "physical_ai_stl.training",
    "utils": "physical_ai_stl.utils",
    "pde_example": "physical_ai_stl.pde_example",
}

# Lightweight helpers (attribute -> "module:object")
_HELPERS: Mapping[str, str] = {
    "seed_everything": "physical_ai_stl.utils.seed:seed_everything",
    "CSVLogger": "physical_ai_stl.utils.logger:CSVLogger",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - tiny shim
    if name in _SUBMODULES:
        module = import_module(_SUBMODULES[name])
        globals()[name] = module
        return module
    if name in _HELPERS:
        mod_name, obj_name = _HELPERS[name].split(":")
        obj = getattr(import_module(mod_name), obj_name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module 'physical_ai_stl' has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - tiny shim
    return sorted(list(globals().keys()) + list(_SUBMODULES.keys()) + list(_HELPERS.keys()))


# ----- Optional dependency inspection ---------------------------------------

# Map "import name" -> distribution name on PyPI (for version lookups). When the
# names match, we can use a single entry. If unknown, leave the dist empty.
_OPT_DEPS: Mapping[str, str] = {
    # Core scientific stack (often present)
    "numpy": "numpy",
    # Physics‑ML frameworks
    "torch": "torch",
    "neuromancer": "neuromancer",
    "physicsnemo": "nvidia-physicsnemo",  # import name -> PyPI dist
    "torchphysics": "torchphysics",
    # STL / spatio‑temporal monitoring
    "rtamt": "rtamt",
    "moonlight": "moonlight",
    "spatial_spec": "spatial-spec",
}


def _probe_module(mod_name: str) -> tuple[bool, str | None]:
    spec = _import_util.find_spec(mod_name)
    if spec is None:
        return False, None
    dist_name = _OPT_DEPS.get(mod_name) or mod_name
    version = None
    try:
        version = _metadata.version(dist_name)  # type: ignore[arg-type]
    except Exception:
        try:
            mod = import_module(mod_name)
            version = getattr(mod, "__version__", None)
        except Exception:
            version = None
    return True, version


def optional_dependencies() -> dict[str, dict[str, str | bool | None]]:
    report: dict[str, dict[str, str | bool | None]] = {}
    for mod in _OPT_DEPS.keys():
        ok, ver = _probe_module(mod)
        report[mod] = {"available": ok, "version": ver}
    return report


def about() -> str:
    lines = [f"physical_ai_stl {__version__}", "Optional deps:"]
    report = optional_dependencies()
    width = max((len(k) for k in report.keys()), default=0)
    for name in sorted(report.keys()):
        avail = report[name]["available"]
        ver = report[name]["version"]
        lines.append(f"  {name.ljust(width)}  {'yes' if avail else 'no ':<3} ({ver or '-'})")
    return "\n".join(lines)


# ----- Static imports for type checkers only --------------------------------

if TYPE_CHECKING:  # pragma: no cover
    # These imports help IDEs and type checkers without paying runtime import cost.
    from . import (  # noqa: F401
        datasets,
        experiments,
        frameworks,
        models,
        monitoring,
        monitors,
        pde_example,
        physics,
        training,
        utils,
    )
    from .utils.logger import CSVLogger as CSVLogger  # noqa: F401
    from .utils.seed import seed_everything as seed_everything  # noqa: F401
