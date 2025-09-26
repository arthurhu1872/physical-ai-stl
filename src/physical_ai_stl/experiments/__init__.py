from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module as _import_module, util as _import_util
from typing import Any, Protocol, TYPE_CHECKING

# ---------------------------------------------------------------------------
# Public surface (declared up front for tools/IDEs)
# ---------------------------------------------------------------------------

# Submodules exposed directly (lazy)
__all__ = ['diffusion1d', 'heat2d']

# Extra conveniences (available via attribute access; not listed in __all__ to
# keep ``from physical_ai_stl.experiments import *`` focused on submodules).
# IDEs will still see these thanks to the TYPE_CHECKING block at the bottom.
# - names()      : list available experiment keys
# - available()  : quick availability probe
# - get_runner() : fetch the callable for a key
# - run()        : dispatch by key
# - register()   : add a new entry at runtime
# - about()      : compact human‑readable summary

# ---------------------------------------------------------------------------
# Registry (name → "module:function")
# ---------------------------------------------------------------------------

_EXPERIMENTS: dict[str, str | Callable[..., Any]] = {
    'diffusion1d': 'physical_ai_stl.experiments.diffusion1d:run_diffusion1d',
    'heat2d':      'physical_ai_stl.experiments.heat2d:run_heat2d',
}

# One‑line blurbs for docs/printing without importing heavy modules
_DOCS: dict[str, str] = {
    'diffusion1d': '1‑D diffusion (heat) PINN with optional STL penalty.',
    'heat2d':      '2‑D heat‑equation PINN with optional STL/STREL penalty.',
}

# Submodules that we expose attributes from on demand
_LAZY_MODULES: dict[str, str] = {
    'diffusion1d': 'physical_ai_stl.experiments.diffusion1d',
    'heat2d':      'physical_ai_stl.experiments.heat2d',
}

# Forwarded names for nice imports like:
#   from physical_ai_stl.experiments import run_diffusion1d
_FORWARD_ATTRS: dict[str, tuple[str, str]] = {
    'run_diffusion1d': ('diffusion1d', 'run_diffusion1d'),
    'Diffusion1DConfig': ('diffusion1d', 'Diffusion1DConfig'),
    'run_heat2d': ('heat2d', 'run_heat2d'),
    'Heat2DConfig': ('heat2d', 'Heat2DConfig'),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_torch() -> bool:
    return _import_util.find_spec('torch') is not None


def _import_with_friendly_error(mod_path: str):
    try:
        return _import_module(mod_path)
    except ImportError as e:  # pragma: no cover - tiny UX shim
        msg = str(e).lower()
        if 'torch' in msg or 'pytorch' in msg or (_import_util.find_spec('torch') is None):
            raise ImportError(
                "This experiment requires PyTorch. Install the optional extra:\n"
                '    pip install "physical-ai-stl[torch]"\n'
                "or use the provided requirements file:\n"
                "    pip install -r requirements-extra.txt"
            ) from e
        raise


def _split_target(target: str) -> tuple[str, str]:
    if ':' not in target:
        raise ValueError(f"Expected 'module:function' but got {target!r}.")
    mod, func = target.split(':', 1)
    return mod, func


class Runner(Protocol):
    def __call__(self, cfg: Mapping[str, Any] | dict[str, Any]) -> Any:  # pragma: no cover - typing only
        ...


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def names() -> list[str]:
    return sorted(_EXPERIMENTS.keys())


def available() -> dict[str, bool]:
    has_torch = _has_torch()
    return {name: has_torch for name in _EXPERIMENTS.keys()}


def register(name: str, target: str | Callable[..., Any]) -> None:
    _EXPERIMENTS[name] = target


def get_runner(name: str) -> Runner:
    key = name.lower().strip()
    if key not in _EXPERIMENTS:
        opts = ', '.join(sorted(_EXPERIMENTS))
        raise KeyError(f"Unknown experiment {name!r}. Available: {opts}.")
    target = _EXPERIMENTS[key]
    if callable(target):
        fn = target  # type: ignore[assignment]
    else:
        mod_path, func_name = _split_target(target)
        mod = _import_with_friendly_error(mod_path)
        try:
            fn = getattr(mod, func_name)
        except AttributeError as e:  # pragma: no cover - rare
            raise AttributeError(f"Module {mod_path!r} has no attribute {func_name!r}.") from e
    if not callable(fn):
        raise TypeError(f"Registered target for {name!r} is not callable: {fn!r}")
    return fn  # type: ignore[return-value]


def run(name: str, cfg: Mapping[str, Any] | dict[str, Any]) -> Any:
    fn = get_runner(name)
    return fn(cfg)


def about() -> str:
    width = max((len(n) for n in _EXPERIMENTS), default=0)
    avail = available()
    lines = ['experiments:',]
    for n in names():
        tag = 'yes' if avail.get(n, False) else 'no '
        blurb = _DOCS.get(n, '')
        lines.append(f"  {n.ljust(width)}  {tag}  {blurb}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Lazy attribute access (PEP 562)
# ---------------------------------------------------------------------------

def __getattr__(name: str) -> Any:  # pragma: no cover - tiny shim
    if name in _LAZY_MODULES:
        mod = _import_with_friendly_error(_LAZY_MODULES[name])
        globals()[name] = mod
        return mod
    if name in _FORWARD_ATTRS:
        mod_key, obj_name = _FORWARD_ATTRS[name]
        mod = _import_with_friendly_error(_LAZY_MODULES[mod_key])
        obj = getattr(mod, obj_name)
        globals()[name] = obj
        return obj
    if name in {'names', 'available', 'get_runner', 'run', 'register', 'about'}:
        return globals()[name]
    raise AttributeError(f"module 'physical_ai_stl.experiments' has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - tiny shim
    base = set(globals().keys())
    return sorted(base | set(__all__) | set(_FORWARD_ATTRS.keys()) | set(_LAZY_MODULES.keys())
                  | {'names', 'available', 'get_runner', 'run', 'register', 'about'})


# ---------------------------------------------------------------------------
# Static imports for type checkers only (avoid runtime import cost)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:  # pragma: no cover
    from . import diffusion1d as diffusion1d
    from . import heat2d as heat2d
    from .diffusion1d import Diffusion1DConfig, run_diffusion1d
    from .heat2d import Heat2DConfig, run_heat2d
