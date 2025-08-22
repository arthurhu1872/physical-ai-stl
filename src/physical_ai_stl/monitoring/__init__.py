from __future__ import annotations

from importlib import import_module as _import_module
from importlib import util as _import_util
from importlib import metadata as _metadata
from typing import Any, Mapping, TYPE_CHECKING

# ----- Public surface (declared up front for tools/IDEs) ---------------------

__all__ = [
    # Submodules (lazy)
    "moonlight_helper",
    "rtamt_monitor",
    "stl_soft",
    # Backends API
    "get_backend",
    "available_backends",
    "ensure",
    "about",
    # Selected conveniences re‑exported lazily
    "load_script_from_file",
    "get_monitor",
    "build_grid_graph",
    "field_to_signal",
    "evaluate_series",
    "evaluate_multi",
    "satisfied",
    "stl_always_upper_bound",
    "stl_response_within",
    "softmin",
    "softmax",
    "soft_and",
    "soft_or",
    "soft_not",
    "pred_leq",
    "pred_geq",
    "pred_abs_leq",
    "pred_linear_leq",
    "always",
    "eventually",
    "always_window",
    "eventually_window",
    "shift_left",
    "STLPenalty",
]

# Map "logical backend name" -> loader module
_BACKENDS: Mapping[str, str] = {
    "rtamt": "physical_ai_stl.monitoring.rtamt_monitor",
    "moonlight": "physical_ai_stl.monitoring.moonlight_helper",
    "soft": "physical_ai_stl.monitoring.stl_soft",
}

# For *availability* checks we probe the true external distributions.
# Keys here must include any extra runtime deps a backend requires.
_OPT_DEPS: Mapping[str, str] = {
    "rtamt": "rtamt",
    "moonlight": "moonlight",  # Python bindings for MoonLight (Java)
    "torch": "torch",          # required by stl_soft
}

# Submodules we expose directly (lazy)
_SUBMODULES: Mapping[str, str] = {
    "moonlight_helper": _BACKENDS["moonlight"],
    "rtamt_monitor": _BACKENDS["rtamt"],
    "stl_soft": _BACKENDS["soft"],
}

# Friendly, dependency‑aware re‑exports (name -> "module:object")
_REEXPORTS: Mapping[str, str] = {
    # MoonLight helpers
    "load_script_from_file": f'{_BACKENDS["moonlight"]}:load_script_from_file',
    "get_monitor":           f'{_BACKENDS["moonlight"]}:get_monitor',
    "build_grid_graph":      f'{_BACKENDS["moonlight"]}:build_grid_graph',
    "field_to_signal":       f'{_BACKENDS["moonlight"]}:field_to_signal',
    # RTAMT helpers
    "evaluate_series":       f'{_BACKENDS["rtamt"]}:evaluate_series',
    "evaluate_multi":        f'{_BACKENDS["rtamt"]}:evaluate_multi',
    "satisfied":             f'{_BACKENDS["rtamt"]}:satisfied',
    "stl_always_upper_bound":  f'{_BACKENDS["rtamt"]}:stl_always_upper_bound',
    "stl_response_within":     f'{_BACKENDS["rtamt"]}:stl_response_within',
    # Soft STL (PyTorch)
    "softmin":               f'{_BACKENDS["soft"]}:softmin',
    "softmax":               f'{_BACKENDS["soft"]}:softmax',
    "soft_and":              f'{_BACKENDS["soft"]}:soft_and',
    "soft_or":               f'{_BACKENDS["soft"]}:soft_or',
    "soft_not":              f'{_BACKENDS["soft"]}:soft_not',
    "pred_leq":              f'{_BACKENDS["soft"]}:pred_leq',
    "pred_geq":              f'{_BACKENDS["soft"]}:pred_geq',
    "pred_abs_leq":          f'{_BACKENDS["soft"]}:pred_abs_leq',
    "pred_linear_leq":       f'{_BACKENDS["soft"]}:pred_linear_leq',
    "always":                f'{_BACKENDS["soft"]}:always',
    "eventually":            f'{_BACKENDS["soft"]}:eventually',
    "always_window":         f'{_BACKENDS["soft"]}:always_window',
    "eventually_window":     f'{_BACKENDS["soft"]}:eventually_window',
    "shift_left":            f'{_BACKENDS["soft"]}:shift_left',
    "STLPenalty":            f'{_BACKENDS["soft"]}:STLPenalty',
}

# ----- Lightweight backend/dep probing --------------------------------------

def _probe(mod_name: str) -> tuple[bool, str | None]:
    if _import_util.find_spec(mod_name) is None:
        return False, None
    dist = _OPT_DEPS.get(mod_name) or mod_name
    try:
        ver = _metadata.version(dist)  # type: ignore[arg-type]
    except Exception:
        try:
            m = _import_module(mod_name)
            ver = getattr(m, "__version__", None)
        except Exception:
            ver = None
    return True, ver

def available_backends() -> dict[str, dict[str, bool | str | None]]:
    report: dict[str, dict[str, bool | str | None]] = {}
    for name in ("rtamt", "moonlight"):
        ok, ver = _probe(name)
        report[name] = {"available": ok, "version": ver}
    ok, ver = _probe("torch")
    report["soft"] = {"available": ok, "version": ver}
    return report

def ensure(*backends: str) -> None:
    missing: list[str] = []
    versions: dict[str, str | None] = {}
    for b in backends:
        key = "torch" if b == "soft" else b
        ok, ver = _probe(key)
        versions[b] = ver
        if not ok:
            missing.append(b)
    if missing:
        # Helpful hints per backend
        hints: dict[str, str] = {
            "rtamt": (
                "Install with:  pip install rtamt\n"
                "Docs: https://arxiv.org/abs/2005.11827"
            ),
            "moonlight": (
                "Install Python bindings and ensure a Java runtime is available:\n"
                "  pip install moonlight\n"
                "Paper: https://link.springer.com/article/10.1007/s10009-023-00710-5"
            ),
            "soft": (
                "Install PyTorch for differentiable STL:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cpu  # (CPU wheel)"
            ),
        }
        msg = ["Missing backends: " + ", ".join(missing)]
        for b in missing:
            msg.append(f"- {b}: " + hints.get(b, "please install the required package(s)."))
        raise ImportError("\n".join(msg))

def get_backend(name: str) -> Any:
    key = name.lower().strip()
    if key not in _BACKENDS:
        raise KeyError(f"Unknown backend {name!r}. Expected one of: {', '.join(_BACKENDS)}.")
    # Quick dependency check before import (keeps errors helpful)
    ensure("soft" if key == "soft" else key)
    mod = _import_module(_BACKENDS[key])
    return mod

def about() -> str:
    rep = available_backends()
    width = max(len(k) for k in rep) if rep else 7
    lines = ["physical_ai_stl.monitoring backends:"]
    for name in ("rtamt", "moonlight", "soft"):
        avail = rep[name]["available"]
        ver = rep[name]["version"]
        lines.append(f"  {name.ljust(width)}  {'yes' if avail else 'no ':<3} ({ver or '-'})")
    return "\n".join(lines)

# ----- Lazy attribute access & dir() -----------------------------------------

def __getattr__(name: str) -> Any:  # pragma: no cover - tiny shim
    if name in _SUBMODULES:
        mod = _import_module(_SUBMODULES[name])
        globals()[name] = mod
        return mod
    if name in _REEXPORTS:
        mod_name, obj_name = _REEXPORTS[name].split(":")
        obj = getattr(_import_module(mod_name), obj_name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module 'physical_ai_stl.monitoring' has no attribute {name!r}")

def __dir__() -> list[str]:  # pragma: no cover - tiny shim
    return sorted(list(globals().keys()) + list(_SUBMODULES.keys()) + list(_REEXPORTS.keys()))

# ----- Static imports for type checkers only ---------------------------------

if TYPE_CHECKING:  # pragma: no cover
    # Submodules
    from . import moonlight_helper as moonlight_helper  # noqa: F401
    from . import rtamt_monitor as rtamt_monitor        # noqa: F401
    from . import stl_soft as stl_soft                  # noqa: F401
    # Re‑exports
    from .moonlight_helper import (                     # noqa: F401
        load_script_from_file as load_script_from_file,
        get_monitor as get_monitor,
        build_grid_graph as build_grid_graph,
        field_to_signal as field_to_signal,
    )
    from .rtamt_monitor import (                        # noqa: F401
        evaluate_series as evaluate_series,
        evaluate_multi as evaluate_multi,
        satisfied as satisfied,
        stl_always_upper_bound as stl_always_upper_bound,
        stl_response_within as stl_response_within,
    )
    from .stl_soft import (                             # noqa: F401
        softmin as softmin,
        softmax as softmax,
        soft_and as soft_and,
        soft_or as soft_or,
        soft_not as soft_not,
        pred_leq as pred_leq,
        pred_geq as pred_geq,
        pred_abs_leq as pred_abs_leq,
        pred_linear_leq as pred_linear_leq,
        always as always,
        eventually as eventually,
        always_window as always_window,
        eventually_window as eventually_window,
        shift_left as shift_left,
        STLPenalty as STLPenalty,
    )
