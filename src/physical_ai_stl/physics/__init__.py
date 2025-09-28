# src/physical_ai_stl/physics/__init__.py
from __future__ import annotations

import importlib
from typing import Any, TYPE_CHECKING

# ----- Lazily exposed submodules ------------------------------------------------

_LAZY_MODULES: dict[str, str] = {
    "diffusion1d": "physical_ai_stl.physics.diffusion1d",
    "heat2d": "physical_ai_stl.physics.heat2d",
}

# Forwarded attributes: provide nice `from physical_ai_stl.physics import pde_residual`.
# These are fetched from their submodules on first access and then cached here.
_FORWARD_ATTRS: dict[str, tuple[str, str]] = {
    # 1‑D diffusion
    "pde_residual": ("diffusion1d", "pde_residual"),
    "residual_loss": ("diffusion1d", "residual_loss"),
    "boundary_loss": ("diffusion1d", "boundary_loss"),
    "Interval1D": ("diffusion1d", "Interval1D"),
    "sine_ic": ("diffusion1d", "sine_ic"),
    "sine_solution": ("diffusion1d", "sine_solution"),
    "make_dirichlet_mask_1d": ("diffusion1d", "make_dirichlet_mask_1d"),
    # 2‑D heat
    "residual_heat2d": ("heat2d", "residual_heat2d"),
    "bc_ic_heat2d": ("heat2d", "bc_ic_heat2d"),
    "SquareDomain2D": ("heat2d", "SquareDomain2D"),
    "gaussian_ic": ("heat2d", "gaussian_ic"),
    "make_dirichlet_mask": ("heat2d", "make_dirichlet_mask"),
}

# What `from physical_ai_stl.physics import *` exposes.
__all__ = sorted({*list(_LAZY_MODULES.keys()), *list(_FORWARD_ATTRS.keys())})


def _import_module(mod_path: str) -> Any:  # pragma: no cover - trivial wrapper
    try:
        return importlib.import_module(mod_path)
    except ImportError as e:
        # Offer a friendlier nudge if the failure was due to missing torch.
        msg = str(e).lower()
        if "torch" in msg or "pytorch" in msg:
            raise ImportError(
                "The physics helpers require PyTorch. "
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
    # Expose both already‑bound globals and the lazy attributes
    return sorted(list(globals().keys()) + list(__all__))


# ----- Static imports for IDEs / type checkers only ----------------------------

if TYPE_CHECKING:  # pragma: no cover
    from . import diffusion1d as diffusion1d  # noqa: F401
    from . import heat2d as heat2d  # noqa: F401
    from .diffusion1d import (  # noqa: F401
        boundary_loss,
        Interval1D,
        make_dirichlet_mask_1d,
        pde_residual,
        residual_loss,
        sine_ic,
        sine_solution,
    )
    from .heat2d import (  # noqa: F401
        bc_ic_heat2d,
        gaussian_ic,
        make_dirichlet_mask,
        residual_heat2d,
        SquareDomain2D,
    )
