from __future__ import annotations

from collections.abc import Callable
from importlib import import_module, util as _import_util
from typing import Any


def _import_neuromancer() -> Any:
    try:
        return import_module("neuromancer")
    except Exception as e:  # pragma: no cover
        raise ImportError("neuromancer not installed") from e


def _resolve(nm: Any, dotted_alternatives: tuple[tuple[str, ...], ...]) -> Any:
    for parts in dotted_alternatives:
        obj: Any = nm
        ok = True
        for name in parts:
            obj = getattr(obj, name, None)
            if obj is None:
                ok = False
                break
        if ok:
            return obj
    # Not found: craft a helpful error message.
    alts = " | ".join(".".join(("nm",) + p) for p in dotted_alternatives)
    raise AttributeError(f"Could not resolve any of: {alts}")


def neuromancer_version() -> str:
    nm = _import_neuromancer()
    ver = getattr(nm, "__version__", None)
    return ver if isinstance(ver, str) and ver else "unknown"


def neuromancer_available() -> bool:
    spec = _import_util.find_spec("neuromancer")
    return spec is not None


def neuromancer_smoke(batch_size: int = 4) -> dict[str, float | str]:
    nm = _import_neuromancer()
    try:
        import torch  # defer heavy import
    except Exception as e:  # pragma: no cover
        raise ImportError("torch is required for neuromancer_smoke") from e

    # Resolve core entry points with version tolerance.
    variable: Callable[[str], Any] = _resolve(
        nm, (("constraint", "variable"), ("constraints", "variable"), ("variable",))
    )
    Node = _resolve(nm, (("system", "Node"), ("Node",),))
    PenaltyLoss = _resolve(nm, (("loss", "PenaltyLoss"), ("PenaltyLoss",),))
    Problem = _resolve(nm, (("problem", "Problem"), ("Problem",),))

    # Identity map p -> x, wrapped as a Node
    class _Id(torch.nn.Module):
        def forward(self, p: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return p

    node = Node(_Id(), ["p"], ["x"], name="id_map")

    # Symbolic variable and simple convex objective: min (x - 0.5)^2
    x = variable("x")
    obj = ((x - 0.5) ** 2).minimize(weight=1.0)

    # Assemble loss and problem
    loss = PenaltyLoss(objectives=[obj], constraints=[])
    problem = Problem(nodes=[node], loss=loss)

    # Dummy batch on CPU
    p = torch.ones(batch_size, 1, dtype=torch.float32)
    batch = {"p": p}

    # Evaluate loss in a version-robust way
    with torch.no_grad():
        out = problem(batch)
    if isinstance(out, dict) and "loss" in out:
        loss_tensor = out["loss"]
    elif isinstance(out, torch.Tensor):
        loss_tensor = out
    else:  # fall back to explicit API
        loss_tensor = problem.compute_loss(batch)  # type: ignore[attr-defined]

    loss_value = float(loss_tensor.detach().cpu().item())
    return {"version": neuromancer_version(), "loss": loss_value, "samples": float(batch_size)}


__all__ = ["neuromancer_version", "neuromancer_available", "neuromancer_smoke"]
