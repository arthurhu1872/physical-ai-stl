from __future__ import annotations

from dataclasses import dataclass

try:
    # SpaTiaL is published on PyPI under the module name "spatial"
    # We import lazily inside functions so the package remains importable
    # when SpaTiaL isn't installed (e.g., in minimal CI).
    from spatial.logic import Spatial  # type: ignore
except Exception:  # pragma: no cover - optional dependency path
    Spatial = None  # type: ignore


@dataclass(slots=True)
class ToyScene:
    T: int = 50
    agent_speed: float = 0.35
    agent_radius: float = 0.30
    goal_pos: tuple[float, float] = (12.0, 0.0)
    goal_radius: float = 0.40
    reach_eps: float = 0.0  # 0.0 means "touching or overlapping"


def build_scene(T: int = 50) -> ToyScene:
    return ToyScene(T=T)


def _build_spatial_objects(cfg: ToyScene):
    # Import here to avoid import errors when SpaTiaL isn't available.
    try:
        import numpy as np
        from spatial.geometry import Circle, DynamicObject, PolygonCollection, StaticObject  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "SpaTiaL not available – install with: `pip install spatial shapely lark-parser`"
        ) from e

    agent = DynamicObject()
    # Build a single-circle collection per time step; this implements SpatialInterface.
    for t in range(cfg.T):
        center = np.array([t * cfg.agent_speed, 0.0], dtype=float)
        footprint = PolygonCollection({Circle(center, cfg.agent_radius)})
        agent.addObject(footprint, time=t)

    goal_center = np.array(cfg.goal_pos, dtype=float)
    goal_shape = PolygonCollection({Circle(goal_center, cfg.goal_radius)})
    goal = StaticObject(goal_shape)

    return {"agent": agent, "goal": goal}


def _first_parsed(sp: Spatial, candidates: list[str]):
    for s in candidates:
        try:
            tree = sp.parse(s)
            if tree is not None:
                return s, tree
        except Exception:
            # Try the next spelling/variant
            pass
    # If we get here, re-raise a helpful message showing the attempted spellings.
    raise ValueError(
        "Could not parse any SpaTiaL formula; tried variants:\n" + "\n".join(candidates)
    )


def evaluate_formula(cfg: ToyScene) -> float:
    if Spatial is None:
        raise RuntimeError(
            "SpaTiaL not available – install with: `pip install spatial shapely lark-parser`"
        )

    # Build variables for SpaTiaL
    vars_map = _build_spatial_objects(cfg)

    # Create a quantitative interpreter
    sp = Spatial(quantitative=True)  # type: ignore[arg-type]

    # Register variables with both internal interpreters for maximum compatibility
    for name, obj in vars_map.items():
        sp.assign_variable(name, obj)
    # Also update the temporal interpreter's view of variables
    try:
        sp.update_variables(vars_map)  # present on SpaTiaL ≥ 0.2
    except Exception:
        pass

    upper = cfg.T - 1
    eps = float(cfg.reach_eps)

    # A small set of grammar variants (bounded eventually, distance compare / touching).
    # We order by *most precise quantitative semantics* first.
    candidates = [
        f"eventually [0, {upper}] ( distance(agent, goal) <= {eps} )",
        f"F[0, {upper}] ( distance(agent, goal) <= {eps} )",
        f"eventually[0,{upper}](distance(agent, goal) <= {eps})",
        f"F[0,{upper}](distance(agent, goal) <= {eps})",
        # Touching/close_to fallbacks (if `distance` comparison syntax differs)
        f"eventually [0, {upper}] ( touching(agent, goal) )",
        f"F[0, {upper}] ( touching(agent, goal) )",
        f"eventually [0, {upper}] ( close_to(agent, goal) )",
        f"F[0, {upper}] ( close_to(agent, goal) )",
        # CamelCase variants of spatial relations (some papers use these spellings)
        f"eventually [0, {upper}] ( leftOf(agent, goal) )",
        f"eventually [0, {upper}] ( touching(agent, goal) )",
    ]

    formula_str, tree = _first_parsed(sp, candidates)

    # Evaluate over the trace [0, upper]
    val = sp.interpret(tree, lower=0, upper=upper)  # type: ignore[arg-type]
    try:
        return float(val)  # numpy scalar or python float
    except Exception as e:  # pragma: no cover - very unlikely
        raise RuntimeError(
            f"Unexpected SpaTiaL return type: {type(val)} from formula: {formula_str}"
        ) from e


def run_demo(T: int = 50) -> float:
    return evaluate_formula(ToyScene(T=T))


__all__ = ["ToyScene", "build_scene", "evaluate_formula", "run_demo"]
