"""SpaTiaL minimal demo: specify a spatial-temporal formula and evaluate it on a toy log."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

try:
    # SpaTiaL uses module name 'spatial' on PyPI
    from spatial.logic import Spatial  # type: ignore
    from spatial.geometry import Circle, Polygon, StaticObject, DynamicObject  # type: ignore
except Exception as e:  # pragma: no cover - optional dependency path
    Spatial = None  # type: ignore

@dataclass
class ToyScene:
    T: int = 50
    workspace_half: float = 5.0
    r: float = 0.5  # disc radius

def _rect_polygon(half: float) -> Polygon:
    # rectangle centered at origin with half-extent 'half'
    verts = np.array([
        [-half, -half],
        [ half, -half],
        [ half,  half],
        [-half,  half],
    ], dtype=float)
    # SpaTiaL Polygon expects vertices in an np.ndarray (will convex-hull itself)
    from spatial.geometry import Polygon as _Poly  # type: ignore
    return _Poly(verts)

def build_scene(cfg: ToyScene):
    if Spatial is None:  # pragma: no cover
        raise ImportError("SpaTiaL (package 'spatial') not installed; pip install spatial-spec")
    A = DynamicObject()
    B = DynamicObject()
    # A moves on x from -3 to +3; B moves parallel but shifted right
    xs = np.linspace(-3.0, 3.0, cfg.T)
    for t, x in enumerate(xs):
        A.addObject(Circle(np.array([x, 0.0]), cfg.r), time=t)
        B.addObject(Circle(np.array([x + 1.0, 0.0]), cfg.r), time=t)
    workspace = StaticObject(_rect_polygon(cfg.workspace_half))
    return A, B, workspace

def evaluate_formula(cfg: ToyScene) -> float:
    """Return quantitative robustness (>=0 satisfied) of the SpaTiaL formula."""
    if Spatial is None:  # pragma: no cover
        raise ImportError("SpaTiaL (package 'spatial') not installed; pip install spatial-spec")
    A, B, W = build_scene(cfg)
    S = Spatial(quantitative=True)
    # Assign variables used in the formula string
    S.assign_variable("A", A)
    S.assign_variable("B", B)
    S.assign_variable("Workspace", W)
    # Note: SpaTiaL exposes a grammar via 'Spatial.parse'. The following relies on the
    # documented relation names in spatial.logic (left_of, enclosed_in) and bounded 'always'.
    formula = S.parse("always[0,{T}] spatial( left_of(A, B) and enclosed_in(A, Workspace) )".format(T=cfg.T-1))  # type: ignore
    val = S.interpret(formula, lower=0, upper=cfg.T-1)  # quantitative robustness
    # Cast to float for JSON-serializable return
    try:
        return float(val)
    except Exception:
        # If SpaTiaL returns numpy scalar
        return float(np.asarray(val).item())

def run_demo(T: int = 50) -> float:
    return evaluate_formula(ToyScene(T=T))

__all__ = ["ToyScene", "build_scene", "evaluate_formula", "run_demo"]
