"""SpaTiaL minimal demo: specify a spatial-temporal formula and evaluate it on a toy log."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    # SpaTiaL uses module name 'spatial' on PyPI
    from spatial.logic import Spatial  # type: ignore
    from spatial.geometry import Circle, Polygon, StaticObject, DynamicObject  # type: ignore
except Exception:  # pragma: no cover - optional dependency path
    Spatial = None  # type: ignore


@dataclass
class ToyScene:
    T: int = 50


def build_scene(T: int = 50) -> ToyScene:
    return ToyScene(T=T)


def evaluate_formula(cfg: ToyScene) -> float:
    if Spatial is None:
        raise RuntimeError("SpaTiaL not available")
    # Placeholders: return a dummy robustness value
    return float(0.0)


def run_demo(T: int = 50) -> float:
    return evaluate_formula(ToyScene(T=T))


__all__ = ["ToyScene", "build_scene", "evaluate_formula", "run_demo"]
