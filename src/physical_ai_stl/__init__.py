"""Top-level package for the physical-ai-stl project.

This package contains modules for monitoring signals using Signal Temporal
Logic (STL) and spatio-temporal logic (STREL) as well as training simple
machine-learning models with logic-based constraints. It is intended as
 a teaching/prototyping framework and not a polished library.

Subpackages:
  data: Data generation utilities and dataset definitions.
  models: Implementations of physics-informed neural networks and other
          architectures that support logic-based training.
  specs: Definitions of differentiable STL operators and utilities for
         interfacing with RTAMT and MoonLight.
"""

# Expose important classes and functions at the package level.
# These imports allow users to write `import physical_ai_stl as stl`
# and access key functionality directly.

from .specs.smooth_stl import (
    smooth_max,
    smooth_min,
    robustness_bound,
    stl_always,
    stl_eventually,
    stl_and,
    stl_or,
    penalty_from_robustness,
)

from .specs.rtamt_utils import compile_stl, evaluate_stl
from .specs.moonlight_utils import compile_strel, evaluate_strel

from .data.diffusion_data import DiffusionDataset
from .models.torchphysics_pinn import DiffusionNet, train_diffusion_nn

__all__ = [
    # Smooth STL operators
    "smooth_max",
    "smooth_min",
    "robustness_bound",
    "stl_always",
    "stl_eventually",
    "stl_and",
    "stl_or",
    "penalty_from_robustness",
    # RTAMT/MoonLight utilities
    "compile_stl",
    "evaluate_stl",
    "compile_strel",
    "evaluate_strel",
    # Data sets
    "DiffusionDataset",
    # Models and training
    "DiffusionNet",
    "train_diffusion_nn",
]
