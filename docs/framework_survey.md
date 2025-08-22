# Framework Survey

This document summarizes candidate libraries for physics-based machine learning and Signal Temporal Logic (STL) monitoring.

## Physical AI frameworks

| Library | Focus |
| --- | --- |
| **Neuromancer** | Differentiable programming library for constrained optimization and physics-informed system ID. Built on PyTorch. |
| **NVIDIA PhysicsNeMo** | NVIDIA's deep-learning framework for building and training physics-based models, successor to Modulus. |
| **TorchPhysics** | Bosch Research library for mesh-free physics-informed learning including PINNs, Deep Ritz, FNOs, and more. |

## STL and STREL tooling

| Library | Notes |
| --- | --- |
| **RTAMT** | Python runtime monitoring of Signal Temporal Logic. |
| **MoonLight** | Java-based monitoring of temporal and spatio-temporal specs (STREL). Python interface available. |
| **SpaTiaL ('spatial-spec')** | Specify spatial and temporal relations for planning tasks. |

## Suggested datasets/problem spaces

* 1D diffusion and 2D heat equations (already included here) for testing STL enforcement.
* Air quality or synthetic trajectory datasets from STLnet for spatio-temporal experiments.
* Additional PDE benchmarks from PhysicsNeMo or TorchPhysics (e.g., Burgers', Navier–Stokes) once STL monitoring is integrated.

These notes provide a starting point for integrating STL monitoring with physics-informed learning libraries.
