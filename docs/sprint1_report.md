# Sprint 1 Report

## Overview

This sprint adds a minimal partial differential equation (PDE) example to the repository and begins exploring potential frameworks for future experiments.  The goals were to:
- Provide a simple 1D diffusion solver that can be used as a stand‑alone example for testing signal temporal logic (STL) monitoring.
- Compute basic robustness values for one‑dimensional and two‑dimensional signals without relying on external STL libraries.
- Compare a few candidate physics‑ML frameworks to decide which one to adopt for subsequent sprints.

The new module `src/physical_ai_stl/pde_example.py` contains:
- `simulate_diffusion`: an explicit finite–difference solver for the heat equation in 1D with Neumann boundary conditions.
- `simulate_diffusion_with_clipping`: wraps the solver and clips the state after each time step to enforce hard safety bounds.
- `compute_robustness`: returns the minimum distance from a 1D signal to user‑supplied lower/upper bounds.
- `compute_spatiotemporal_robustness`: returns the minimum distance from any entry of a 2D signal matrix to the bounds.

Accompanying tests in `tests/test_pde_example.py` verify the solver’s shape, the clipping behaviour, and the robustness functions.

## Framework evaluation

Although we implemented our own solver for this sprint, the long‑term plan is to leverage existing physics‑ML libraries that support neural ODEs/PDEs and physics–informed neural networks (PINNs).  Based on the instructor’s suggestions and preliminary reading, the following frameworks were considered:

### Neuromancer

Neuromancer is a PyTorch library for differentiable programming developed by PNNL.  It provides a high‑level API for formulating neural ODEs/PDEs, PINNs and constrained optimization problems.  Key features include:

- **Symbolic constraints** and automatic penalty generation for enforcing algebraic, time‑domain or state constraints.
- Support for **neural ODEs** and hybrid models via modular blocks.
- Tools for **differentiable predictive control** and parametric optimization problems.

Neuromancer seems attractive for projects where constraints and physical laws must be encoded explicitly in the loss function.  It is pure Python, relatively easy to install and integrates smoothly with PyTorch.  Its constraint API may make it simpler to implement STL‑based penalties in later sprints.  On the downside, documentation is still evolving and its community is smaller than that of more established libraries.

### TorchPhysics

TorchPhysics is a lightweight collection of components for building PINNs in PyTorch.  Highlights include:

- Ready‑made examples for solving PDEs (e.g., Poisson, Burgers, wave equations) with configurable boundary and initial conditions.
- Modular design: domains, constraints, losses and neural operators can be assembled with minimal boilerplate.
- Built‑in support for Fourier neural operators (FNO) and DeepONets.

TorchPhysics is well suited for quick prototyping and experimenting with different network architectures.  The library has a small dependency footprint and does not enforce a rigid training loop, making it easy to integrate custom logic.  However, it lacks a high‑level API for expressing generic constraints and does not provide built‑in STL or spatio‑temporal logic monitoring; these would need to be added manually.

### PhysicsNeMo (NVIDIA Modulus/NeMo)

PhysicsNeMo (formerly NVIDIA Modulus) is a large‑scale framework combining PINNs, neural operators and high‑performance kernels.  Important capabilities are:

- A rich suite of PDE examples ranging from diffusion–reaction equations to Navier‑Stokes and Darcy flow.
- Built‑in mesh and geometry generation, automatic differentiation and GPU acceleration.
- Support for neural operators such as FNO, DeepONet and U‑Nets.

PhysicsNeMo is powerful and production‑oriented but heavy‑weight.  It requires CUDA and a relatively complex installation.  It may be overkill for an undergraduate project focused on exploring STL monitoring, especially if only small 1D or 2D problems are needed.  Nevertheless, it could be useful later if we tackle more demanding PDE benchmarks.

## Choice for future sprints

For the next sprint we plan to experiment with **TorchPhysics**, since it offers a balance between ease of use and flexibility.  We can start by reproducing one of its simple PDE examples (e.g. the 1D Burgers equation) and then add our own STL robustness penalties.  If we find that expressing constraints is cumbersome, we will revisit **Neuromancer**.  We will leave **PhysicsNeMo** for potential future work if we require GPU acceleration or neural operators.

## What’s next

- Integrate the PDE solver with our existing MoonLight monitors to evaluate spatio‑temporal logic properties on small grids.
- Reproduce a TorchPhysics example and implement a robustness penalty similar to the one used in our diffusion example.
- Continue refining the robustness functions, adding softmin/softmax approximations to make them differentiable for training.
