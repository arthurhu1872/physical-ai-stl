# Sprint 2 Report

## Progress Overview
In this sprint, we integrated the **TorchPhysics** framework to solve a more challenging PDE (the 1D Burgers' equation) and applied a signal temporal logic (STL) penalty during training... 

## TorchPhysics Integration (1D Burgers' Equation)
We chose the viscous Burgers' equation as a testbed, since it introduces nonlinearity (convection) beyond the simpler diffusion/heat examples... Using TorchPhysics, we defined a fully-connected PINN and set up training **conditions** for the PDE residual, initial condition, and boundaries... The framework’s modular conditions simplified the PINN setup...

## STL Penalty in Training
To incorporate STL, we added an optional training condition representing the specification **G (mean_x u ≤ 1.0)** (i.e., the spatial average of u should never exceed a safety threshold 1.0). In practice, we approximated this by penalizing any point in the domain where u(x,t) > 1.0... When the STL penalty is active, the network **learns to reduce the peak of u(x,t)** to satisfy the threshold... the STL-penalized model successfully remained within the safety bound (no overshoot of 1.0)...

## Other Tools Considered
- **Neuromancer:** ... its constraint API could express STL-like requirements elegantly. However, integrating it would require significant refactoring, so we deferred it for now.
- **SpaTiaL:** ... we decided not to integrate SpaTiaL at this stage because MoonLight already covers our needs for spatio-temporal monitoring.

## Next Steps
- **Complex STL Specifications:** ... e.g., "if a hotspot occurs, it cools down within τ time" during training.
- **Extended Examples:** ... apply logic-enforced PINNs to additional PDEs (Burgers', Navier–Stokes, etc.) to evaluate scalability...
- **Comparison and Evaluation:** ... quantify the benefit of STL guidance by comparing spec satisfaction and accuracy with vs. without STL penalties, using RTAMT/MoonLight to measure robustness.
