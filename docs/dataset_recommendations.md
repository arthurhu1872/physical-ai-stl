# Candidate Datasets and Problem Spaces

This note lists example domains suitable for combining physics-informed models with Signal Temporal Logic (STL) monitoring. They align with the `physical AI` direction.
## Air quality time series
- **Source:** UCI Air Quality dataset and synthetic versions used in [STLnet](https://github.com/meiyima/STLnet).
- **STL angle:** Expressions over pollutant thresholds (e.g., `PM2.5 must remain below a bound within 1 hour after a spike`).

## Diffusion and heat PDEs
- **Source:** canonical 1D/2D diffusion equations (already implemented in `pde_example.py`).
- **STL angle:** Ensure temperatures stay within safe limits everywhere and for all times; test spatial containment of hotspots.

## Burgers' equation / fluid dynamics
- **Source:** sample problems bundled with [PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) or [TorchPhysics](https://github.com/boschresearch/torchphysics).
- **STL angle:** Monitor shock formation or velocity bounds with STL or spatio-temporal logic.

## Traffic or crowd models
- **Source:** public datasets such as the [NGSIM traffic trajectories](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm).
- **STL angle:** Specify safety distances or throughput targets using STL formulas.

These candidates provide a range of continuous and discrete dynamics for experimenting with temporal and spatio-temporal specifications.
