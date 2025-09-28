# Report Outline — Physical AI + STL
(Living document; keep updated as results come in.)

## Abstract
One-paragraph summary of problem, method (STL/STREL integration with physics-ML), and key results.

## 1. Introduction
Motivation (safety in AI-driven CPS), problem statement, contributions.

## 2. Background
Signal Temporal Logic (STL) and STREL; monitoring (RTAMT, MoonLight, SpaTiaL); physical AI frameworks (Neuromancer, PhysicsNeMo, TorchPhysics).

## 3. Related Work
Prior STL‑guided training (e.g., STLnet) and verification in CPS (NN controllers, reachability).

## 4. Methods
- Soft STL losses (differentiable margins).
- Offline monitoring with RTAMT; spatio‑temporal monitoring with MoonLight (STREL).
- Integration points with chosen framework(s).

## 5. Experimental Setup
Datasets/problems, specs, metrics (robustness, task loss, Pareto), compute env.

## 6. Results
Baselines vs STL‑enforced; ablations; spatial containment; robustness plots.

## 7. Discussion
Benefits/limitations, sensitivity, compute cost, threats to validity.

## 8. Conclusion & Future Work

## References
(Use a .bib file or the CITATION metadata generated in this patch set.)
