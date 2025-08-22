# Dataset & Problem-Space Recommendations (STL/STREL‑ready)

> **Purpose.** Curate *small, reproducible, STL‑friendly* datasets and problem spaces that let us (1) integrate runtime monitoring (RTAMT, MoonLight/STREL, SpaTiaL) with physics‑ML frameworks (Neuromancer, NVIDIA PhysicsNeMo, TorchPhysics) and (2) run fast on a laptop‐class CPU/GPU. Where helpful, we note larger stretch targets.

---

## TL;DR — top picks (fast, low‑friction)

| Tier | Problem / Dataset | Why it’s a great fit | STL / STREL spec ideas | Frameworks |
| --- | --- | --- | --- | --- |
| **T1 (pilot)** | **1D diffusion / 2D heat (synthetic)** | Already implemented here; deterministic; tiny compute; easy to visualize spatial constraints. | **STL:** bounded overshoot/settling: `G_[0,T] (min_x u(x,t) ≥ L ∧ max_x u(x,t) ≤ U)`; *eventual cooling:* `F_[0,τ] G_[0,T] (u ≤ U_safe)`.<br>**STREL:** neighborhood containment of hotspots within radius *r*. | Neuromancer, TorchPhysics; PhysicsNeMo (optional) |
| **T1 (pilot)** | **Beijing Multi‑Site Air Quality (UCI)** | Real multivariate time series; used by **STLnet**; clear safety‑style thresholds; easy STL monitoring. | **STL:** `G_[0,24h] (PM2.5 < θ_max)`; recovery: `G (spike → F_[0,3h] PM2.5 < θ_rec)`; cross‑signal rules with wind/rain. | Neuromancer (RNN/seq), RTAMT; MoonLight for regional stations |
| **T2** | **Traffic speeds — METR‑LA / PEMS‑BAY** | Canonical spatio‑temporal sensor graphs; rich literature; natural spatial logic (“neighbors”). | **STL:** *no prolonged congestion:* `G_[0,T] (speed > v_min) ∨ F_[0,Δ] (speed > v_rec)`.<br>**STREL:** propagate recovery within k‑hop neighborhood. | PhysicsNeMo (neural operators), TorchPhysics (operators), Neuromancer (seq) |
| **T2** | **PDEBench mini‑tasks** (Burgers, Advection‑Diffusion, Darcy) | Ready‑made PDE benchmarks with many IC/BCs; scalable from CPU to GPU; aligns with physics ML. | **STL:** shock bound: `G (|∂_x u| ≤ s_max)`; reach‑and‑stay: `F_[0,τ] G_[0,T] (u∈[L,U])`.<br>**STREL:** spatial bounds on fronts. | PhysicsNeMo, TorchPhysics, Neuromancer |
| **Stretch** | **2D Navier–Stokes (FNO dataset)** | Standard for neural operators; available off‑the‑shelf; pairs well with STREL for vortex/drag bounds. | **STL:** shed frequency within band; **STREL:** vorticity magnitude limited in wake region. | PhysicsNeMo (operators), TorchPhysics |

---

## Datasets / problem spaces (details & how to start)

### 1) Synthetic PDEs (Diffusion / Heat) — **fastest on CPU**
- **What:** 1D diffusion and 2D heat solvers already in this repo (`src/physical_ai_stl/physics/*`, `experiments/*`). Tiny grids run in seconds; ideal to wire STL loss/monitoring end‑to‑end.
- **STL/STREL ideas**
  - *Bounded temperature:* `G_[0,T] (u(x,t) ≤ U_safe)`; *eventual cooling:* `F_[0,τ] G_[0,T] (u ≤ U_safe)`.
  - *Spatial containment* (STREL): high‑temp region does not expand beyond radius *r* around source during `[0,τ]`.
- **Why now:** No data logistics; deterministic; supports differentiable robustness (soft min/max) for training.
- **Where:** This repo (unit tests & examples).

### 2) Air quality — **Beijing Multi‑Site (UCI)**
- **What:** Hourly pollutants + weather at multiple Beijing stations (2013–2017). Used directly by **STLnet** for logic‑guided forecasting.  
  Sources: [UCI dataset page](https://archive.ics.uci.edu/dataset/501/beijing%2Bmulti%2Bsite%2Bair%2Bquality%2Bdata) and [STLnet (NeurIPS 2020) paper](https://proceedings.neurips.cc/paper/2020/file/a7da6ba0505a41b98bd85907244c4c30-Paper.pdf).
- **Why:** Natural STL safety predicates (AQI/PM thresholds, recovery after spikes); multi‑signal coupling (e.g., wind/rain imply drop in PM2.5).
- **Quickstart**
  - Subselect a few stations (e.g., 4–8) and 1–3 months to keep runs light.
  - Train a small RNN/TCN baseline in PyTorch/Neuromancer; attach RTAMT monitor for offline robustness and a *soft‑robustness* term for training.
  - Optional: Map nearby stations and use MoonLight to express spatial “within‑radius” constraints.

### 3) Traffic graphs — **METR‑LA / PEMS‑BAY**
- **What:** Road‑sensor speed datasets widely used in spatio‑temporal forecasting; provide sensor coordinates/adjacency with 5‑min sampling.  
  Sources: DCRNN data prep for [METR‑LA/PEMS‑BAY](https://github.com/liyaguang/DCRNN) (original paper: ICLR’18 OpenReview PDF).
- **STL/STREL ideas**
  - *No prolonged congestion:* `G_[0,T] (speed > v_min) ∨ F_[0,Δ] (speed > v_rec)` (every segment must eventually recover).
  - *Spatial recovery:* STREL rule that recovery at a node propagates to its *k*‑hop neighborhood within Δ.
- **Quickstart**
  - Begin with METR‑LA (fewer sensors than PEMS‑BAY). Use the public preprocessing script to produce small `.npz` splits.
  - Evaluate STL monitors offline (RTAMT). For spatial operators, either (a) encode k‑hop neighborhoods as MoonLight locations or (b) approximate via graph distance thresholds.
- **Notes:** Training full SOTA models is not required; focus on demonstrating **logic‑aware training/evaluation**.

### 4) PDEBench (Burgers / Advection‑Diffusion / Darcy) — **scalable**
- **What:** Open benchmark suite with code and datasets for many PDEs; supports multiple IC/BC distributions and resolutions.  
  Sources: [PDEBench paper](https://arxiv.org/abs/2210.07182) and [code/data](https://github.com/pdebench/PDEBench).
- **Why:** Standardizes PDE tasks; cleanly integrates with PINNs and neural operators.
- **STL/STREL ideas:** Bound shock steepness, ensure front arrival by time τ, enforce region‑wise bounds.
- **Quickstart:** Start with 1D Burgers (small grids), then 2D Advection‑Diffusion. Use PhysicsNeMo or TorchPhysics examples as templates.

### 5) 2D Navier–Stokes (FNO dataset) — **stretch goal**
- **What:** Pre‑generated vorticity‑form dataset from the FNO paper; e.g., `NavierStokes_V1e-3_N5000_T50.mat` (≈ 5000 samples of 64×64 over 50 steps).  
  Sources: [FNO paper](https://arxiv.org/abs/2010.08895), and example dataset links (e.g., GitHub mirrors).
- **Why:** Rich spatio‑temporal patterns (vortex shedding) ideal for STREL; common in PhysicsNeMo/Modulus examples.
- **STL/STREL ideas:** Enforce bounds on vorticity magnitude in wake; constrain shedding frequency/band via temporal operators.

---

## Example STL / STREL snippets

> We’ll keep two parallel paths: *exact* monitors (RTAMT/MoonLight) for evaluation, and *differentiable* approximations for training.

```text
# STL (RTAMT) — no prolonged exceedance (air quality)
phi := always_[0,24h] ( PM25 < 35  or  eventually_[0,3h] PM25 < 35 )

# STL — reach-and-stay (diffusion cooling)
phi := eventually_[0,1.0] always_[0,2.0] ( temp ≤ 45 )

# STREL (MoonLight) — spatial containment for 2D fields
phi := always_[0,1.0]  ( forAllWithin(radius = 0.1)  ( temp ≤ 60 ) )
```

---

## Selection criteria we used

- **STL‑expressiveness:** easy to write meaningful safety/performance specs (thresholds, recovery, spatial propagation).  
- **Reproducibility & weight:** download size small or synthetic; scripts exist; license OK.  
- **Framework fit:** supports PINNs/operators/sequences across **Neuromancer**, **PhysicsNeMo**, **TorchPhysics**.  
- **Compute:** runnable on CPU or a single modest GPU; can scale up later.

---

## Proposed evaluation matrix (first 3–4 weeks)

| Dataset | Baseline model | Monitor(s) | Targets |
| --- | --- | --- | --- |
| Diffusion/Heat (synthetic) | Small CNN/FNO or PINN | RTAMT (STL) + MoonLight (STREL) + soft‑robustness loss | E2E wiring; plots of robustness vs. training; pass unit tests |
| Beijing Air Quality | RNN/TCN (Neuromancer / PyTorch) | RTAMT offline; soft‑robustness as auxiliary loss | Improve robustness without hurting MAE; ablations on θ |
| METR‑LA (subset) | Lightweight GNN/TCN | RTAMT + MoonLight | Show spatial rule satisfaction vs. vanilla training |
| PDEBench (1D Burgers) | Tiny FNO/PINN | RTAMT + soft‑robustness | Demonstrate spec‑aware training on PDE benchmark |

---

## Pointers (official pages / docs)

- **Neuromancer:** [GitHub](https://github.com/pnnl/neuromancer) · [Docs](https://pnnl.github.io/neuromancer/).  
- **NVIDIA PhysicsNeMo:** [GitHub](https://github.com/NVIDIA/physicsnemo) · [Overview](https://developer.nvidia.com/physicsnemo). *(Renamed from Modulus; v1.2.0 released Aug 2025 on GitHub.)*  
- **TorchPhysics:** [GitHub](https://github.com/boschresearch/torchphysics) · [Docs](https://boschresearch.github.io/torchphysics/).  
- **RTAMT (STL monitors):** [GitHub](https://github.com/nickovic/rtamt).  
- **MoonLight / STREL:** [GitHub](https://github.com/MoonLightSuite/moonlight) · [Paper PDF](https://link.springer.com/content/pdf/10.1007/s10009-023-00710-5.pdf).  
- **SpaTiaL:** [API Docs](https://kth-rpl-planiacs.github.io/SpaTiaL/).  
- **STLnet (NeurIPS 2020):** [Paper](https://proceedings.neurips.cc/paper/2020/file/a7da6ba0505a41b98bd85907244c4c30-Paper.pdf).  
- **PDEBench:** [Paper](https://arxiv.org/abs/2210.07182) · [Code/Data](https://github.com/pdebench/PDEBench).  
- **FNO datasets:** [Paper](https://arxiv.org/abs/2010.08895) · example Navier–Stokes `.mat` files (see referenced repos).  
- **Traffic (METR‑LA/PEMS‑BAY):** [DCRNN repo (data prep)](https://github.com/liyaguang/DCRNN).

---

## Recommended starting scope (agrees with “fast, resource‑efficient”)

1. **Week 1:** finalize specs for *Diffusion/Heat* and *Beijing AQ*; wire RTAMT + soft‑robustness; tiny baselines.  
2. **Week 2:** add MoonLight STREL to *Diffusion/Heat*; run METR‑LA subset; measure logic satisfaction & robustness.  
3. **Week 3+:** move one PDEBench task; optional Navier–Stokes (FNO) if GPU time allows.

> These cover the professor’s guidance: (a) evaluate **Neuromancer / PhysicsNeMo / TorchPhysics**, (b) integrate **STL/STREL**, and (c) **recommend datasets** aligned with the STL angle.

---

*Last updated: 2025‑09‑30.*
