# Dataset & Problem‑Space Recommendations (STL‑constrained Physical‑AI)

**Scope.** This note curates datasets and concrete problem spaces where **Signal Temporal Logic (STL)** and **spatial STL (e.g., STREL)** monitoring are natural and useful. It is aimed at rapid prototyping with **Neuromancer**, **PhysicsNeMo**, or **TorchPhysics**, together with **RTAMT** (STL, real‑time/offline) and **MoonLight**/**SpaTiaL** (spatio‑temporal).

---

## TL;DR — Shortlist to start this week

| Priority | Domain | Dataset / Source | Why it fits STL | Suggested Framework | Suggested Monitor |
|---|---|---|---|---|---|
| **A** | **PDEs (heat/diffusion & Burgers)** | **PDEBench** (ready‑to‑use 1D/2D heat, Burgers, etc.) | Safety bounds, dissipation rates, hotspot containment, slope limits; STREL for “everywhere-in-radius” constraints | Neuromancer or TorchPhysics (PINNs / FNO); PhysicsNeMo for neural operators | MoonLight (STREL) for spatial constraints; RTAMT for temporal |
| **B** | **Traffic speeds (sensor networks)** | **METR‑LA** (207 sensors) & **PEMS‑BAY** (325 sensors) | Speed/flow thresholds, recovery windows after incidents; spatial neighborhoods via sensor graph | Neuromancer (state‑space, control‑style baselines) or PhysicsNeMo (operators); TorchPhysics optional | RTAMT for STL over multivariate time series; SpaTiaL for graph‑space relations |
| **C** | **Air quality** | **OpenAQ/EPA AQS** (live/archival) or **UCI Air Quality** (hourly lab+reference) | Health thresholds (e.g., PM2.5), exposure windows, post‑spike recovery; spatial dispersion when multiple stations | Neuromancer (time‑series models) | RTAMT for STL; MoonLight for spatial networks |
| **D (stretch)** | **Vehicle trajectories** | **NGSIM** (I‑80, US‑101, Lankershim) | Collision/spacing headways, speed‑limit compliance, stop‑go dynamics; spatial relations across lanes | Neuromancer (hybrid/ODE control demos) | RTAMT for kinematic STL; SpaTiaL for multi‑agent proximity |

> **Recommendation:** Start with **A (PDEBench Heat/Burgers)** for the clearest spatio‑temporal specs and fast iteration, then add **B (METR‑LA/PEMS‑BAY)** for real‑world spatio‑temporal specs on graphs. Keep **C** as a third “policy‑relevant” domain. Use **D** for a stretch goal if time allows.

---

## Selection criteria (why these)
1. **STL‑friendliness:** Natural atomic predicates (bounds, rates, recoveries) and spatio‑temporal neighborhoods (e.g., “everywhere within radius r”).
2. **Reproducibility & licensing:** Public, stable hosting; citations; light preprocessing.
3. **Compute realism:** Tractable on a single GPU / CPU; easy to downsample/slice.
4. **Integration hooks:** Clean Python APIs; fits PyTorch ecosystems used by Neuromancer / PhysicsNeMo / TorchPhysics and Python STL monitors.

---

## A. PDEs: Heat/Diffusion & Burgers via **PDEBench**

**Why this domain.** PDE fields (temperature/velocity) are **spatial grids over time**—a perfect match for spatio‑temporal logic: “no hotspot above θ anywhere,” “if a hotspot arises, it must dissipate within Δt,” “gradients remain bounded,” “shock speed never exceeds limit,” etc.

**Dataset/source.** **PDEBench** provides ready‑to‑use datasets (and code to generate more) for a wide range of PDEs (1D/2D/3D), with consistent train/val/test splits. Includes **heat/diffusion**, **Burgers**, **Darcy**, **Navier–Stokes**, **shallow water**, etc.  
Links:  
- Repo & docs: https://github.com/pdebench/PDEBench  
- DOIs for datasets & pretrained models are in the repo README.

**Suggested tasks (fast path).**
- **Heat/Diffusion 2D (toy → small)**: 64×64 spatial grid, short horizons.  
  *Specs*:  
  - **Safety bound:** `G_[0,T]  max_x T(x,t) ≤ Tmax`.  
  - **Containment:** When a hotspot appears, **everywhere** within radius ρ stays below θ within Δt (spatial “everywhere” + temporal “eventually”).  
- **Burgers 1D (waves/shocks)**:  
  *Specs*:  
  - **Slope limit:** `G_[0,T]  |∂u/∂x| ≤ κ`.  
  - **Shock speed:** shock indicator’s propagation speed ≤ v_max.

**Framework & monitors.**
- Framework: **Neuromancer** (state‑space, differentiable control constraints), **TorchPhysics** (PINNs, FNO), **PhysicsNeMo** (operators, PDE residual modules).
- Monitor: **MoonLight** (STREL) for spatial neighborhoods / distances; **RTAMT** for temporal robustness.

**Evaluation (add to training logs).**
- MAE / RMSE on fields; **robustness** ρ(φ) (mean & min across batch); **violation rate**; distance‑to‑violation; PDE residual (if PINNs); compute wall‑clock.

**Data handling & efficiency.**
- Start with provided **small shards** (e.g., 64×64 grids, short horizons).  
- Downsample in space/time for ablations (×2, ×4).  
- Keep batched dataloaders to enable monitor evaluation per‑batch.

---

## B. Traffic speeds on sensor networks: **METR‑LA** & **PEMS‑BAY**

**Why this domain.** Sensor networks over roads form **graphs**; STL can express **speed/flow thresholds**, **recovery after an incident**, and **spatial locality** (downstream neighbors satisfy bounds soon after upstream clears).

**Datasets.**
- **METR‑LA**: loop‑detector speeds from **207 sensors** over ~**4 months** (5‑min intervals).  
- **PEMS‑BAY**: speeds from **325 sensors** over ~**6 months** (5‑min intervals).

**Canonical access.**
- DCRNN repo (preprocessed `.h5` & scripts): https://github.com/liyaguang/DCRNN  
- Additional summaries: PyTorch Geometric Temporal docs; LibCity docs (sensor counts, durations).

**Example STL graph‑time specs.**
- **Recovery:** If a sensor’s speed drops below `v_low`, it **must** return above `v_min` within **Δt** and remain so for **dwell**:  
  `G_[0,T] ( speed < v_low  ->  F_[0,Δt]  G_[0,dwell] (speed ≥ v_min) )`.  
- **Downstream propagation (spatial):** **Neighbors within 1 hop** clear within Δt after upstream clears (use SpaTiaL neighborhood or encode adjacency in MoonLight if applicable).

**Framework & monitors.**
- Baselines from DCRNN/Graph WaveNet literature; for this project use **Neuromancer** (RNN/state‑space) or **PhysicsNeMo** (operator‑style) for forecasting; **RTAMT** for STL; **SpaTiaL** for spatial relations on graphs.

**Evaluation.**
- Forecast MAE/RMSE; **robustness** ρ(φ) per sensor and aggregated across subgraphs; **incident recovery time** distribution; **fraction of satisfied windows**.

**Efficiency tips.**
- Use **a subset** of sensors (e.g., 64–128 nodes) or a single corridor subgraph.  
- Clip horizon (e.g., 12 steps = 1 hour).

---

## C. Air quality (station networks)

**Why this domain.** Environmental standards translate **directly** into STL: exposure thresholds (annual or 24‑hour), post‑spike recovery, rolling windows. Spatial logic applies across stations in a city.

**Options.**
- **OpenAQ / EPA AQS:** live/archival station data worldwide/U.S.; ideal for **multi‑station STL** (spatial neighborhoods).  
- **UCI Air Quality** (Italy, 2004–2005): compact hourly dataset with lab reference analyzers; good for **fast prototypes**.

**Policy‑aligned STL examples.**
- **Short‑term health:** `G_[0,T] ( PM2.5_24h_avg ≤ 35 µg/m³ )`.  
- **Annual exposure (for reference on longer spans):** `AnnualAvg(PM2.5) ≤ 9 µg/m³`.  
- **Post‑spike recovery:** after a 1‑hour spike above θ, return below θ within Δt and stay below for dwell.

**Framework & monitors.**
- Neuromancer (time‑series); RTAMT for STL; MoonLight for spatial networks across stations.

**Efficiency tips.**
- For OpenAQ/AQS, pick **one metro area** and a **subset** of stations; for UCI, use the full hourly series directly.

---

## D. Stretch: multi‑agent vehicle trajectories (**NGSIM**)

**Why (if time permits).** Enables **hybrid/ODE** models with **agent interactions** and natural STL: headway, lane‑keeping, speed limits, stop‑go cycles.

**Data.**
- **NGSIM** collected **10 Hz** vehicle trajectories on US‑101, I‑80, and Lankershim Blvd; public access via USDOT portals.

**STL examples.**
- **Time headway safety:** `G ( headway ≥ τ_min )`.  
- **Speed compliance with recovery:** if `v > v_max`, must drop below within Δt.  
- **Lane‑change dwell:** minimum dwell time between lane changes.

**Notes.**
- Start with a **short time window** and **limited segment** (e.g., one site, 5–10 min).  
- Use RTAMT for temporal monitors; SpaTiaL for inter‑vehicle proximity.

---

## Concrete specifications (ready to code)

### 1) Heat/Diffusion (2D) — “Hotspot must dissipate quickly everywhere”
**English.** If any location exceeds θ, then **every location within radius ρ** must drop below θ within Δt and remain below for `dwell`.

**STREL sketch (MoonLight).**
```ml
# Pseudocode-ish STREL (MoonLight supports spatial 'everywhere' operators)
phi_safe      := (T <= theta)
phi_recover   := F_[0, Δt] G_[0, dwell] (T <= theta)
phi_local     := everywhere(radius = ρ, phi_recover)
phi_spec      := G_[0, T] ( (T > theta) -> phi_local )
