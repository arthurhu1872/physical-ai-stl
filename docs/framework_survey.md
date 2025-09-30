# Framework Survey — Physical AI + (Spatial) Signal Temporal Logic (STL)

> **Purpose.** Compare three candidate physics‑ML frameworks and three STL/STREL toolchains, then recommend a primary stack for **CS‑3860‑01 undergraduate research**. Emphasis: **small, CPU‑friendly demos first**, clean integration points for **STL monitoring/enforcement**, and **clear paths to scale**.

---

## TL;DR — Recommendation

- **Primary framework:** **Neuromancer** — best balance of **PyTorch‑native APIs**, differentiable constrained optimization, PINNs/system ID, and **simple hooks for STL losses/monitors**. Works well on CPU; optional GPU via Conda.  
- **Secondary (PDE‑focused) framework:** **TorchPhysics** — light, mesh‑free PINNs/DeepRitz/DeepONets; fast to prototype extra PDEs (e.g., Burgers).  
- **Stretch / GPU‑optimized stack:** **NVIDIA PhysicsNeMo** — richest **neural‑operator** & physics‑AI ecosystem, but heavier install and GPU‑leaning; keep for later or when scaling.

**STL/STREL tools:**  
- **RTAMT** (Python) for STL robustness monitoring (offline + online bounded‑future).  
- **MoonLight** (Java core, Python bindings) for **STREL** (spatio‑temporal reach/escape); use for grid‑based PDE fields (e.g., 2D heat).  
- **SpaTiaL** (Python) for object‑centric spatial relations and simple planning; optional for robotics‑style demos.

> Decision drivers: **ease of integration**, **CPU‑first reproducibility**, **robust STL monitoring support**, and **clear community/docs**.

---

## 1) Frameworks at a glance

| Framework | What it is (official) | Strengths for us | Friction / risks | License |
|---|---|---|---|---|
| **Neuromancer** ([GitHub](https://github.com/pnnl/neuromancer) · [Docs](https://pnnl.github.io/neuromancer/)) | PyTorch‑based **differentiable programming** library for parametric constrained optimization, physics‑informed system ID, and model‑based control. | PyTorch‑native; easy to plug STL losses/monitors into training loop; supports ODE/PDE workflows; prior DPC examples; good CPU path. | Smaller ecosystem than NVIDIA; fewer canned neural‑operator baselines. | BSD‑3‑Clause |
| **NVIDIA PhysicsNeMo** ([GitHub](https://github.com/NVIDIA/physicsnemo) · [Docs](https://docs.nvidia.com/physicsnemo/) · [Site](https://developer.nvidia.com/physicsnemo)) | Open‑source deep‑learning **Physics‑AI** framework (successor path from Modulus) for building/training physics‑ML models and neural operators. | Strong operator learning (FNO/UNO/PINO variants), scalable training, rich examples; integrates with NVIDIA tooling. | Heavier dependencies; truly shines on NVIDIA GPUs; overkill for week‑1 CPU demos. | Apache‑2.0 |
| **Bosch TorchPhysics** ([GitHub](https://github.com/boschresearch/torchphysics) · [Docs](https://boschresearch.github.io/torchphysics/)) | Mesh‑free deep learning methods to **solve ODE/PDEs** (PINNs, DeepRitz; operator learning support). | Very simple API; quick PDE baselines; good for adding Burgers/Advection‑Diffusion tasks. | Smaller community; fewer utilities around monitoring/logging. | MIT |

**Why Neuromancer first?** We can add STL regularizers directly around PyTorch tensors with minimal ceremony, while retaining a path to optimal control/DPC. PhysicsNeMo becomes attractive once we want **neural operators** at scale, but that’s not a blocker for our initial STL‑centric milestones.

---

## 2) STL/STREL toolchain

| Tool | What it does | Why it fits our needs | Notes |
|---|---|---|---|
| **RTAMT** ([GitHub](https://github.com/nickovic/rtamt)) | **STL robustness** monitoring in Python; offline and online (bounded‑future); discrete & dense time; optimized C++ backend for online discrete. | Drop‑in robustness metric `ρ(φ, x)` for training/eval; easy CPU install. | Good docs; pair with our training loop for soft constraints. |
| **MoonLight** ([GitHub](https://github.com/MoonLightSuite/moonlight) · [Tool paper, 2023](https://link.springer.com/content/pdf/10.1007/s10009-023-00710-5.pdf)) | Java tool with Python interface for **temporal, spatial, and spatio‑temporal** monitoring (**STREL**). | Needed for grid/field data (e.g., 2D heat); supports **reach/escape** operators over neighborhoods. | Requires Java ≥ 11; Python bindings available; slightly heavier. |
| **SpaTiaL** ([Paper, 2023](https://link.springer.com/article/10.1007/s10514-023-10145-1)) | Spatio‑temporal spec/monitor/plan framework for **object‑centric** tasks. | Useful for robotics‑style demos; complements STREL. | PyPI package: `spatial-spec`; Linux/macOS only at the moment. |

**Related exemplar:** **STLnet** (NeurIPS 2020) enforces STL properties during training to improve prediction robustness in CPS time‑series ([paper](https://proceedings.neurips.cc/paper/2020/file/a7da6ba0505a41b98bd85907244c4c30-Paper.pdf), [code](https://github.com/meiyima/STLnet)). We adapt the “**soft enforcement via robustness‑based loss**” idea to PDE fields and neural ODE/PDEs.

---

## 3) How we integrate STL into training (soft enforcement)

Let `ρ(φ, x)` be the **robustness** of an STL/STREL formula `φ` on trace `x`. For a *safety* spec (e.g., bounds), define a smooth penalty
\[
\mathcal{L}_{\text{stl}}(x) \,=\, \operatorname{softplus}\big(\tau^{-1}\,[m - ρ(φ, x)]\big),
\]
with **margin** `m ≥ 0` and **temperature** `τ > 0`. The **total loss** becomes  
\( \mathcal{L} = \mathcal{L}_{\text{task}} + \lambda\,\mathcal{L}_{\text{stl}} \).  
Tune \(\lambda, m, τ\) via ablations.

**Implementation sketch (PyTorch):**

```python
# y_hat: model output over time/space; y_true optional
rho = monitor.robustness(y_hat, t, x)      # RTAMT (STL) or MoonLight (STREL)
stl_loss = F.softplus((margin - rho) / temp)
loss = task_loss(y_hat, y_true) + lam * stl_loss.mean()
loss.backward(); opt.step()
```

For **STREL** on 2D heat fields, supply MoonLight a **grid graph** (cells as nodes; edges tie 4‑ or 8‑neighbors); evaluate operators like “**within r hops some neighbor exceeds threshold**” or “**always in a region temperature stays bounded**”.

---

## 4) Minimal runnable demos (in this repo)

> Designed to run **on CPU** in minutes.

| Demo | What it shows | Entry point |
|---|---|---|
| **1D diffusion + STL bound** | Soft enforcement of `G_[0,T] (u ≤ U)` via RTAMT. | `python scripts/train_diffusion_stl.py --config configs/diffusion1d_stl.yaml` |
| **2D heat + STREL** | MoonLight monitoring on a grid: region bounds / reach‑avoid. | `python scripts/train_heat2d_strel.py --config configs/heat2d_baseline.yaml` |
| **Neuromancer sine with bound** | Neuromancer loop with STL penalty; CPU‑fast. | `python scripts/train_neuromancer_stl.py --config configs/neuromancer_sine_bound.yaml` |
| **Burgers (TorchPhysics)** | Quick PDE baseline; later add STL. | `python scripts/train_burgers_torchphysics.py` |
| **Evaluation** | Robustness/violation curves and ablations. | `python scripts/run_ablations_diffusion.py`; `python scripts/plot_ablations.py` |

> See `docs/REPRODUCIBILITY.md` for environment notes (CPU‑only path by default; Java required for MoonLight).

---

## 5) Evaluation rubric (how we will compare frameworks)

**A. Engineering**  
- *Install & footprint*: CPU‑only install; optional extras (GPU/Java).  
- *API ergonomics*: lines‑of‑code to a working PDE+STL demo.  
- *Extensibility*: ease of adding STREL, new PDEs/operators.  
- *Logging & reproducibility*: seeds, configs, eval scripts.

**B. Science**  
- *Task fit*: ODE/PDE coverage; operator learning availability.  
- *STL compatibility*: working offline/online monitors; robustness semantics; spatial support.  
- *Performance on CPU*: wall‑clock to reach spec satisfaction; steady‑state robustness; violation rate.  
- *Robustness*: performance under noise/OOD initial conditions.

**C. Community & docs**  
- Stars/issues cadence; documentation quality; example coverage; license clarity.

We will report **Pareto curves (task loss vs. spec robustness)** and **ablation plots** (λ, τ, monitor cadence).

---

## 6) Suggested problem spaces & datasets (STL/STREL‑ready)

**Tier‑1 (pilot, minutes on CPU)**
1. **1D diffusion / 2D heat (synthetic)** — already scaffolded here; ideal for bound and region invariants.  
   *Specs:* `G_[0,T] (u ∈ [L, U])`, `G_[0,T] (Region_A ⇒ u ≤ U_A)`, `F_[0,τ] G_[0,Δ] (u ≤ U)`.

2. **Air quality — Beijing Multi‑Site (UCI)** — multivariate time‑series with spatial stations (STREL across stations).  
   *Specs:* “eventually within N hours PM2.5 ≤ θ”, “always NO₂ below θ in residential zones”.  
   *Hook:* STLnet‑style loss on RNN/seq model; optional MoonLight for station neighborhoods.  
   **Refs:** [STLnet paper](https://proceedings.neurips.cc/paper/2020/file/a7da6ba0505a41b98bd85907244c4c30-Paper.pdf), [code](https://github.com/meiyima/STLnet).

**Tier‑2 (still reasonable; small GPU optional)**
3. **Traffic speeds — METR‑LA / PEMS‑BAY** — canonical graph time‑series; spatial adjacency suits STREL.  
   *Specs:* “main corridors stay under congestion threshold except finite bursts”; reach‑avoid between zones.

4. **PDEBench mini‑tasks** — e.g., **Burgers, Advection‑Diffusion**; small subsets for CPU or single‑GPU.  
   *Specs:* bounds/invariants; reachability (fronts crossing a region).  
   **Refs:** [PDEBench](https://github.com/pdebench/PDEBench) (NeurIPS 2022 datasets paper).

**Stretch**
5. **Navier–Stokes / CFD tiles** (PhysicsNeMo examples) — for neural operators and scaling studies.

---

## 7) Concrete specs we will start with

- **STL (scalar signals)**  
  - **Bounded invariance:** \( φ_1 := G_{[0,T]}(u ≤ U) \)  
  - **Bounded response:** \( φ_2 := G_{[0,T]}(a > θ \to F_{[0,τ]} (u ≤ U)) \)

- **STREL (grid fields)**  
  - **Regional bound:** \( φ_3 := G_{[0,T]}(\text{in\_A} \to u ≤ U_A) \)  
  - **Spatial reach‑avoid:** \( φ_4 := G_{[0,T]}(\text{in\_safe} \land \neg \text{near\_hot}) \)

We will encode \( \text{in\_A} \) via mask sets on the grid and “**near**” using r‑hop neighborhoods in MoonLight.

---

## 8) Risks & mitigations

- **MoonLight Java friction.** Mitigation: keep STL‑only experiments (RTAMT) as CPU‑first baseline; add MoonLight when Java is available.  
- **GPU reliance (PhysicsNeMo).** Mitigation: postpone to stretch tasks; keep TorchPhysics/Neuromancer baselines CPU‑friendly.  
- **Spec mis‑specification.** Mitigation: add unit tests for monitors on synthetic traces; visualize robustness signals.

---

## 9) Quick pointers / primary sources

- **Neuromancer:** GitHub repo and docs — features DPC, physics‑informed system ID, PyTorch DP APIs.  
  Sources: [GitHub](https://github.com/pnnl/neuromancer), [Docs](https://pnnl.github.io/neuromancer/), [DPC examples](https://github.com/pnnl/deps_arXiv2020).

- **NVIDIA PhysicsNeMo:** open‑source physics‑AI framework with operator models; best with NVIDIA GPUs.  
  Sources: [GitHub](https://github.com/NVIDIA/physicsnemo), [Docs](https://docs.nvidia.com/physicsnemo/), [Overview](https://developer.nvidia.com/physicsnemo).

- **TorchPhysics:** mesh‑free PDE learning (PINNs/DeepRitz).  
  Sources: [GitHub](https://github.com/boschresearch/torchphysics), [Docs](https://boschresearch.github.io/torchphysics/).

- **RTAMT:** STL monitoring (offline + online bounded‑future; discrete & dense).  
  Source: [GitHub](https://github.com/nickovic/rtamt).

- **MoonLight / STREL:** spatio‑temporal monitoring; Python interface available.  
  Sources: [GitHub](https://github.com/MoonLightSuite/moonlight), [Tool paper (2023)](https://link.springer.com/content/pdf/10.1007/s10009-023-00710-5.pdf).

- **SpaTiaL:** spatio‑temporal spec & planning for robotics.  
  Source: [Paper (2023)](https://link.springer.com/article/10.1007/s10514-023-10145-1).

- **STLnet (NeurIPS 2020):** STL‑guided RNN training.  
  Sources: [Paper](https://proceedings.neurips.cc/paper/2020/file/a7da6ba0505a41b98bd85907244c4c30-Paper.pdf), [Code](https://github.com/meiyima/STLnet).

---

## 10) What we will do next (first two weeks)

1) **Finalize the primary stack** (Neuromancer first; keep TorchPhysics handy; bookmark PhysicsNeMo).  
2) **Reproduce CPU pilots**: 1D diffusion (RTAMT), 2D heat (MoonLight).  
3) **Propose 2–3 specs per task** (from Section 7) and run ablations on \(\lambda, τ\).  
4) **Select one Tier‑2 dataset** (traffic or PDEBench mini) and draft the integration plan.  
5) **Write down decisions** (pros/cons observed, wall‑clock & robustness plots) for the end‑of‑semester report.

---

> _Maintainer note_: keep this file **precise and stable**. Link out to official docs; keep the **CPU‑first path** front‑and‑center; prefer **small, testable specs** over sprawling plans.
