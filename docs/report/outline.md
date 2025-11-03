# Report Outline — Physical AI × (Spatial) Signal Temporal Logic

> **Course:** Vanderbilt **CS‑3860‑01 Undergraduate Research** (3 credits)  
> **Instructor:** Prof. Taylor T. Johnson  
> **Student:** Arthur Hu  
> **Repo:** [`physical-ai-stl`](../../README.md) · *living document; updated as results arrive.*

---

## 0) One‑paragraph abstract (to fill **last**)  
*We will write this after Results are finalized.*  
**Template (≤ 150 words):** Problem (safety for physics‑ML in CPS), approach (integrate **STL/STREL** monitoring into physics‑ML frameworks), experiments (ODE/PDE demos; air‑quality/traffic), key findings (robustness ↑ at modest cost), and takeaway.

---

## 1) Introduction
- **Motivation.** Physics‑ML is promising for modeling and control in CPS but needs **specification‑level guarantees**. **STL/STREL** provide machine‑checkable temporal and **spatio‑temporal** properties for safety/requirements.
- **Goal.** Prototype a **lightweight, reproducible** pipeline that **monitors** and **(optionally) enforces** STL/STREL specs in physics‑ML models (ODE/PDE, fields over space) and evaluate trade‑offs.
- **Contributions.**  
  1) **Integration design** & reference code that connects **RTAMT** (STL) and **MoonLight** (STREL) monitors with physics‑ML training/eval;  
  2) **Framework comparison**: Neuromancer, NVIDIA **PhysicsNeMo**, and **TorchPhysics**;  
  3) **STL‑ready benchmarks** (diffusion/heat, Burgers, air quality, traffic);  
  4) Empirical study: **robustness vs task error vs compute**;  
  5) Open‑source artifact with scripts/tests for **CPU‑first reproducibility**.

---

## 2) Background & tools
- **Signal Temporal Logic (STL).** Real‑valued **robustness** semantics; `always`, `eventually`, `until`; discrete/dense time; offline/online monitors.  
  - We use **RTAMT** (Python; C++ backend optional) for authoritative monitoring and slicing of formulas; bounded‑future online supported.  
    *Refs:* RTAMT repo & docs; recent RTAMT runtime‑robustness preprint.  
- **Spatio‑Temporal Reach and Escape Logic (STREL).** Extends STL with spatial operators (**reach/escape**, neighborhood graphs) for fields on grids/graphs.  
  - We use **MoonLight** (Java core; Python bridge) to monitor **spatial containment/propagation** properties on lattice graphs.  
    *Refs:* MoonLight tool paper; STREL logic paper.  
- **Object‑centric spatial relations (optional).** **SpaTiaL** for scene‑level spatial predicates when working with discrete objects/agents.  
- **Physics‑ML frameworks.**  
  - **Neuromancer** — PyTorch **differentiable programming** for constrained optimization, system ID, and **Differentiable Predictive Control (DPC)**; symbolic constraints & nodes.  
  - **NVIDIA PhysicsNeMo** — GPU‑optimized Physics‑ML (ex‑Modulus) with **PINNs** and **Neural Operators** (e.g., **FNO**, DeepONet) and distributed training.  
  - **TorchPhysics** — mesh‑free PDE learning with **PINNs**, **Deep Ritz**, **DeepONet**, **FNO**; lightweight and CPU‑friendly.  
  - (Pointers consolidated in [Framework Survey](../framework_survey.md).)

---

## 3) Problem formulation
- **Signals.**  
  - **ODE/PDE state**: $u(x,t)\in\mathbb{R}^m$ on spatial lattice $\Omega\subset\mathbb{R}^d$ and time grid $[0,T]$.  
  - **Observables**: scalars or vector fields extracted from $u$ (e.g., temperature, velocity magnitude).  
- **Specifications.**  
  - **STL (temporal only)** examples:  
    - **Safety bound:** $`\mathbf{G}_{[0,T]}(u \le U_{\mathrm{safe}})`$.  
    - **Response time:** $`\mathbf{F}_{[0,\tau]}\,\mathbf{G}_{[0,T]}(u \le U_{\mathrm{safe}})`$.  
  - **STREL (spatio‑temporal)** examples on grid graph $G=(V,E)$:  
    - **Containment:** “hotspot stays within radius $r$” using **reach/escape**;  
    - **No‑leak**: high‑value region never reaches boundary nodes within $[0,\tau]$.  
- **Robustness.** Quantitative robustness $\rho(\varphi, w, t)$ used as a scalar monitor; we aggregate over space/time.  
- **Use in ML.**  
  - **Monitor‑only**: compute robustness at eval and report violations.  
  - **Loss shaping (soft enforcement)**: add penalty $\mathcal{L}_{\mathrm{STL}} = \mathrm{softAgg}(\rho)$ with **smooth min/max** to enable backprop.

---

## 4) System design & implementation plan
- **Architecture.** *(See repo `src/physical_ai_stl/…`.)*  
  1) **Model** (PyTorch/Neuromancer/TorchPhysics/PhysicsNeMo) →  
  2) **Spec** (STL via RTAMT, STREL via MoonLight script) →  
  3) **Monitors** produce robustness traces →  
  4) **Trainer** optionally adds **differentiable STL loss** (softmin/softmax approximations).  
- **STL differentiable core.** Implemented in `monitoring/stl_soft.py`: `softmin/softmax`, `always`, `eventually`, sliding‑window ops; validated against RTAMT on small cases.  
- **STREL bridge.** `monitoring/moonlight_helper.py` builds **grid graphs** (4‑connected) and calls MoonLight (Python entry points) on `.mls` scripts; example spec `scripts/specs/contain_hotspot.mls`.  
- **RTAMT wrapper.** `monitoring/rtamt_monitor.py` creates discrete/dense‑time monitors; used for **ground‑truth robustness** and ablations.  
- **Efficiency features.**  
  - **CPU‑first** demos (PyTorch with small nets); optional GPU paths guarded behind extras;  
  - **Subsampling for monitors** (`nx, nt`) and **evaluate‑every‑k** to amortize cost;  
  - **Vectorized** robustness ops and **log‑sum‑exp** aggregations for stability;  
  - All scripts run in **≤ 1–2 minutes on CPU** for smoke tests; full runs stay < ~30 minutes.

---

## 5) Experimental plan
### 5.1 Tasks / datasets (STL‑ready)
| Tier | Task | Why | Spec ideas (examples) | Primary framework(s) |
|---|---|---|---|---|
| **T1** | **1D diffusion** (synthetic) | Deterministic, seconds‑fast | $\mathbf{G}(u\le U_{\mathrm{safe}})$; response‑time cooling | PyTorch (+soft STL), Neuromancer |
| **T1** | **2D heat** (synthetic grid) | Natural **STREL** demo | **Containment** of hotspot; **no‑leak to boundary** | PyTorch (+MoonLight), TorchPhysics |
| **T2** | **Burgers/Advection** (*PDEBench* mini) | Standard PDE operators | amplitude bounds; front speed limits | PhysicsNeMo, TorchPhysics |
| **T2** | **Air quality** (multi‑site) | Real spatio‑temporal signals | “PM2.5 stays below threshold; violations short‑lived” | Neuromancer (+RTAMT) |
| **Stretch** | **Traffic (METR‑LA/PEMS‑BAY)** | Graph‑temporal | “No corridor exceeds speed drop for >τ” | Neural operators (PhysicsNeMo/TorchPhysics) |

*(See details in* [`docs/dataset_recommendations.md`](../dataset_recommendations.md) *and* PDEBench refs.)*

### 5.2 Models & baselines
- **Baselines:** plain data loss (MSE/PDE residual) with no STL;  
- **STL‑loss variants:** softmin temperature τ, spatial aggregation (mean vs softmax), monitor frequency;  
- **Framework ablation:** Neuromancer vs PhysicsNeMo vs TorchPhysics on the same T1/T2 tasks.

### 5.3 Metrics
- **Task quality:** MSE/MAE; physics residual error (PINN);  
- **Specification quality:** average/min **robustness**, **violation rate** (% timesteps violating), time‑to‑compliance;  
- **Efficiency:** wall‑clock time/epoch, peak memory, #params/steps;  
- **Pareto**: robustness vs task loss curves.

### 5.4 Reproducibility & environment
- Deterministic seeds; small fixed configs; environment captured in `requirements*.txt` and [`docs/REPRODUCIBILITY.md`](../REPRODUCIBILITY.md).  
- Scripts: `scripts/train_diffusion_stl.py`, `scripts/train_heat2d_strel.py`, `scripts/train_neuromancer_stl.py`; plots via `scripts/utils_plot.py`.  
- CI‑style smoke tests under `tests/` keep examples quick and CPU‑safe.

---

## 6) Results (figure plan)
1. **Table:** Frameworks × tasks × metrics (MSE, robustness, time).  
2. **Curves:** robustness vs training epoch; and **Pareto** (robustness vs task loss).  
3. **Spatial heatmaps:** violation maps over the 2D grid; **MoonLight** robustness over time.  
4. **Ablations:** temperature of softmin; monitor cadence; spatial aggregation.  
5. **Efficiency plot:** time/epoch vs monitor frequency; CPU vs GPU (optional).

---

## 7) Discussion
- **Benefits/limitations.** When does STL loss help? Where does it hurt task error or compute?  
- **Specification engineering.** Writing meaningful properties; debugging false positives; monitor‑loss mismatch.  
- **Scalability.** Paths to larger domains (neural operators, batching monitors, sparse graphs).  
- **Threats to validity.** Sensitivity to smooth approximations; dataset bias; discretization effects.

---

## 8) Conclusion & next steps
- Summary of findings; recommendations for applying STL/STREL in physics‑ML.  
- **Future work:** closed‑loop control with STL in the loop (Neuromancer DPC), **NNV‑style** verification of learned controllers, richer STREL (surround/escape), and multi‑agent grids.

---

## 9) Mapping to instructor requirements (✅ = covered in this report)
- **Evaluate Neuromancer, PhysicsNeMo, TorchPhysics** → Sections **2**, **5–6** ✅  
- **Integrate STL/STREL monitoring** (RTAMT, MoonLight, SpaTiaL optional) → Sections **2**, **4–5** ✅  
- **Identify problem spaces/datasets** from the **STL angle** → Section **5.1** and [`dataset_recommendations.md`](../dataset_recommendations.md) ✅  
- **Produce an end‑of‑semester report** → this outline + scripts/figures plan ✅

---

## References (selection)
[1] **Neuromancer** — PNNL PyTorch DP library (constraints, DPC, PINNs). GitHub README and docs: https://github.com/pnnl/neuromancer .  
[2] **NVIDIA PhysicsNeMo** — Physics‑ML framework (PINNs, neural operators, distributed GPU). https://github.com/NVIDIA/physicsnemo .  
[3] **TorchPhysics** — Mesh‑free PDE learning (PINNs, Deep Ritz, DeepONet, FNO). https://github.com/boschresearch/torchphysics .  
[4] **RTAMT** — Real‑time STL monitors (offline/online; bounded‑future online; Python API with optional C++ backend). https://github.com/nickovic/rtamt .  
[5] **MoonLight** — STREL runtime monitoring; Java tool with Python bindings; supports **reach/escape** operators. https://github.com/MoonLightSuite/moonlight .  
[6] **STREL logic** — Nenzi et al., *LMCS* (2022): https://lmcs.episciences.org/8936/pdf .  
[7] **STLnet** — NeurIPS 2020: STL‑enforced RNNs; paper + code: https://proceedings.neurips.cc/paper/2020/hash/a7da6ba0505a41b98bd85907244c4c30-Abstract.html , https://github.com/meiyima/STLnet .  
[8] **PDEBench** — NeurIPS 2022 Datasets & Benchmarks: https://github.com/pdebench/PDEBench , https://arxiv.org/abs/2210.07182 .
