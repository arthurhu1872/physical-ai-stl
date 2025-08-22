# Sprint 2 Report — Physical AI × (Spatial) Signal Temporal Logic

**Course:** Vanderbilt **CS‑3860‑01 Undergraduate Research** (3 credits)  
**Instructor:** Prof. Taylor T. Johnson  
**Student:** Arthur Hu  
**Repo:** `physical-ai-stl` → docs, code, and reproducibility scripts live here.  
**Sprint window:** **Sep 16–Sep 30, 2025** (2 weeks)

---

## 1) Executive summary (what we decided + why)

- **Primary framework for prototypes: _Neuromancer_** — concise PyTorch‑native API for differentiable constrained optimization, physics‑informed system ID, and predictive control; easiest path to inject **STL‑shaped regularizers** directly into loss graphs. Strong docs and active maintenance from PNNL. [GitHub](https://github.com/pnnl/neuromancer), [docs](https://pnnl.github.io/neuromancer/).  
- **PDE‑focused secondary: _TorchPhysics_** — lightweight, mesh‑free PDE learning (PINNs, DeepRitz/DeepONet/FNO), with clean geometry/sampling utilities; ideal for fast PDE ablations (diffusion, Burgers, shallow‑water). [GitHub](https://github.com/boschresearch/torchphysics), [docs](https://boschresearch.github.io/torchphysics/).  
- **Scaling stack (keep warm): _NVIDIA PhysicsNeMo_** — rich SciML modules (PINNs, neural operators, distributed training); heavier install, container‑first; reserve for GPU scale‑out or when neural‑operator baselines are needed. [GitHub](https://github.com/NVIDIA/physicsnemo), [docs](https://docs.nvidia.com/physicsnemo/latest/getting-started/installation.html), [overview](https://developer.nvidia.com/physicsnemo).  

- **STL/STREL monitoring:**  
  - **RTAMT** for **STL** robustness (offline + online bounded‑future) with optional C++ backend. Good fit for CPU‑fast audits and unit tests. [GitHub](https://github.com/nickovic/rtamt).  
  - **MoonLight** for **STREL** (spatio‑temporal reach/escape) with Python bindings around a Java core; use when the **spec is over a spatial graph** (grids, sensor networks). [GitHub](https://github.com/MoonLightSuite/moonlight), [tool paper](https://link.springer.com/article/10.1007/s10009-023-00710-5), [STREL paper (LMCS 2022)](https://lmcs.episciences.org/8936/pdf).  
  - **SpaTiaL** for **object‑centric** spatial‑temporal relations and automaton‑based planning; keep as optional add‑on. [Docs](https://kth-rpl-planiacs.github.io/SpaTiaL/), [GitHub](https://github.com/KTH-RPL-Planiacs).  

- **Candidate problem spaces/datasets (STL‑friendly):**
  1) **PDEBench** (NeurIPS’22) — standard PDE surrogates (advection, **Burgers**, diffusion‑reaction, shallow‑water, Navier–Stokes) with large, ready‑to‑use datasets; ideal for STL/STREL over **fields**. [Paper](https://arxiv.org/abs/2210.07182), [repo](https://github.com/pdebench/PDEBench).  
  2) **Traffic forecasting (METR‑LA/PEMS‑BAY)** — road‑graph speed sensors; natural **spatial adjacency** → STREL specs (“congestion should not propagate upstream beyond _k_ hops without relief”). [METR‑LA summary](https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset).  
  3) **Air‑quality sensors (US EPA/AirData)** — hourly PM2.5/O₃ time series over stations; STL caps and recovery clauses for public‑health ranges; optional STREL via spatial proximity. [EPA AirData](https://www.epa.gov/outdoor-air-quality-data/air-data-basic-information).  

- **Enforcement approach (differentiable):** use **robust STL semantics** (Donzé–Maler) as a training penalty and apply **smooth max/min** via **log‑sum‑exp** (softmax/softmin) to keep gradients stable; retain RTAMT/MoonLight as truth‑level auditors for exact post‑hoc checks.  
  _Refs:_ Robust semantics [FORMATS 2010](https://www-verimag.imag.fr/~maler/Papers/sensiform.pdf); STLnet (NeurIPS 2020) shows logic‑guided sequence learning benefits and informs our regularizer design: [paper](https://proceedings.neurips.cc/paper/2020/file/a7da6ba0505a41b98bd85907244c4c30-Paper.pdf).

**Deliverable of Sprint 2:** a concrete design + working scaffolds (CPU‑only) to **train small PDE surrogates with soft‑STL losses** and to **audit with RTAMT/MoonLight**; plus a crisp recommendation on frameworks & datasets to carry forward.

---

## 2) What we built this sprint

> The code for all bullets below lives under `src/physical_ai_stl/` and `tests/`. The emphasis was **deterministic CPU smoke tests** with tiny grids so everything is fast to run and easy to debug.

1) **STL‑as‑loss scaffolding (soft semantics).**  
   We implemented a minimal, framework‑agnostic helper that maps an STL formula over a scalar **trace** or a spatio‑temporal **field** into a **robustness** score ρ and a **soft** robustness ρ̃ suitable for gradient‑based training:


   - **Soft max/min**:  
     \[
       \operatorname{softmax}_\beta(x_1,\dots,x_n)=\tfrac{1}{\beta}\log\sum_i e^{\beta x_i},\quad
       \operatorname{softmin}_\beta(\cdot)=-\operatorname{softmax}_\beta(-\cdot)
     \]
     We use these to smooth the \(\max/\min\) in STL’s quantitative semantics and anneal \(\beta\uparrow\) over training for a closer match to exact robustness.
   - **Loss:** \(\mathcal{L}_{\text{STL}} = \operatorname{ReLU}(-\tilde{\rho}(\varphi; x))\) for “must‑satisfy” specs (penalize only when violated), optionally **marginized** (target \(\tilde{\rho}\ge m\)).  
   - **Auditors:** exact robustness via **RTAMT** (time‑only) and **MoonLight** (spatio‑temporal) run **offline** after training to confirm satisfaction and quantify margins.

2) **Tiny PDE exemplars (CPU‑fast):**  
   - **1D diffusion (“heat”)** and **1D viscous Burgers** solvers/generators with fixed seeds for reproducible synthetic datasets.  
   - Torch‑shaped arrays + helpers to convert field snapshots to **signals** required by STL/STREL monitors (e.g., max over a spatial slice becomes a signal predicate).  
   - Simple **“safety” specs** used in tests, e.g.  
     - Range bound: *“always, field value in \([-1,1])\”* → \(\mathbf{G}_{[0,T]}(|u|\le 1)\).  
     - Hot‑spot recovery: *“if \(u>\tau\) then within \([0,\Delta]\) it returns below \(\tau\)”* → \(\mathbf{G}(u>\tau\Rightarrow \mathbf{F}_{[0,\Delta]}(u\le \tau))\).

3) **MoonLight “hello STREL”.**  
   - A minimal grid‑graph wrapper so **spatial operators** like **reach/escape** apply to 1D/2D lattices produced by the PDE examples.  
   - Example STREL spec: *“Everywhere, if a cell exceeds \(\tau\), a cell within 2 hops drops below \(\tau\) within \(\Delta\) time”* (captures **dissipation spread**).

4) **Project plumbing for speed/repro:**  
   - `Makefile` targets for quickstart + fast tests; deterministic seeds; small default grids; CI‑friendly design.  
   - Modular directories (`experiments/`, `frameworks/`, `physics/`, `training/`, `utils/`) to keep enforcement logic decoupled from framework code.

> **Note.** Optional heavy dependencies (Neuromancer, PhysicsNeMo, TorchPhysics) are _not_ required to run the CPU smoke tests; they will be enabled behind flags and separate extras to keep the dev loop fast.

---

## 3) Frameworks — succinct evaluation

| Criterion | **Neuromancer** | **TorchPhysics** | **NVIDIA PhysicsNeMo** |
|---|---|---|---|
| **Focus** | Differentiable constrained optimization; **PINNs / system ID / DPC** | Mesh‑free PDE learning (**PINN**, **DeepRitz**, **DeepONet**, **FNO**) | End‑to‑end **Physics‑AI** stack; PINNs + **neural operators**; multi‑GPU |
| **Install** | `pip`/Conda; PyTorch‑native; good docs | `pip`; light deps; good docs | Container‑first; Hydra configs; best with NVIDIA stack |
| **API ergonomics** | Clean modules for losses/constraints; easy to add custom penalties | Clear geometry/sampler abstractions; quick PDE setups | Rich but heavy; strong examples; best for scale |
| **Fit for STL enforcement** | **Excellent** (attach \(\mathcal{L}_{\text{STL}}\) directly) | **Good** (add to PINN residual loss) | **Good** (Hydra loss hooks; more boilerplate) |
| **When to use** | **Default** for ODE/PDE toy to mid‑scale | **PDE baselines** and fast field experiments | **Scaling** or **neural‑operator** baselines |

Refs: Neuromancer [GitHub/doc] ; TorchPhysics [doc] ; PhysicsNeMo [GitHub/Docs/Overview].

---

## 4) Datasets + STL/STREL spec ideas

1) **PDEBench** (fields on grids).  
   - **Why:** standardizes PDE tasks and lets us compare across PINN vs. neural‑operator surrogates.  
   - **Specs:**  
     - **Energy/range bounds:** \(\mathbf{G}(|u|\le M)\) for stability.  
     - **Dissipation:** if a hot spot occurs, **eventually** the gradient magnitude decays below \(\gamma\) within \(\Delta\): \(\mathbf{G}(\|\nabla u\|>\gamma \Rightarrow \mathbf{F}_{[0,\Delta]}\|\nabla u\|\le\gamma)\).  
     - **STREL** surround: a high region must be **surrounded** by lows after \(\Delta\) (checks diffusion spread).

2) **Traffic (METR‑LA / PEMS‑BAY).**  
   - **Why:** canonical spatio‑temporal graphs; natural **reach/escape** semantics along road topology.  
   - **Specs:**  
     - **No persistent gridlock:** \(\mathbf{G}(v<\tau \Rightarrow \mathbf{F}_{[0,\Delta]} v\ge \tau)\).  
     - **No upstream blowback:** congestion should **not reach** upstream beyond **k hops** without relief (STREL **reach**).  

3) **Air Quality (EPA AirData).**  
   - **Why:** public, hourly, geographically distributed; policy‑relevant **threshold rules**.  
   - **Specs:** **AQI cap** and **recovery** windows; **spatial correlation** (if one station spikes, neighbors de‑spike within \(\Delta\)).

---

## 5) Results (Sprint‑2, CPU‑only smoke tests)

> Purpose here is **method validation**, not SOTA accuracy. We targeted tiny problems that run fast and demonstrate end‑to‑end STL‑in‑the‑loop training plus auditing.

- **1D diffusion** with bound spec \(\mathbf{G}(|u|\le 1)\):  
  - Training with \(\lambda_{\text{STL}}>0\) eliminated overshoots seen in the baseline (\(\lambda_{\text{STL}}=0\)) on synthetic runs.  
  - Post‑hoc **RTAMT** robustness margins were **non‑negative** across seeds, confirming satisfaction.
- **Burgers (viscous)** with recovery spec: violations decreased markedly under STL‑regularized training; **MoonLight** STREL check on a 1D chain confirmed **reach‑then‑recover** behavior within the budgeted horizon.

*(Full numeric tables and plots will be placed in the final report; for now we keep the examples minimal and fast to reproduce.)*

---

## 6) Design details you can audit quickly

### 6.1 Soft STL loss (drop‑in)

```python
# Pseudo‑code (framework‑agnostic)
def softmax_beta(x, beta):      # smooth max
    return (1.0 / beta) * log_sum_exp(beta * x)

def softmin_beta(x, beta):      # smooth min
    return -softmax_beta(-x, beta)

def soft_robustness(phi, trace, beta):
    # Implement STL quantitative semantics with softmin/softmax in place of min/max:
    # ρ̃(¬φ) = −ρ̃(φ),  ρ̃(φ1 ∧ φ2) = softmin(ρ̃(φ1), ρ̃(φ2)), ρ̃(G_I φ) = softmin_{t∈I} ρ̃(φ,t), etc.
    ...

def stl_loss(phi, trace, beta=10.0, margin=0.0, weight=1.0):
    rho_tilde = soft_robustness(phi, trace, beta)
    violation = relu(margin - rho_tilde)   # penalize only when margin not met
    return weight * violation.mean()
```

- **Audit path:** after training, compute **exact** robustness via RTAMT/MoonLight on saved traces; compare to soft proxy.

### 6.2 Where this plugs in

- **Neuromancer:** add `stl_loss(...)` to the objective dict next to data fidelity; weight with a schedule.  
- **TorchPhysics:** append to the composite loss along with PDE residual and boundary terms.  
- **PhysicsNeMo:** add to Hydra loss stack in the training loop (kept for later scale‑out).

---

## 7) Risks & mitigations

- **Non‑differentiability at \(\max/\min\)** → mitigated via **soft** operators and **annealing** \(\beta\).  
- **Spec brittleness** (too strict → over‑regularize) → adopt **margin m** and curriculum on \(\lambda_{\text{STL}}\).  
- **Install friction** (MoonLight’s Java, PhysicsNeMo containers) → keep **audits optional** and **CPU‑only mode** by default; container recipes later.

---

## 8) Plan for Sprint 3 (Oct 1–Oct 14, 2025)

**Goals (measurable):**
1. **Neuromancer**: end‑to‑end training on **PDEBench 1D diffusion** with two specs; table of MAE vs. STL‑satisfaction margin (± robust semantics).  
2. **TorchPhysics**: replicate the above on **1D Burgers**; ablate \(\beta\) and \(\lambda_{\text{STL}}\).  
3. **MoonLight STREL**: 2D grid demo (diffusion) with **reach/escape** and **surround**; include a simple figure.  
4. **Dataset decision**: down‑select **two** problem spaces to carry to the final report (likely PDEBench + Traffic _or_ Air‑quality) with concrete specs.  
5. **Reproducibility**: add `requirements-optional.txt` and a `make experiments` target for one‑command runs.

**Stretch:** spin up **PhysicsNeMo** container and replicate 1 task using a neural operator (FNO/UNO) to test STL regularization in that stack.

---

## 9) Pointers & references (curated)

- **Neuromancer:** PNNL GitHub and docs — differentiable programming for constrained optimization and physics‑informed modeling.  
  <https://github.com/pnnl/neuromancer> · <https://pnnl.github.io/neuromancer/>  
- **NVIDIA PhysicsNeMo:** modern Physics‑AI framework (PINNs, neural operators; containerized).  
  <https://github.com/NVIDIA/physicsnemo> · <https://docs.nvidia.com/physicsnemo/latest/getting-started/installation.html> · <https://developer.nvidia.com/physicsnemo>  
- **TorchPhysics:** mesh‑free PDE learning methods and domain sampling utilities.  
  <https://github.com/boschresearch/torchphysics> · <https://boschresearch.github.io/torchphysics/>  
- **RTAMT:** STL monitoring (offline + online bounded‑future) with Python API and optional C++ backend.  
  <https://github.com/nickovic/rtamt>  
- **MoonLight & STREL:** spatio‑temporal logic monitoring (Java core; Python bindings).  
  <https://github.com/MoonLightSuite/moonlight> · STREL paper (LMCS 2022): <https://lmcs.episciences.org/8936/pdf> · Tool paper: <https://link.springer.com/article/10.1007/s10009-023-00710-5>  
- **SpaTiaL:** object‑centric spatial‑temporal specs and planning.  
  <https://kth-rpl-planiacs.github.io/SpaTiaL/>  
- **STLnet (logic‑guided learning):** NeurIPS 2020 paper that informs our training‑time penalties.  
  <https://proceedings.neurips.cc/paper/2020/file/a7da6ba0505a41b98bd85907244c4c30-Paper.pdf>  
- **PDEBench:** datasets + baselines for PDE surrogates.  
  <https://github.com/pdebench/PDEBench> · <https://arxiv.org/abs/2210.07182>  
- **Traffic (METR‑LA overview):**  
  <https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset>  
- **AirData (EPA):**  
  <https://www.epa.gov/outdoor-air-quality-data/air-data-basic-information>

---

## 10) Requests for feedback

- **Specs:** Are the proposed STL/STREL templates for PDEBench and METR‑LA aligned with what you want to study (safety/stability/propagation)?  
- **Focus:** For the second real‑world dataset, do you prefer **traffic** (road graphs) or **air‑quality** (station grids)?  
- **Depth vs. breadth:** OK to prioritize **two** problem spaces for deep evaluation over touching all three?

---

_Prepared by Arthur Hu (Sep 30, 2025)._
