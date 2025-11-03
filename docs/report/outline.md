# Report Outline — Physical AI × (Spatial) Signal Temporal Logic

> **Course:** Vanderbilt **CS‑3860‑01 Undergraduate Research** (3 credits)  
> **Instructor:** Prof. Taylor T. Johnson  
> **Student:** Arthur Hu  
> **Repo:** [`physical-ai-stl`](../../README.md) · *living document; updated as results arrive.*

---

## 0) Abstract (write last)
**≤150 words template.** We study whether *specification‑level* feedback improves safety and reliability of physics‑aware ML (“physical AI”) for cyber‑physical systems (CPS). We integrate **Signal Temporal Logic (STL)** and **spatio‑temporal** extensions (e.g., **STREL**) into physics‑ML training/evaluation loops, using small ODE/PDE demos and real spatio‑temporal datasets. We compare three representative frameworks (Neuromancer, NVIDIA PhysicsNeMo, TorchPhysics), prototype differentiable STL losses, and use runtime monitors (RTAMT; MoonLight) for ground‑truth robustness. Experiments measure task error, specification robustness, and compute cost; we report Pareto trade‑offs and ablations. Findings: (i) monitor‑only diagnostics catch failures early; (ii) soft STL penalties can raise robustness with modest error/compute overhead when specs are aligned with the task; (iii) spatial logic helps contain undesirable phenomena in PDE‑style fields. We release a CPU‑first, reproducible artifact and guidelines for STL/STREL in physics‑ML.

---

## 1) Introduction
### Motivation
Physics‑ML—PINNs, DeepONets, neural operators, differentiable predictive control—offers compact models and rapid what‑if simulation for CPS. But *requirements* must be stated and checked at the level users care about (safety envelopes, response windows, containment), not only via MSE or residuals. **STL** provides machine‑checkable temporal properties with **quantitative robustness**; **STREL** extends these to spatial graphs/grids for PDE‑like fields.

### Goal
Build a **lightweight, reproducible** pipeline that (a) **monitors** STL/STREL properties during training/eval and (b) optionally **enforces** them via differentiable penalties—then **compare** across frameworks and tasks.

### Contributions (targeted)
1) **Integration design** linking physics‑ML stacks with STL/STREL monitors (reference code, minimal APIs).  
2) **Framework comparison**: Neuromancer, PhysicsNeMo, TorchPhysics on the same “STL‑ready” tasks.  
3) **Benchmarks**: small ODE/PDE demos and real spatio‑temporal signals with curated specs.  
4) **Empirical study** of **robustness vs. task error vs. compute**, including key ablations.  
5) **Reproducible artifact** (CPU‑first) with tests, scripts, and fixed seeds.

---

## 2) Background & tools (concise survey)
- **STL (Signal Temporal Logic).** Temporal operators `G` (always), `F` (eventually), `U` (until); dense/discrete time; **robustness semantics** enabling optimization and sensitivity; offline and bounded‑future online monitors. We use **RTAMT** for authoritative robustness and slicing (Python API; optional C++ backend).  
- **STREL (spatio‑temporal).** Adds *reach*/*escape* over graph neighborhoods to reason about propagation/containment in spatially distributed CPS; we use **MoonLight** (tool + Python bindings) for runtime monitoring on grid/graph signals.  
- **Object‑centric spatial relations (optional).** **SpaTiaL** specifies relations among discrete entities (e.g., “A left‑of B within τ”), useful for agent/scene abstractions.  
- **Physics‑ML frameworks.**  
  - **Neuromancer** — PyTorch differentiable programming for constrained optimization, system ID, and **Differentiable Predictive Control (DPC)**.  
  - **NVIDIA PhysicsNeMo** — open‑source Physics‑ML with **PINNs** and **neural operators** (e.g., FNO, DeepONet), strong GPU support, but CPU paths available.  
  - **TorchPhysics** — mesh‑free PDE learning (PINNs, Deep Ritz, DeepONet, FNO); lightweight and approachable.

> References for the above are given at the end; we cite official repos/docs and tool papers.

---

## 3) Problem formulation
### Signals
- State \(u(x,t)\in\mathbb{R}^m\) on grid \(\Omega\subset\mathbb{R}^d\) and time horizon \[0,T].  
- Observables \(y = h(u)\) (scalars or vector fields: temperature, velocity magnitude, PM2.5, speed).

### Specifications
- **Temporal (STL)** examples  
  - **Safety bound** — \(\mathbf G_{[0,T]}(y\le U_{\mathrm{safe}})\).  
  - **Recovery** — after an event, \(\mathbf F_{[0,\tau]}\,\mathbf G_{[0,T]}(y\le U_{\mathrm{safe}})\).  
- **Spatio‑temporal (STREL)** on grid graph \(G=(V,E)\)  
  - **Containment** — “hotspot stays within radius r.”  
  - **No‑leak** — “high‑value region never reaches boundary within \[0,τ].”

### Robustness & use in ML
- Quantitative robustness \(\rho(\varphi,w,t)\) serves as:  
  (i) a **diagnostic** metric (monitor‑only), and  
  (ii) a **loss shaper** \(\mathcal L_{\mathrm{STL}}=\mathrm{softAgg}(\rho)\) using smooth min/max for backprop.

---

## 4) System design & implementation plan
**Data flow.**  
**Model** (PyTorch/Neuromancer/PhysicsNeMo/TorchPhysics) → **Spec** (STL via RTAMT; STREL via MoonLight) → **Monitors** produce robustness traces → **Trainer** optionally adds **differentiable STL loss**.

**Modules (repo pointers).**
- `monitoring/stl_soft.py` — *differentiable STL core*: `softmin/softmax`, sliding windows, `G`/`F`/`U`; unit‑tested against RTAMT on toy traces.  
- `monitoring/rtamt_monitor.py` — RTAMT wrapper for discrete/dense time monitors; provides “ground‑truth” robustness for evaluation and ablations.  
- `monitoring/moonlight_helper.py` — MoonLight bridge: grid/graph builders, `.mls` spec launcher, result parsing; examples under `scripts/specs/`.  
- `frameworks/*` — Tiny “hello” demos for Neuromancer / PhysicsNeMo / TorchPhysics to standardize inputs/outputs.  
- `experiments/*` — 1D diffusion and 2D heat demos; CLI scripts in `scripts/`.

**Efficiency tactics.**
- CPU‑first defaults; optional GPU guarded behind extras.  
- Monitor **subsampling** (space/time) and **evaluate‑every‑k** epochs to amortize runtime.  
- Vectorized robustness ops; numerically stable **log‑sum‑exp** aggregations.  
- Smoke tests complete in **≤2 minutes on CPU**; full runs kept **<~30 minutes**.

---

## 5) Experimental plan
### 5.1 Tasks / datasets (STL‑ready)
| Tier | Task | Why | Example specs | Primary stack(s) |
|---:|---|---|---|---|
| **T1** | **1D diffusion** (synthetic) | Deterministic, seconds‑fast | \(\mathbf G(y\le U_{\rm safe})\), response‑time cooling | PyTorch (+soft STL), Neuromancer |
| **T1** | **2D heat** (grid) | Natural **STREL** demo | **Containment** of hotspot; **no‑leak** to boundary | PyTorch (+MoonLight), TorchPhysics |
| **T2** | **Burgers/Advection** (*PDEBench* mini) | Standard PDE operators | amplitude bounds; wavefront speed limits | PhysicsNeMo, TorchPhysics |
| **T2** | **Air quality (multi‑site)** | Real spatio‑temporal signals | “PM2.5 stays below threshold; violations short‑lived” | Neuromancer (+RTAMT) |
| **Stretch** | **Traffic (METR‑LA/PEMS‑BAY)** | Graph‑temporal forecasting | “No corridor exceeds drop for >τ” | Neural operators (PhysicsNeMo/TorchPhysics) |

*(Selection rationale: public, small‑enough to run on CPU; readily specifiable; has precedent in logic‑aware ML.)*

### 5.2 Models & baselines
- **Baselines**: task loss only (MSE; PINN residual).  
- **STL loss variants**: temperature τ for softmin; spatial aggregation (mean vs softmax); monitor cadence (every k epochs).  
- **Framework ablations**: same task/spec across Neuromancer vs PhysicsNeMo vs TorchPhysics.

### 5.3 Metrics
- **Task quality:** MSE/MAE; physics residual (PINN).  
- **Spec quality:** avg/min **robustness**, **violation rate**, **time‑to‑compliance**.  
- **Efficiency:** time/epoch, peak RAM, #params/steps.  
- **Pareto:** robustness vs task loss.

### 5.4 Reproducibility
- Fixed seeds; deterministic configs; env captured in `requirements*.txt` and `docs/REPRODUCIBILITY.md`.  
- CLI scripts: `scripts/train_diffusion_stl.py`, `scripts/train_heat2d_strel.py`, `scripts/train_neuromancer_stl.py`.  
- CI‑style smoke tests under `tests/` keep examples quick and portable.

---

## 6) Results (figure plan)
1. **Table** — Framework × task × metrics (MSE, robustness, time).  
2. **Curves** — robustness vs epoch; **Pareto** (robustness vs task loss).  
3. **Spatial heatmaps** — violation maps on 2D grids; MoonLight robustness trace.  
4. **Ablations** — softmin temperature; monitor cadence; spatial aggregation.  
5. **Efficiency** — time/epoch vs monitor frequency; (optional) CPU vs GPU.

---

## 7) Discussion
- **When STL helps** and when it harms task error or compute; spec/task alignment.  
- **Specification engineering** — writing meaningful properties; handling false positives; monitor‑loss mismatch.  
- **Scalability paths** — neural operators and batching monitors; sparse graphs for STREL.  
- **Threats to validity** — smooth approximations; dataset bias; discretization artifacts.

---

## 8) Conclusion & next steps
- **Summary** — STL/STREL monitoring as a practical guardrail for physics‑ML on small CPS‑style tasks.  
- **Future** — closed‑loop control with STL inside DPC (Neuromancer), controller verification (NNV‑style), richer STREL (surround/escape) and multi‑agent grids.

---

## 9) Practical plan (for CS‑3860)
- **Cadence** — group meeting **Fridays 11:00** (per lab announcement); short async updates via repo Issues.  
- **Hours** — ~6–9 hrs/week for 3 credits; front‑loaded reading in Weeks 1–2.  
- **Milestones** *(adjust as needed)*  
  - **Week 1–2** — run *hello* demos; finalize tasks/specs; write minimal monitors.  
  - **Week 3–4** — implement differentiable STL core; RTAMT parity tests.  
  - **Week 5–6** — add MoonLight STREL demo on 2D heat; first ablations.  
  - **Week 7–8** — framework comparison on T1 tasks; draft results tables.  
  - **Week 9–10** — T2 dataset; efficiency sweeps; write discussion.  
  - **Finals** — polish report; release artifact.

---

## 10) Mapping to instructor requirements (✅ = covered)
- Evaluate **Neuromancer / PhysicsNeMo / TorchPhysics** → §§2, 5–6 ✅  
- Integrate **STL/STREL monitoring** (RTAMT, MoonLight; SpaTiaL optional) → §§2, 4–5 ✅  
- Identify **problem spaces/datasets** from the **STL angle** → §5.1 + dataset notes ✅  
- Produce **end‑of‑semester report** → this outline + figure plan ✅

---

## References (selection, to cite in the final report)
- **Neuromancer** — repo + docs.  
- **NVIDIA PhysicsNeMo** — repo + docs (PINNs, neural operators, distributed).  
- **TorchPhysics** — repo + site (PINNs/Deep Ritz/DeepONet/FNO).  
- **RTAMT** — real‑time STL monitoring (Python/C++).  
- **MoonLight** — STREL runtime monitoring; Python bindings.  
- **STREL logic** — Nenzi et al., LMCS (2022).  
- **STLnet** — NeurIPS 2020 paper + code.  
- **PDEBench** — NeurIPS 2022 datasets/benchmarks.  
- **PINNs/DeepONet** — foundational papers for physics‑ML.  

(Inline hyperlinks live in the repository README and dataset notes; we will reference primary sources and official docs in the final write‑up.)
