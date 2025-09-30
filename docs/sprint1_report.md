# Sprint 1 Report — Physical AI × STL
**Course:** CS‑3860‑01 Undergraduate Research (3 credits)  
**Instructor:** Prof. Taylor Thomas Johnson  
**Student:** Arthur Hu  
**Date:** September 30, 2025

---

## 1) Objective & context

Per our email exchange and your guidance, the initial goal is to **evaluate a small set of “physical AI” frameworks** and **identify problem spaces/datasets** where we can **monitor and/or enforce Signal Temporal Logic (STL) and spatial STL (STREL)** specifications. The near‑term deliverable was a lightweight scaffold plus a written comparison and concrete dataset recommendations to enable quick prototyping in Sprint 2.

Lab cadence: group meeting Fridays 11:00; workload target ≈ **6–9 hours/week** (3 credits). *(Noted 2025‑09‑30.)*

---

## 2) What shipped in Sprint 1

A minimal, reproducible scaffold for experiments (this repository) with fast CPU‑only demos and optional heavier stacks. Key pieces now in place:

- **Project skeleton & environment**
  - `pyproject.toml`, `requirements.txt` (minimal NumPy only), optional `requirements-extra.txt` (installs RTAMT, MoonLight, SpaTiaL, Neuromancer, TorchPhysics, PhysicsNeMo), and `requirements-dev.txt` for tests.
  - **Reproducibility:** Dockerfile (CPU), `Makefile` helpers, deterministic seeds (`src/physical_ai_stl/utils/seed.py`), and `docs/REPRODUCIBILITY.md`.
- **Toy PDE & monitoring hooks (for smoke tests)**
  - `src/physical_ai_stl/physics/{diffusion1d.py, heat2d.py}` and `experiments/{diffusion1d.py, heat2d.py}` — tiny NumPy/PyTorch‑ready solvers/scaffolds for 1D diffusion and 2D heat.
  - `src/physical_ai_stl/monitors/{stl_soft.py, rtamt_monitor.py, moonlight_helper.py}` — soft robustness primitives (differentiable min/max via log‑sum‑exp), plus adapters to RTAMT and MoonLight monitors for offline checks.
- **Hello‑world integration stubs**
  - `frameworks/{neuromancer_hello.py, torchphysics_hello.py, physicsnemo_hello.py}`  
  - `monitors/{rtamt_hello.py, moonlight_strel_hello.py}` and `scripts/specs/*.stl` (example formulas).
- **Unit tests (quick, CPU)**
  - `tests/` contains small smoke tests for the PDE stubs, soft STL penalties, and each “hello” path; default run keeps everything <1–2 minutes on CPU.

> The emphasis this sprint was scaffolding and **evaluation groundwork**; the first end‑to‑end training experiments with STL/STREL penalties land in Sprint 2.

---

## 3) Frameworks evaluated (decision for Sprint 2)

Below is a concise comparison focused on **fit for STL/STREL‑guided learning**, small‑to‑medium PDE demos, ease of setup, and PyTorch interoperability.

### 3.1 Summary table

| Framework | Core strengths (from official docs) | Practical notes for this project | Verdict |
|---|---|---|---|
| **Neuromancer** [1, 2] | PyTorch‑based **differentiable programming** for parametric constrained optimization, **physics‑informed system ID**, and **model‑based optimal control**. | Clean PyTorch interop; natural place to explore constrained training and differentiable predictive control with STL penalties as soft constraints. | **Keep** as option for control/system‑ID experiments; not the first target for PDE benchmarks. |
| **NVIDIA PhysicsNeMo** [3, 4, 5] | Framework for **Physics AI** with SciML methods; modules for **symbolic PDE residuals** and domain sampling; sub‑stacks for **CFD** and deployment. | Powerful, but heavier; best on Linux/NVIDIA stack. Excellent candidate once we scale to neural operators or larger PDE suites (e.g., Navier–Stokes). | **Defer** to later sprints; use if we need GPU and operator models at scale. |
| **Bosch TorchPhysics** [6, 7] | Mesh‑free deep‑learning methods for ODE/PDE (e.g., **PINNs**, **Deep Ritz**, operator learning) with a simple, PyTorch‑native API. | Lightest setup; good examples; fast to extend with STL penalties in training loops. | **Choose for Sprint 2.** Ideal for quick PINN/Burgers/heat experiments with STL regularization. |

### 3.2 Rationale
- **TorchPhysics first:** fastest path to a working **STL‑regularized PINN** on 1D Burgers / 2D heat with clean PyTorch hooks.  
- **Neuromancer second:** revisit for **constrained training/DPC** once STL penalties are stable.  
- **PhysicsNeMo later:** bring in for **neural operators / CFD** when we need performance and deployment‑oriented tooling.

---

## 4) STL / STREL tooling

| Tool | What it provides | Why it’s relevant here |
|---|---|---|
| **RTAMT** [8–10] | Python library for offline and online monitoring of STL (discrete & dense time); quantitative robustness; C++‑accelerated back‑ends; ROS integration in variants. | Reference monitor for **ground‑truth evaluation** during/after training; also drives non‑differentiable validation metrics. |
| **MoonLight (STREL)** [11–13] | Java tool with Python/Matlab interfaces for **Spatio‑Temporal Reach and Escape Logic (STREL)** and standard temporal logic; monitors properties over spatial graphs and fields. | Enables **spatial** specifications (e.g., hotspot containment, neighborhood constraints) crucial for PDE fields and gridded sensors. |
| **SpaTiaL** [14–16] | Research framework for specifying **spatial–temporal relations** in robotics; supports monitoring/planning; Python packages (`spatial-spec`, `spatial`) distributed via PyPI/source. | Useful to prototype **object‑centric** spatio‑temporal specs; complementary to STREL and classic STL for certain tasks. |

*Differentiable STL:* for learning, we use smooth (log‑sum‑exp) approximations of min/max to create **gradient‑friendly robustness penalties** (cf. smoothed STL robustness used in controller synthesis literature). We retain RTAMT/MoonLight for **exact monitoring** at evaluation time.

---

## 5) Candidate problem spaces & datasets (from the STL/STREL angle)

### A. Canonical PDEs (synthetic, fast, controllable)
1) **1D diffusion / 2D heat** — quick baselines already scaffolded here.  
   **STL/STREL examples:**  
   - Safety: \(\mathbf G_{[t_0,t_1]}\, (T(x,t) \in [L, U])\) (global bound in time and space).  
   - Recovery: \(\mathbf G\big(\text{spike} \Rightarrow \mathbf F_{\le \tau}\, T(x,t) < L\big)\).  
   - Spatial containment (STREL): hotspots must **not spread** beyond a radius (reach/escape operators).

2) **Burgers’ equation (1D)** — standard PINN demo (shocks).  
   **Specs:** bound the velocity; enforce **settling‑time** after shocks; spatial gradient constraints via STREL on 1D lattice.

*Why A/B?* Deterministic, tiny, and ideal for ablating STL penalty strength and smoothing temperature with minimal compute.

### B. Public PDE benchmarks (scalable, realistic)
- **PDEBench** suites — diffusion–reaction, Darcy, shallow water, **Navier–Stokes (2D/3D)** with HDF5 data and standard ML baselines. [17–20]  
  **Specs:** e.g., *No‑flooding*: water height never exceeds \(h_{\max}\) over critical cells (STREL), or *Energy does not increase* after \(t_0\).

- **FNO Navier–Stokes data** used widely for operator learning (Darcy/Burgers/Navier–Stokes). [21, 22]  
  **Specs:** vorticity bounds; eventual decay; spatial smoothness within neighborhood radius.

### C. Spatio‑temporal sensor networks (discrete graphs)
- **METR‑LA / PEMS‑BAY** traffic speed datasets (graph time series). [23–25]  
  **Specs:** *Throughput within bounds on corridors*, *No‑jam persists beyond \(\tau\)*; **spatial** *no more than k adjacent sensors exceed threshold simultaneously* (STREL neighborhood).

### D. STLnet‑style city signals (time series)
- **Air‑quality / urban sensing** tasks from STLnet (RNNs with STL‑guided training). [26, 27]  
  **Specs:** *If PM2.5 spikes, it must drop below threshold within 60 min and stay there for 2 hours*.

**Recommendation:** Start Sprint 2 on **(A)** with TorchPhysics + soft‑STL penalty; add **(B)** (PDEBench) next; consider **(C)**/**(D)** if time permits for diversity.

---

## 6) Minimal integration design (training with STL penalty)

**Loss = task loss + λ × soft‑violation.** We compute a differentiable robustness surrogate \(\tilde\rho\) (negative if the spec is violated) and penalize \(\max(0, -\tilde\rho)\). Exact RTAMT/MoonLight monitors remain in the loop for evaluation.

```python
# Pseudocode (PyTorch), used identically with TorchPhysics models
for batch in loader:
    pred = model(batch.inputs)                     # PDE surrogate (PINN/operator/NN)
    task_loss = mse(pred, batch.targets)           # or PDE residuals, BCs, ICs
    rob = soft_stl_robustness(pred, batch.t, batch.x, spec)  # log-sum-exp min/max
    loss = task_loss + lam * torch.relu(-rob)      # penalize violations
    loss.backward(); opt.step(); opt.zero_grad()
```

**Specs library.** We maintain simple spec templates (YAML/JSON) for common properties (bounds, recovery, spatial containment) and compile them to both:  
(i) differentiable surrogates for training, and (ii) RTAMT/MoonLight monitors for evaluation parity.

---

## 7) Metrics & experiments plan

- **Task metrics:** PDE residual norms, MSE vs reference, operator error.  
- **Logic metrics:** average robustness, % satisfying traces, worst‑case robustness.  
- **Compute metrics:** wall‑clock per epoch, peak memory.  
- **Ablations:** λ (penalty weight), smoothing temperature (softmin/softmax), spec variants, sampling of space–time points.

**Baselines:** model without STL penalty; penalty with *incorrect* spec (sanity check); exact vs soft robustness correlation (calibration).

---

## 8) Risks & mitigations

- **MoonLight/JDK friction** (Java dependency). *Mitigation:* ship helper scripts, or run only as a post‑hoc check on saved trajectories.  
- **Non‑differentiable monitors** (exact STL) unsuitable for gradient‑based training. *Mitigation:* smooth surrogates for training; exact for validation.  
- **PhysicsNeMo stack weight** (GPU/driver). *Mitigation:* scope to CPU‑friendly TorchPhysics until we truly need operators/CFD.

---

## 9) What’s next (Sprint 2 checklist)

1. **Reproduce a TorchPhysics PDE** (Burgers 1D or heat 2D).  
2. **Implement STL penalty**: bounds + recovery; add spatial containment (STREL) on the 2D grid.  
3. **Run ablations** over λ and smoothing temperature; record robustness vs accuracy trade‑offs.  
4. **Select PDEBench case** to scale (e.g., 2D diffusion–reaction); prepare data loaders.  
5. **Document** scripts, configs, and seeds; keep all CPU‑runnable (<~15 min) for review.

---

## References

**Frameworks**  
[1] PNNL **Neuromancer** (GitHub). https://github.com/pnnl/neuromancer  
[2] Neuromancer documentation. https://pnnl.github.io/neuromancer/  
[3] NVIDIA **PhysicsNeMo** (GitHub). https://github.com/NVIDIA/physicsnemo  
[4] PhysicsNeMo **Symbolic** module (GitHub). https://github.com/NVIDIA/physicsnemo-sym  
[5] PhysicsNeMo **CFD** module (GitHub). https://github.com/NVIDIA/physicsnemo-cfd  
[6] Bosch **TorchPhysics** (GitHub). https://github.com/boschresearch/torchphysics  
[7] TorchPhysics documentation. https://boschresearch.github.io/torchphysics/

**STL / STREL tooling**  
[8] **RTAMT** (GitHub). https://github.com/nickovic/rtamt  
[9] RTAMT (PyPI). https://pypi.org/project/rtamt/  
[10] D. Ničković et al., “RTAMT: Online Robustness Monitors from STL,” *arXiv*, 2020. https://arxiv.org/pdf/2005.11827  
[11] **MoonLight** (GitHub). https://github.com/MoonLightSuite/moonlight  
[12] E. Bartocci et al., “MoonLight: A Lightweight Tool for Monitoring Spatio‑Temporal Properties,” *arXiv*, 2021. https://arxiv.org/abs/2104.14333  
[13] STREL = **Spatio‑Temporal Reach and Escape Logic** (intro slides/paper). https://moodle2.units.it/pluginfile.php/589724/mod_resource/content/1/16-STREL.pdf

**Datasets & problem spaces**  
[14] **SpaTiaL** overview/org. https://github.com/KTH-RPL-Planiacs  
[15] C. Pek et al., “SpaTiaL: Monitoring and Planning of Robotic Tasks Using Spatio‑Temporal Logic Specifications,” *Autonomous Robots*, 2023. https://link.springer.com/article/10.1007/s10514-023-10145-1  
[16] `spatial-spec` (PyPI). https://pypi.org/project/spatial-spec  
[17] **PDEBench** (GitHub). https://github.com/pdebench/PDEBench  
[18] PDEBench dataset (DaRUS). https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986  
[19] M. Takamoto et al., “PDEBench,” *NeurIPS Datasets & Benchmarks*, 2022. https://proceedings.neurips.cc/paper_files/paper/2022/hash/0a9747136d411fb83f0cf81820d44afb-Abstract-Datasets_and_Benchmarks.html  
[20] PDEBench paper (arXiv PDF). https://arxiv.org/abs/2210.07182  
[21] Z. Li et al., “Fourier Neural Operator for Parametric PDEs,” 2020. https://arxiv.org/abs/2010.08895  
[22] FNO datasets/code pointers. https://github.com/li-Pingan/fourier-neural-operator  
[23] **DCRNN** repo (METR‑LA/PEMS‑BAY data links). https://github.com/liyaguang/DCRNN  
[24] METR‑LA (Kaggle mirror). https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset  
[25] METR‑LA write‑ups/usage. https://medium.com/stanford-cs224w/traffic-forecasting-with-directed-gnns-608da078e1a1  

**Prior STL‑guided learning**  
[26] M. Ma et al., “STLnet: Signal Temporal Logic Enforced Multivariate RNNs,” *NeurIPS*, 2020. https://proceedings.neurips.cc/paper/2020/file/a7da6ba0505a41b98bd85907244c4c30-Paper.pdf  
[27] STLnet abstract page. https://proceedings.neurips.cc/paper/2020/hash/a7da6ba0505a41b98bd85907244c4c30-Abstract.html
