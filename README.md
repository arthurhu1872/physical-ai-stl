# physical-ai-stl

*A minimal, fast, and reproducible scaffold for experimenting with **Signal Temporal Logic (STL)** and **spatio‑temporal logics (STREL/SpaTiaL)** on physics‑based ML models (a.k.a. “physical AI”): neural ODE/PDE surrogates, PINNs/DeepONets/FNOs, and differentiable predictive control.*

> **Course context.** Built for **Vanderbilt CS‑3860‑01 Undergraduate Research (Fall 2025)** with **Prof. Taylor Thomas Johnson**. Scope: prototype STL/STREL monitoring and (optionally) soft enforcement within physics‑aware ML frameworks; compare a few representative stacks; run on small, CPU‑friendly demos; deliver a concise end‑of‑semester report.

---

## 🔎 Goals (what this repo is for)

1. **Evaluate 3 physics‑ML frameworks** on small PDE/ODE demos:
   - **Neuromancer** (PyTorch SciML; constrained optimization / PINNs / DPC)  
   - **NVIDIA PhysicsNeMo** (ex‑Modulus; neural operators, PINNs; GPU‑optimized but has CPU‑only paths)
   - **Bosch TorchPhysics** (mesh‑free PINNs/DeepRitz/DeepONets/FNO)
2. **Wire up STL/STREL monitoring**:
   - **RTAMT** for STL (offline/online robustness; bounded‑future online)
   - **MoonLight** for **STREL** (spatio‑temporal reach/escape operators; Java engine with Python wrapper)
   - **SpaTiaL** for object‑centric spatio‑temporal specs and simple planning/monitoring
3. **Prototype soft/differentiable STL penalties** to *nudge* models to satisfy specs (no hard guarantees).
4. **Recommend 2–3 problem spaces/datasets** (STL‑friendly) and document integration points.
5. **Produce an end‑of‑semester report** with methodology, results, and recommendations.

> **Weekly cadence per CS‑3860**: ~**6–9 hrs/week** for **3 credits** (≈2–3 hours per credit) and **group meeting Fridays 11:00** (Zoom/in‑person per lab announcement). *One paged living plan + a final report are the required deliverables.*

---

## 🚀 Quickstart (CPU‑only, lean & fast)

Requirements: **Python ≥ 3.10**, macOS/Linux/WSL (Windows works; SpaTiaL is Linux/macOS only).

```bash
# 1) Create venv and install minimal runtime + dev test deps (tiny, <30 MB)
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt -r requirements-dev.txt

# 2) (Optional) Install extras — STL/STREL + physics‑ML stacks (may pull PyTorch)
python -m pip install -r requirements-extra.txt
# If you want *CPU* PyTorch wheels only (to avoid CUDA downloads):
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch

# 3) Smoke tests (safe: will *skip* if optional deps are missing)
pytest -q
# Or run a subset quickly:
pytest -q -k "hello or stl_soft or pde_example"

# (Optional) Makefile helpers:
make -n quickstart   # shows steps
make quickstart      # creates venv + installs + runs fast tests
```

**Docker (fully reproducible, CPU):**
```bash
# Build. WITH_EXTRAS=1 installs STL/STREL + physics stacks (best effort).
docker build --build-arg WITH_EXTRAS=1 -t physical-ai-stl:cpu .
# Run tests inside the container (mirrors CI)
docker run --rm -it physical-ai-stl:cpu
```

---

## 🧭 Repository map

```
src/physical_ai_stl/
  frameworks/         # tiny “hello” demos for Neuromancer / PhysicsNeMo / TorchPhysics
  monitoring/         # RTAMT + MoonLight helpers (and differentiable STL ‘stl_soft’)
  monitors/           # simple spec snippets (STL/STREL) for quick reuse
  datasets/           # small synthetic datasets (e.g., STLnet-style toy generator)
  experiments/        # 1D diffusion, 2D heat (minimal CPU demos)
  physics/            # PDE utilities for toy problems
  training/           # light train/eval scaffolding (grids, seeds)
tests/                # all tests are skip‑aware for optional heavy deps
configs/, scripts/, docs/ (lightweight helpers)
```

**Design for speed & stability**
- *Lean base install* (`requirements.txt`) keeps CI and first‑run fast; heavy stacks live in `requirements-extra.txt`.
- *Optional dependency pattern* throughout (`pytest.importorskip`, lazy imports) lets tests pass even without extras.
- *CPU‑by‑default*; CUDA never auto‑downloads. Torch CPU wheels are used unless you opt into GPU yourself.
- *Determinism* helpers (`utils/seed.py`) for reproducibility on small demos.

---

## 🧪 What already runs (baseline demos)

All demos are tiny; expect seconds on CPU.

- **STL soft penalties** (`monitoring/stl_soft.py`) — smooth min/max & temporal ops; unit tests in `tests/test_stl_soft.py`.
- **RTAMT monitor “hello”** (`monitors/rtamt_hello.py`) — evaluate `G_[a,b](u ≤ u_max)` on short traces; tested in `tests/test_rtamt_hello.py`.
- **MoonLight STREL “hello”** (`monitors/moonlight_hello.py`, `monitors/moonlight_strel_hello.py`) — requires Java 21+; tested with skips if missing.
- **SpaTiaL spec “hello”** (`monitors/spatial_demo.py`) — Linux/macOS only (MONA dependency); tests skip on Windows.
- **Framework smoke tests**:
  - `frameworks/neuromancer_hello.py` (`tests/test_neuromancer_hello.py`)
  - `frameworks/physicsnemo_hello.py` (`tests/test_physicsnemo_hello.py`)
  - `frameworks/torchphysics_hello.py` (`tests/test_torchphysics_hello.py`)
- **Toy PDEs**: 1D diffusion & 2D heat (`experiments/`, `physics/`) with monitors; see `tests/test_pde_example.py`, `tests/test_pde_robustness.py`.

> CI runs `pytest -q` on Ubuntu and only installs the *lean* deps. Optional stacks are tested locally/Docker.

---

## 🧩 Integration patterns (how STL/STREL ties in)

**A. Post‑hoc monitoring** (zero‑risk):  
Run RTAMT (STL) and MoonLight (STREL) on model outputs and log *robustness* values. Use for dashboards, ablations, and safety thresholding.

**B. Soft constraints in training** (lightweight, differentiable, *no guarantees*):  
Use `stl_soft` to approximate STL robustness (smooth min/max via log‑sum‑exp / softplus), then add to loss:
- **Neuromancer**: plug robustness penalties into `nm.loss.PenaltyLoss` (naturally supports constrained training).  
- **TorchPhysics**: add STL penalty into the Lightning loss alongside PDE residuals.  
- **PhysicsNeMo**: add a callback/head computing soft robustness; aggregate into Hydra‑configured loss.

**C. Spatial logic** (STREL/SpaTiaL):  
For PDE fields: encode properties like “**hotspot dissipates within T across the domain**” (STREL reach/escape).  
For object/sensor graphs: specify relations like “**no more than K sensors within radius R below speed threshold for >Δt**” (SpaTiaL).

---

## 📚 Frameworks & tooling at a glance

| Component | What it is (one‑liner) | Install notes |
|---|---|---|
| **Neuromancer** | PyTorch SciML library for constrained optimization, PINNs, DPC | `pip install neuromancer` or clone; docs & examples provided. |
| **PhysicsNeMo** | NVIDIA’s (ex‑Modulus) Physics‑AI stack; neural operators/PINNs with Hydra tooling | `pip install nvidia-physicsnemo` or use NGC container; CPU ok, GPU recommended for big models. |
| **TorchPhysics** | Mesh‑free PDE library implementing PINNs/DeepRitz/DeepONets/FNO | `pip install torchphysics` (needs PyTorch ≥ 2.0). |
| **RTAMT** | STL monitoring (offline & bounded‑future online) with robustness | `pip install rtamt` (C++ backend optional). |
| **MoonLight (STREL)** | Java engine + Python pkg for spatio‑temporal monitoring (STREL) | `pip install moonlight` + **Java 21+**. |
| **SpaTiaL** | Spatio‑temporal object relations + planning/monitoring | `pip install spatial-spec` (+ MONA/`ltlf2dfa`; **Linux/macOS**). |

**References:** Neuromancer repo & docs; NVIDIA PhysicsNeMo repo & install guide; TorchPhysics repo & docs; RTAMT repo; MoonLight repo (STREL support, Java 21+); SpaTiaL PyPI & paper. Full links are in **References** below.

---

## 🗺️ Candidate problem spaces / datasets (STL‑friendly)

Pick **one primary** + **one backup** early in the semester.

1) **Toy PDE fields (deterministic, tiny)** — *fastest path to results*  
   - **1D diffusion / 2D heat** with synthetic initial conditions.  
   - Example specs:  
     - `G_[0,T] (u(x,t) ≤ u_max)` (safety bound)  
     - `F_[0,T] G_[0,τ] (|∇u| ≤ L)` (eventual spatial smoothness)  
   - Runs entirely on CPU in seconds.

2) **2D Darcy flow (neural operator baseline)** — *PhysicsNeMo Sym example*  
   - Use FNO tutorial datasets (per docs) and monitor pressure/flux bounds or boundary conditions.  
   - Example specs: `G_[0,T] (p_max − p(x,t) ≥ 0 ∧ p(x,t) − p_min ≥ 0)`; `G_[0,T] (‖∇p‖ ≤ c)`.

3) **Urban sensor networks (traffic speed)** — *graph sensors, natural STREL*  
   - **METR‑LA** (207 sensors, 5‑minute speeds, Mar–Jun 2012) and **PEMS‑BAY** (325 sensors, Jan–May 2017).  
   - Example specs: “No cascade: *not* (≥ 10 sensors within 2‑hop radius below 20 mph for ≥ 15 min)” or “Congestion waves propagate slower than `v_max` across adjacency.”

> Alternative: **Air‑quality sensors** (e.g., PM2.5 city networks) with STREL “surround/reach” constraints (see STLnet and related smart‑city work).

---

## 🗓️ Suggested semester plan (draft)

- **Weeks 1–2**: Environment ready; run all “hello” tests; short evaluation notes for the 3 frameworks + 3 logic toolchains.  
- **Weeks 3–4**: Choose **primary dataset**; reproduce a tiny baseline (e.g., Darcy FNO tutorial or diffusion PINN). Define **2–3 specs**.  
- **Weeks 5–6**: Post‑hoc monitoring + plots; sensitivity to spec thresholds.  
- **Weeks 7–9**: Add **soft STL penalties** in training; ablations vs. baselines (accuracy, robustness).  
- **Weeks 10–12**: Add **spatial** specs (MoonLight/SpaTiaL) for the chosen domain.  
- **Week 13+**: Polish figures, **final report**, and repository artifacts.

**Deliverables**
- *Running code & configs* for one primary experiment;  
- *Short written report* (problem, specs, methods, results, discussion, limits);  
- *Comparison table* for stacks & tooling;  
- *Reproducibility checklist* (env, seed, dataset pointers).

---

## 🔧 Usage snippets

**Evaluate an STL spec on a short trace with RTAMT**
```python
from physical_ai_stl.monitoring.rtamt_monitor import stl_always_upper_bound, evaluate_series
spec = stl_always_upper_bound("u", u_max=1.0)   # G(u <= 1.0)
rob  = evaluate_series(spec, "u", [0.2, 0.4, 1.1], dt=1.0)  # -> negative means violation magnitude
```

**Soft STL penalty in training (framework‑agnostic)**
```python
from physical_ai_stl.monitoring import stl_soft as S
# Soft robustness: G_[0,T] (u <= umax)
def soft_G_always_leq(u, umax):
    return S.soft_min(umax - u)   # vectorized; larger is better; negative violates

penalty = (-soft_G_always_leq(u_pred, umax)).relu().mean()  # add to loss
```

> These “soft” semantics are differentiable approximations for convenience; they **do not** provide formal guarantees.

---

## 🧱 Reproducibility & performance tips

- Prefer **CPU** runs first; turn on GPU only after correctness.  
- For MoonLight (STREL), ensure **Java 21+** is installed and on `PATH`.  
- SpaTiaL relies on **MONA/ltlf2dfa** and is best on Linux/macOS (it is skipped on Windows in tests).  
- Keep datasets **small** (e.g., 16–64² grids, 100–500 samples) for rapid iteration; scale later.  
- Use `pytest -q -k ...` to run focused subsets; CI mirrors this minimal footprint.

---

## 🧾 References (primary docs & datasets)

- **Neuromancer** — Repo & docs (features, install, examples).  
  - GitHub: https://github.com/pnnl/neuromancer  
  - Docs: https://pnnl.github.io/neuromancer/

- **NVIDIA PhysicsNeMo** — Repo & install guide (ex‑Modulus; CPU/GPU; tutorials e.g., **Darcy FNO**).  
  - GitHub: https://github.com/NVIDIA/physicsnemo  
  - Install: https://docs.nvidia.com/physicsnemo/latest/getting-started/installation.html  
  - Sym tutorial (Darcy FNO): https://docs.nvidia.com/physicsnemo/latest/physicsnemo-sym/user_guide/neural_operators/darcy_fno.html

- **TorchPhysics** — Repo & docs (PINNs/DeepRitz/DeepONets/FNO).  
  - GitHub: https://github.com/boschresearch/torchphysics  
  - Docs: https://boschresearch.github.io/torchphysics/

- **RTAMT** — STL monitoring (robustness; bounded‑future online).  
  - GitHub: https://github.com/nickovic/rtamt

- **MoonLight (STREL)** — Java engine + Python package; **requires Java 21+**.  
  - GitHub: https://github.com/MoonLightSuite/moonlight

- **SpaTiaL** — Spatio‑temporal specifications; Linux/macOS recommended (MONA/`ltlf2dfa`).  
  - PyPI: https://pypi.org/project/spatial-spec/  
  - Paper: https://link.springer.com/article/10.1007/s10514-023-10145-1

- **STLnet** — NeurIPS 2020 paper enforcing STL in sequence models; relevant for STL‑guided training ideas.  
  - Abstract/PDF: https://proceedings.neurips.cc/paper/2020/file/a7da6ba0505a41b98bd85907244c4c30-Paper.pdf

- **Physics datasets** (starter options)  
  - **PDEBench** (DarUS archive & toolkit): https://github.com/pdebench/PDEBench  
  - **FNO datasets** (Burgers/Darcy/NS): https://github.com/li-Pingan/fourier-neural-operator

- **Traffic datasets** (sensor networks)  
  - **METR‑LA**/**PEMS‑BAY** summaries: https://arxiv.org/abs/2403.16495  
  - LargeST (large‑scale, if needed): https://proceedings.neurips.cc/paper_files/paper/2023/file/ee57cd73a76bd927ffca3dda1dc3b9d4-Paper-Datasets_and_Benchmarks.pdf

---

## 📝 Contributing / issues

PRs and issues welcome (tests are fast and skip optional stacks automatically). Please:
- keep base installs lean; gate heavy deps behind `requirements-extra.txt`;
- add tests (`pytest`) and short docs for new monitors/experiments.

---

## 📄 License & citation

- **MIT License** (see [LICENSE](LICENSE)).  
- If you use this scaffold, please cite via **[CITATION.cff](CITATION.cff)**.

> Acknowledgments: thanks to Prof. Taylor T. Johnson for guidance on **physical AI** directions and to the Neuromancer, PhysicsNeMo, TorchPhysics, RTAMT, MoonLight, and SpaTiaL communities for their open‑source work.
