# physical-ai-stl

*A minimal, fast, and reproducible scaffold for experimenting with **Signal Temporal Logic (STL)** and **spatio‑temporal logics (STREL/SpaTiaL)** on physics‑based ML models (a.k.a. “physical AI”): neural ODE/PDE surrogates, PINNs/DeepONets/FNOs, and differentiable predictive control.*

> **Course context.** Built for **Vanderbilt CS‑3860‑01 Undergraduate Research (Fall 2025)** with **Prof. Taylor Thomas Johnson**. Scope: prototype STL/STREL monitoring and (optionally) soft enforcement within physics‑aware ML frameworks; compare a few representative stacks; run on small, CPU‑friendly demos (1‑D diffusion, 2‑D heat, Burgers, Neuromancer); and deliver a concise end‑of‑semester report with several concrete examples.

---

## 🔎 Goals (what this repo is for)

1. **Evaluate 3 physics‑ML frameworks** on small PDE/ODE demos:
   - **Neuromancer** (PyTorch SciML; constrained optimization / PINNs / DPC)
   - **NVIDIA PhysicsNeMo** (ex‑Modulus; neural operators, PINNs; GPU‑optimized but has CPU‑only paths)
   - **Bosch TorchPhysics** (mesh‑free PINNs/DeepRitz/DeepONets/FNO)
2. **Wire up STL/STREL monitoring**:
   - **RTAMT** for STL (offline **and** online robustness; online supports bounded‑future)
   - **MoonLight** for **STREL** (spatio‑temporal reach/escape operators; Java engine + Python wrapper)
   - **SpaTiaL** for object‑centric spatio‑temporal specs and simple planning/monitoring
3. **Prototype soft/differentiable STL penalties** to *nudge* models to satisfy specs (no hard guarantees).
4. **Recommend 2–3 problem spaces/datasets** (STL‑friendly) and document integration points.
5. **Produce an end‑of‑semester report** with methodology, results, and recommendations.

> **Weekly cadence per CS‑3860**: ~**6–9 hrs/week** for **3 credits** (≈2–3 hours per credit) and **group meeting Fridays 11:00** (Zoom/in‑person per lab announcement). *One‑page living plan + a final report are the required deliverables.*

---

## 🚀 Quickstart (CPU‑only, lean & fast)

Requirements: **Python ≥ 3.10**, macOS/Linux/WSL (Windows works; SpaTiaL’s automaton‑based planner is Linux/macOS only).

```bash
# 1) Create venv and install minimal runtime + dev test deps (tiny install)
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts ctivate
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

For full, step‑by‑step reproducibility (environment probe, experiment commands, artifact layout), see **[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)**.

---

## 🧭 Repository map

```
src/physical_ai_stl/
  frameworks/         # tiny “hello” demos for Neuromancer / PhysicsNeMo / TorchPhysics
  monitoring/         # RTAMT + MoonLight helpers (and differentiable STL ‘stl_soft’)
  monitors/           # simple spec snippets (STL/STREL) for quick reuse
  datasets/           # small synthetic datasets (e.g., STLnet‑style toy generator)
  experiments/        # 1D diffusion, 2D heat, Burgers (minimal CPU demos)
  physics/            # PDE utilities for toy problems
  training/           # light train/eval scaffolding (grids, seeds)
tests/                # all tests are skip‑aware for optional heavy deps
scripts/              # CLI entrypoints (train/eval, env probe, plotting, specs)
configs/              # YAML configs for run_experiment.py and friends
docs/                 # reproducibility guide, framework survey, report outline, etc.
assets/               # tiny sample assets (see assets/README.md)
```

## 🧪 What runs now (examples & scripts)

All demos are intentionally tiny; they run in seconds on CPU and either have tests under `tests/` or are wired through small scripts in `scripts/`. Full step‑by‑step commands and expected artifacts live in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

### 1. Environment + sanity checks

- **Environment probe** — `scripts/check_env.py`  
  Prints a one‑page summary of Python, NumPy, optional stacks (Neuromancer / PhysicsNeMo / TorchPhysics, RTAMT / MoonLight / SpaTiaL), and basic hardware. Can emit Markdown or JSON for attaching to a report.

  ```bash
  python scripts/check_env.py --md     # pretty, human‑readable
  python scripts/check_env.py --json > results/env_probe.json
  ```

- **Fast unit tests** — `pytest -q`  
  - Core logic: STL soft semantics, PDE helpers, synthetic STLnet dataset, etc.  
  - Tooling: RTAMT, MoonLight, SpaTiaL “hello” monitors (auto‑skip if deps missing).  
  - Frameworks: Neuromancer / PhysicsNeMo / TorchPhysics import‑level demos.

### 2. STL / STREL monitors (“hello” examples)

- **Differentiable STL (“soft” semantics)** — `monitoring/stl_soft.py`  
  Smooth min/max and temporal operators, with tests in `tests/test_stl_soft.py`. Used by the 1‑D diffusion / Burgers / Neuromancer training scripts.

- **RTAMT STL monitor** — `monitors/rtamt_hello.py` + `tests/test_rtamt_hello.py`  
  Evaluates specs such as `G_[a,b](u ≤ u_max)` on short traces; exercises offline robustness via RTAMT.

- **MoonLight STREL “hello”** — `monitors/moonlight_hello.py`, `monitors/moonlight_strel_hello.py`, tests `test_moonlight_hello.py`, `test_moonlight_strel.py`  
  Requires Java 21+; demonstrates STREL monitoring on small grid graphs.

- **SpaTiaL demo** — `monitors/spatial_demo.py`, tests `test_spatial_spec_hello.py`, `test_spatial_spec_demo.py`  
  Object‑centric spatio‑temporal specs and simple planning; MONA/`ltlf2dfa` recommended (Linux/macOS).

### 3. Physics‑based ML experiments

These are the examples requested for the course project: concrete systems with STL / spatial STL monitoring and plots once you run them.

**1‑D diffusion PINN + STL bound (RTAMT)**

- Code: `experiments/diffusion1d.py`  
- Configs: `configs/diffusion1d_baseline.yaml`, `configs/diffusion1d_stl.yaml`  
- Launcher: `scripts/run_experiment.py`, STL‑focused trainer: `scripts/train_diffusion_stl.py`  
- STL integration:
  - Differentiable “always‑below” safety penalty via `stl_soft` during training.
  - Post‑hoc auditing with RTAMT via `scripts/eval_diffusion_rtamt.py`.

Typical workflow (see also `docs/REPRODUCIBILITY.md §3.1–3.4`):

```bash
# Baseline PINN (no STL penalty)
python scripts/run_experiment.py --config configs/diffusion1d_baseline.yaml

# With soft STL penalty
python scripts/run_experiment.py --config configs/diffusion1d_stl.yaml

# RTAMT audit on the saved field
python scripts/eval_diffusion_rtamt.py   --field results/diffusion1d/..._field.pt   --u-max 1.0 --dt 0.01   --out-json results/diffusion1d_rtamt.json
```

Artifacts (per run) go under `results/` (logs, checkpoints, grid tensor) and can be turned into figures via `scripts/plot_ablations.py`.

**2‑D heat equation PINN + STREL containment (MoonLight)**

- Code: `experiments/heat2d.py`, helper `physics/heat2d.py`  
- Dataset generator: `scripts/gen_heat2d_frames.py`  
- Trainer / auditor: `scripts/train_heat2d_strel.py`, `scripts/eval_heat2d_moonlight.py`  
- Spec: STREL script `scripts/specs/contain_hotspot.mls` (e.g., “hotspot dissipates / stays localized”).

These scripts simulate 2‑D temperature fields on a small grid, train a PINN surrogate, and then audit spatio‑temporal properties with MoonLight. When run, they populate:

```text
results/heat2d/
  frames/            # small .npy field snapshots over time
  audit.json         # STREL robustness summary from MoonLight
```

**1‑D viscous Burgers’ PINN + STL safety constraint (TorchPhysics)**

- Script: `scripts/train_burgers_torchphysics.py`  
- Framework: TorchPhysics PINN with STL‑style `|u| ≤ u_max` constraint.  
- Uses a differentiable STL penalty plus post‑hoc robust satisfaction check; prints ρ (robustness) and writes a packed field tensor for later monitoring / plotting.

This provides a second PDE example beyond diffusion/heat, now in TorchPhysics.

**Neuromancer sine / ODE demo + STL bound**

- Framework demos: `frameworks/neuromancer_hello.py`, `frameworks/neuromancer_stl_demo.py`  
- Training script: `scripts/train_neuromancer_stl.py` with config `configs/neuromancer_sine_bound.yaml`.

This is a small non‑PDE example that shows how to attach STL penalties to Neuromancer objectives and audit them with RTAMT.

**Synthetic STLnet‑style time‑series**

- Dataset: `datasets/stlnet_synthetic.py`  
- Tests: `tests/test_stlnet_dataset.py`

Generates small, STL‑labeled traces to sanity‑check robustness semantics and to serve as a simple 1‑D example (in the spirit of STLnet) alongside the PDE fields.

### 4. Framework “hello‑worlds”

Small import‑level or toy demos that ensure optional frameworks are wired correctly:

- `frameworks/neuromancer_hello.py` — basic constrained optimization demo; tested in `test_neuromancer_hello.py`.  
- `frameworks/physicsnemo_hello.py` — import‑only PhysicsNeMo smoke test; `test_physicsnemo_hello.py`.  
- `frameworks/torchphysics_hello.py` — TorchPhysics PINN toy problem; `test_torchphysics_hello.py`.  
- `frameworks/spatial_spec_hello.py` — quick SpaTiaL usage example; covered by SpaTiaL tests.

> CI runs `pytest -q` on Ubuntu and only installs the *lean* deps. Optional stacks (Neuromancer, PhysicsNeMo, TorchPhysics, RTAMT, MoonLight, SpaTiaL) are intended for local / Docker runs and are guarded so tests skip cleanly if they are not present.

---

## 🧩 Integration patterns (how STL/STREL ties in)

**A. Post‑hoc monitoring** (zero‑risk):  
Run RTAMT (STL) and MoonLight (STREL) on model outputs and log *robustness* values. Use for dashboards, ablations, and safety thresholding.

**B. Soft constraints in training** (lightweight, differentiable, *no guarantees*):  
Use `stl_soft` to approximate STL robustness (smooth min/max via log‑sum‑exp / softplus), then add to loss:
- **Neuromancer**: plug robustness penalties into `nm.loss.PenaltyLoss` (naturally supports constrained training).  
- **TorchPhysics**: add STL penalty into the Lightning loss alongside PDE residuals (see the Burgers’ script).  
- **PhysicsNeMo**: add a callback/head computing soft robustness; aggregate into Hydra‑configured loss.

**C. Spatial logic** (STREL/SpaTiaL):  
For PDE fields: encode properties like **“hotspot dissipates within T across the domain”** (STREL reach/escape).  
For object/sensor graphs: specify relations like **“no more than K sensors within radius R below speed threshold for >Δt”** (SpaTiaL).

---

## 📚 Frameworks & tooling at a glance

| Component | What it is (one‑liner) | Install notes |
|---|---|---|
| **Neuromancer** | PyTorch SciML library for constrained optimization, physics‑informed models, and differentiable predictive control | `pip install neuromancer` or clone; see docs and examples. |
| **PhysicsNeMo** | NVIDIA’s (ex‑Modulus) physics‑AI stack; neural operators/PINNs with Hydra tooling | `pip install nvidia-physicsnemo` (Sym add‑on: `nvidia-physicsnemo-sym`); or use the NGC container. |
| **TorchPhysics** | Mesh‑free PDE library with PINNs/DeepRitz/DeepONets/FNO | `pip install torchphysics` (needs PyTorch ≥ 2.0). |
| **RTAMT** | STL monitoring (offline & bounded‑future online) with robustness semantics | `pip install rtamt` (optional C++ backend). |
| **MoonLight (STREL)** | Java engine + Python package for spatio‑temporal monitoring (STREL) | `pip install moonlight` + **Java 21+** on `PATH`. |
| **SpaTiaL** | Spatio‑temporal specifications over objects/relations; small planner | `pip install spatial-spec` (+ MONA/`ltlf2dfa`; Linux/macOS recommended). |

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
   - Use FNO tutorial datasets (per PhysicsNeMo docs) and monitor pressure/flux bounds or boundary conditions.  
   - Example specs: `G_[0,T] (p_max − p(x,t) ≥ 0 ∧ p(x,t) − p_min ≥ 0)`; `G_[0,T] (‖∇p‖ ≤ c)`.

3) **Urban sensor networks (traffic speed)** — *graph sensors, natural STREL*  
   - **METR‑LA** (207 sensors, 5‑minute speeds, **Mar 1–Jun 30 2012**).  
   - **PEMS‑BAY** (325 sensors, 5‑minute speeds, **Jan 1–May 31 2017**).  
   - Example specs: *No cascade:* **not**(≥ 10 sensors within 2‑hop radius below 20 mph for ≥ 15 min);  
     *Bounded propagation:* congestion waves propagate slower than `v_max` across adjacency.

> Alternative: **Air‑quality sensors** (e.g., PM2.5 city networks) with STREL “surround/reach” constraints (see STLnet and related smart‑city work).  
> *Scaling option later:* **LargeST** (NeurIPS 2023) provides statewide, long‑horizon traffic with metadata—use only after the core pipeline is proven on small sets.

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
- SpaTiaL relies on **MONA/ltlf2dfa** and is best on Linux/macOS (its automaton planner is skipped on Windows in tests).  
- Keep datasets **small** (e.g., 16–64² grids, 100–500 samples) for rapid iteration; scale later.  
- Use `pytest -q -k ...` to run focused subsets; CI mirrors this minimal footprint.  
- Store artifacts under `results/` (logs, checkpoints, fields) and `figs/` (plots), as described in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

---

## 🧾 References (primary docs & datasets)

- **Neuromancer** — Repo & docs (features, install, examples).  
  - GitHub: https://github.com/pnnl/neuromancer  
  - Docs: https://pnnl.github.io/neuromancer/

- **NVIDIA PhysicsNeMo** — Repo & install guide (ex‑Modulus; CPU/GPU; tutorials e.g., **Darcy FNO**).  
  - GitHub: https://github.com/NVIDIA/physicsnemo  
  - Install: https://docs.nvidia.com/physicsnemo/latest/getting-started/installation.html  
  - Darcy FNO tutorial: https://docs.nvidia.com/physicsnemo/latest/physicsnemo-sym/user_guide/neural_operators/darcy_fno.html

- **TorchPhysics** — Repo & docs (PINNs/DeepRitz/DeepONets/FNO).  
  - GitHub: https://github.com/boschresearch/torchphysics  
  - Docs: https://boschresearch.github.io/torchphysics/

- **RTAMT** — STL monitoring (robustness; bounded‑future online).  
  - GitHub: https://github.com/nickovic/rtamt  
  - PyPI: https://pypi.org/project/rtamt/

- **MoonLight (STREL)** — Java engine + Python package; **requires Java 21+**.  
  - GitHub: https://github.com/MoonLightSuite/moonlight

- **SpaTiaL** — Spatio‑temporal specifications; Linux/macOS recommended (MONA/`ltlf2dfa`).  
  - GitHub: https://github.com/KTH-RPL-Planiacs/SpaTiaL  
  - PyPI: https://pypi.org/project/spatial-spec/

- **STLnet** — NeurIPS 2020 paper enforcing STL in sequence models; relevant for STL‑guided training ideas.  
  - Abstract/PDF: https://proceedings.neurips.cc/paper/2020/file/a7da6ba0505a41b98bd85907244c4c30-Paper.pdf  
  - Code: https://github.com/meiyima/STLnet

- **Physics datasets** (starter options)  
  - **PDEBench** (NeurIPS 2022 benchmark + data): https://github.com/pdebench/PDEBench  
  - **FNO datasets** (Burgers/Darcy/NS): https://github.com/li-Pingan/fourier-neural-operator

- **Traffic datasets** (sensor networks)  
  - **METR‑LA**: description & files (207 sensors, Mar–Jun 2012): https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset  
  - **PEMS‑BAY**: summary (325 sensors, Jan–May 2017): https://torch-spatiotemporal.readthedocs.io/en/latest/modules/datasets_in_tsl.html  
  - **LargeST (NeurIPS 2023)**: paper & code:  
    - Paper: https://proceedings.neurips.cc/paper_files/paper/2023/file/ee57cd73a76bd927ffca3dda1dc3b9d4-Paper-Datasets_and_Benchmarks.pdf  
    - Repo: https://github.com/liuxu77/LargeST

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
