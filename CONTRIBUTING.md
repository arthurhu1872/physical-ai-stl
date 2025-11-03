# Contributing

> **Scope.** This repository powers the **Physical‑AI × (Spatial) Signal Temporal Logic (STL)** effort for **Vanderbilt CS‑3860‑01 Undergraduate Research (3 credits)**. The goal is to integrate STL/STREL monitoring with physics‑ML frameworks (Neuromancer, TorchPhysics, PhysicsNeMo), compare them on small, **CPU‑friendly** demos (e.g., diffusion/heat, Burgers), and produce an **end‑of‑semester report**. See **[`docs/roadmap.md`](docs/roadmap.md)** and **[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)**.

---

## TL;DR (fastest path)

```bash
# 0) Local venv (recommended)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 1) Minimal install (tiny; CI-parity)
python -m pip install -r requirements.txt -r requirements-dev.txt

# 2) Optional stacks (STL/STREL + physics‑ML; may pull PyTorch/Java)
python -m pip install -r requirements-extra.txt

# 3) Sanity check (env + optional deps visibility)
python scripts/check_env.py

# 4) Smoke tests first (fast, CPU‑only)
make test-fast

# 5) Full tests (still CPU‑only; optional deps skip gracefully)
make test
```

> **CI reality.** CI runs on Linux, Python **3.10–3.13** (matrix), CPU‑only. Optional deps (PyTorch, MoonLight/Java, PhysicsNeMo) **must skip gracefully** if missing. Keep base installs lean.

---

## Course logistics (for this repo)

- **Standing meeting:** **Fridays 11:00** (Zoom during construction; may adjust).
- **Effort target:** **6–9 hrs/week** (rule of thumb for 3 credits).
- **Deliverable:** concise **report by semester end**; see `docs/report/outline.md`.
- **What to work on:** evaluate frameworks, integrate STL/STREL monitors, and propose STL‑friendly **problem spaces/datasets** (start with **diffusion/heat** + RTAMT/MoonLight).

---

## How to contribute (lanes & expectations)

Contributions that help the course goals are welcome. Pick one of these lanes and follow the checklist.

### 1) Specifications (STL/STREL)

- **STL (RTAMT):** add Python monitors in `src/physical_ai_stl/monitoring/` (e.g., `rtamt_hello.py`).
- **STREL (MoonLight):** place `.mls` spec files in `scripts/specs/` and wire them via helpers (e.g., `moonlight_helper.py`).
- **What to include for each spec:**
  - *Plain‑English paraphrase* of the property and intended safety/robustness intuition.
  - Variable names, **units**, and sampling period `Δt`; state any down‑sampling.
  - A **unit test** that asserts expected **truth/robustness** on toy signals.
- **Practical notes:**
  - Use **robust semantics** for numeric margins where possible.
  - Document **time bounds** precisely (e.g., `eventually_[0, 2.0]`) and align with the discrete sampling grid used in data.

### 2) Framework integrations (Neuromancer, TorchPhysics, PhysicsNeMo)

- Put “hello‑level” demos in `src/physical_ai_stl/frameworks/`.
- Gate heavy deps behind **optional installs** (`requirements-extra.txt`) and **lazy imports**; tests must **skip** if the package is absent.
- Aim for **CPU‑first** examples; if GPU adds value, document a clear fallback.
- Pointers:
  - **Neuromancer:** PyTorch‑based, supports PINNs, system ID, and differentiable predictive control. Good for diffusion/Burgers prototypes.
  - **TorchPhysics:** PDE‑focused library built on PyTorch, with PINNs/DeepONet/FNO examples.
  - **PhysicsNeMo:** NVIDIA’s Physics‑ML framework (PyTorch‑centric) with model zoo and optimized operators.

### 3) Experiments & configs

- Put small, STL‑ready **configs** in `configs/` (e.g., `diffusion1d_*.yaml`, `heat2d_*.yaml`).
- Use `scripts/run_experiment.py` or a focused script in `scripts/`.
- Save artifacts to **ignored** paths (`results/`, `runs/`, `figs/`); don’t commit large files.

### 4) Datasets / problem spaces

- Only use **public, license‑compatible** datasets; record source + license in the PR.
- Prefer tiny or synthetic sets (diffusion/heat toy data, STLnet‑style generators).
- Provide a tiny **download/prepare** helper or explicit steps, and add a **unit test** that uses a **10–100 sample** slice.

---

## Environment & installation

- **Python:** **3.10–3.13** (CI tests all; local ≥3.10).
- **OS:** Linux/macOS/WSL. (SpaTiaL & PhysicsNeMo are Linux/macOS‑friendliest; Windows users should prefer WSL for these.)
- **CPU‑only PyTorch wheels:** to avoid large CUDA downloads:
  ```bash
  python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
  ```
- **MoonLight (STREL) needs Java** at runtime; see `docs/REPRODUCIBILITY.md` for options.

Sanity check anytime:
```bash
python scripts/check_env.py
```

> See also **`docs/INSTALL_EXTRAS.md`** for pragmatic, stack‑by‑stack install notes.

---

## Style, testing, and reproducibility

- **Lint/format:** `ruff` + `codespell` via `pre-commit`. Run `make lint` / `make format`.
- **Types:** `mypy` is welcome (non‑blocking in CI).
- **Unit tests:** keep **fast** (seconds). **Always** skip tests that depend on absent optional deps with `pytest.importorskip(...)`.
- **Determinism:** set seeds; rely on `tests/conftest.py` defaults (single‑thread BLAS/OpenMP, deterministic RNG). Prefer **auditable logs** (seed, package versions).
- **Pre‑commit hooks:** enable locally:
  ```bash
  python -m pip install pre-commit ruff codespell mypy
  pre-commit install
  pre-commit run --all-files
  ```

---

## Dependency policy (keep it lean)

- **Do not** add heavy packages to `requirements.txt`.
- Put optional stacks in **`requirements-extra.txt`** with clear comments and platform markers.
- Always **lazy‑import** optional deps and guard code paths so CI remains **CPU‑only** and fast.
- If you truly need a new heavy dep, justify it in the PR and ensure **hello‑level tests** exercise it in **seconds**.

---

## Branches, commits, and PRs

- **Branches:** `feature/<topic>` or `fix/<topic>`.
- **Commits:** imperative, concise (e.g., `add rtamt monitor for diffusion1d`); link to the why when non‑obvious.
- **PR size:** small and reviewable; land iterative slices.
- **CI:** green is required (lint + tests).
- **Artifacts:** don’t commit large data or figures; small `.csv`/`.json` snippets only.

**PR checklist (copy into your description):**
- [ ] I ran `make test-fast` locally (and `make test` if my change touches broader areas).
- [ ] I ran `pre-commit run --all-files` (ruff/codespell clean).
- [ ] I added/updated **unit tests** (skip‑aware for optional deps).
- [ ] I updated **docs** (README or `docs/*`) where appropriate.
- [ ] I kept base installs lean (heavy deps only in `requirements-extra.txt`).
- [ ] I included dataset/source **links & licenses** (if applicable).
- [ ] I provided commands to **reproduce** results (when relevant).

---

## Adding a new experiment (definition of done)

1. **Config** in `configs/` (YAML) with clear, minimal defaults (CPU‑friendly).
2. **Spec(s)** in STL/STREL with a short paraphrase and a **unit test** validating truth/robustness on toy signals.
3. **Script** or `scripts/run_experiment.py`; print the **exact** command to reproduce.
4. **Results** into `results/` or `runs/` (ignored); include small summary tables/plots in the PR if helpful.
5. **Compute budget**: report rough runtime/hardware (e.g., “~30s on M2/CPU”).
6. **What we learned**: 3–5 bullets in the PR or `docs/sprint*_report.md`.

---

## Writing good specs (STL/STREL quick guidance)

- **Make the signal model explicit.** Name streams (`T(x,t)`), units, sampling `Δt`, and grid extents.
- **Prefer robust semantics.** Use robustness margins when available (supported by RTAMT and MoonLight).
- **Bound the future.** Prefer bounded operators (e.g., `eventually_[0, τ]`) to keep monitors causally implementable.
- **Spatial with PDEs.** Use STREL or spatial frameworks (MoonLight, SpaTiaL) when properties depend on **where** as well as **when**.
- **Document tolerances.** Note numerical tolerances (e.g., ±0.5 °C) to avoid brittle tests.
- **Cross‑check** with a tiny synthetic trace before wiring into a training loop.

---

## Tools & references (interoperate with)

- **Neuromancer (PNNL):** <https://github.com/pnnl/neuromancer>
- **RTAMT:** <https://github.com/nickovic/rtamt>
- **MoonLight:** <https://github.com/MoonLightSuite/moonlight>
- **SpaTiaL:** <https://github.com/KTH-RPL-Planiacs/SpaTiaL>
- **PhysicsNeMo (NVIDIA):** <https://github.com/NVIDIA/physicsnemo>
- **TorchPhysics (Bosch):** <https://github.com/boschresearch/torchphysics>
- **STLnet (paper & code):** <https://proceedings.neurips.cc/paper/2020/hash/a7da6ba0505a41b98bd85907244c4c30-Abstract.html> / <https://github.com/meiyima/STLnet>

---

## Licensing & attribution

- This project is **MIT‑licensed**. By contributing, you agree to license your changes under MIT.
- Cite external datasets/tools in code comments and docs. Respect third‑party licenses; avoid copying large code chunks.

---

## Code of conduct (short version)

Be respectful. No harassment or discrimination. Assume good intent, prefer small PRs, and write things others can reproduce on a laptop.

---

## Questions?

Open an issue or discussion. For course logistics (meetings, cadence), see **`docs/roadmap.md`**.

> Thanks for helping keep this project **reproducible**, **lean**, and focused on the **Physical‑AI × STL/STREL** goals!
