# Contributing

> **Scope.** This repository powers the **Physical‑AI × (Spatial) Signal Temporal Logic (STL)** effort for **Vanderbilt CS‑3860‑01 Undergraduate Research (3 credits)**. The goal is to integrate STL/STREL monitoring with physics‑ML frameworks (Neuromancer, TorchPhysics, PhysicsNeMo), compare them on small, **CPU‑friendly** demos (e.g., diffusion/heat, Burgers), and produce an **end‑of‑semester report**. See **[`docs/roadmap.md`](docs/roadmap.md)** and **[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)**.

---

## TL;DR (fastest path)

```bash
# 0) Create a project‑local venv (recommended)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 1) Minimal install (tiny; CI‑parity)
python -m pip install -r requirements.txt -r requirements-dev.txt

# 2) Optional stacks (STL/STREL + physics‑ML; may pull PyTorch)
python -m pip install -r requirements-extra.txt

# 3) Smoke tests first (fast, CPU‑only)
make test-fast

# 4) Full tests (still CPU‑only; optional deps skip gracefully)
make test
```

> **Notes.** CI runs on **Python 3.11** (Ubuntu‑24.04) using a lean CPU install. Optional deps (PyTorch, MoonLight/Java, PhysicsNeMo) are **not required** for the core tests and must **skip gracefully** if missing.

---

## Project expectations (course‑specific)

- **Standing meeting:** **Fridays 11:00** (Zoom during construction; adjust as lab finalizes schedule).  
- **Effort target:** **6–9 hrs/week** (rule of thumb for 3 credits).  
- **Deliverable:** a concise **report by semester end** (see template in `docs/report/outline.md`).  
- **What to work on:** evaluate frameworks, integrate STL/STREL monitors, and propose STL‑friendly **problem spaces/datasets**.  
  – Start with **diffusion/heat** demos and RTAMT/MoonLight monitoring; add stretch tasks as needed.  
  – Keep everything **reproducible**, **small**, and **CPU‑first** by default.

---

## How to contribute (areas & expectations)

Contributions that help the course goals are welcome. Typical areas:

1) **Specifications (STL/STREL).**  
   - STL (RTAMT): add Python monitors in `src/physical_ai_stl/monitoring` or examples in `src/physical_ai_stl/monitors/rtamt_hello.py`.  
   - STREL (MoonLight): place `.mls` spec files in `scripts/specs/` and wire via helpers in `src/physical_ai_stl/monitoring/moonlight_helper.py`.  
   - Provide **plain‑English paraphrases**, variable/units, and **unit tests** that assert expected truth/robustness on toy signals.

2) **Framework integrations (Neuromancer, TorchPhysics, PhysicsNeMo).**  
   - New “hello” demos live in `src/physical_ai_stl/frameworks/`.  
   - Gate heavy deps behind **optional installs** (see `requirements-extra.txt`) and **lazy imports** (`pytest.importorskip`).  
   - Prefer **CPU‑first** examples; if GPU is optionally supported, document clearly and ensure CPU fallback works.

3) **Experiments & configs.**  
   - Put small, STL‑ready configs in `configs/` (see `diffusion1d_*.yaml`, `heat2d_*.yaml`).  
   - Use `scripts/run_experiment.py` or a focused training script in `scripts/`.  
   - Save artifacts to **ignored** paths (`results/`, `runs/`, `figs/`); don’t commit large files.

4) **Datasets/problem spaces.**  
   - Only use **public, license‑compatible** datasets; document source + license in the PR.  
   - Prefer tiny or synthetic sets (e.g., diffusion/heat, STLnet‑style toy generators).  
   - Provide a tiny **download/prepare** helper or explicit instructions, and add a **unit test** that uses a **10–100 sample** slice.

5) **Docs & tooling.**  
   - Update `docs/framework_survey.md`, `docs/dataset_recommendations.md`, and sprint notes as appropriate.  
   - Keep developer UX fast: `Makefile` targets (`quickstart`, `test-fast`, `lint`, `format`) should work on a fresh clone.

---

## Environment & installation

- **Python:** ≥ **3.10** (CI pinned to **3.11**).  
- **OS:** Linux/macOS/WSL. (SpaTiaL & PhysicsNeMo are Linux/macOS‑friendly; Windows users should prefer WSL for these.)  
- **CPU‑only PyTorch wheels:** if you install PyTorch manually, prefer CPU wheels to avoid large CUDA downloads:
  ```bash
  python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
  ```
- **MoonLight (STREL) requires Java**; see `docs/REPRODUCIBILITY.md` for options if you need Java locally or in CI.

Sanity check your environment any time:
```bash
python scripts/check_env.py
```

---

## Style, testing, and reproducibility

- **Lint/format:** [`ruff`](https://docs.astral.sh/ruff/) (via `pre-commit`); run `make lint` / `make format`.  
- **Types:** `mypy` is welcomed (non‑blocking in CI).  
- **Unit tests:** keep **fast** (seconds). Tests **must skip** when optional deps aren’t present (`pytest.importorskip`).  
- **Determinism:** set seeds and grids (see `src/physical_ai_stl/utils/seed.py`, `training/grids.py`).  
- **Pre‑commit hooks:** enable locally for quick feedback:
  ```bash
  python -m pip install pre-commit ruff codespell mypy
  pre-commit install
  pre-commit run --all-files
  ```

---

## Dependency policy (keep it lean)

- **Do not** add heavy packages to `requirements.txt`.  
- Put optional stacks in **`requirements-extra.txt`** with clear comments and platform markers.  
- Lazy‑import optional deps and guard code paths so CI remains **CPU‑only** and fast.  
- If you truly need a new heavy dep, justify it in the PR and ensure **hello‑level tests** exercise it in seconds.

---

## Branches, commits, and PRs

- **Branches:** `feature/<topic>` or `fix/<topic>`.  
- **Commits:** imperative, concise; include rationale when non‑obvious.  
- **PR size:** keep PRs **small and reviewable**; land iterative slices.  
- **CI:** green is required (lint + tests).  
- **Artifacts:** don’t commit large data or figures; attach small `.csv`/`.json` snippets if needed.

**PR checklist (copy into your description):**
- [ ] I ran `make test-fast` locally (and `make test` if my change affects broader areas).  
- [ ] I ran `pre-commit run --all-files` (ruff/codespell clean).  
- [ ] I added/updated **unit tests** (skip‑aware for optional deps).  
- [ ] I updated **docs** (README or `docs/*`) where appropriate.  
- [ ] I kept base installs lean (heavy deps only in `requirements-extra.txt`).  
- [ ] I included dataset/source **links & licenses** (if applicable).  
- [ ] I provided commands to reproduce results (when relevant).

---

## Adding a new experiment (definition of done)

When proposing a new experiment:

1. **Config** in `configs/` (YAML) with clear, minimal defaults (CPU‑friendly).  
2. **Spec(s)** in STL/STREL with a short paraphrase and a **unit test** validating truth/robustness on toy signals.  
3. **Script** or use `scripts/run_experiment.py`; print the **exact** command to reproduce.  
4. **Results** saved under `results/` or `runs/` (ignored); include small summary tables/plots in the PR if helpful.  
5. **Compute budget**: report rough runtime/hardware (e.g., “~30s on M2/CPU”).  
6. **What we learned**: 3–5 bullets in the PR description or `docs/sprint*_report.md`.

---

## Licensing & attribution

- This project is **MIT‑licensed**. By contributing, you agree your changes are licensed under MIT.  
- Cite external datasets/tools in code comments and docs. Respect third‑party licenses and avoid copying large chunks of external code.

---

## Questions?

Open an issue or discussion. For course logistics (meetings, cadence), see **`docs/roadmap.md`**.

> Thanks for helping keep this project **reproducible**, **lean**, and focused on the **Physical‑AI × STL/STREL** goals!
