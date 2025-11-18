# Reproducibility Playbook

This repository is built so that **anyone can reproduce the core experiments and figures on a clean machine** with minimal fuss (CPU‑only by default), while keeping heavier stacks (GPU, Java‑backed STREL, etc.) **optional**.

The steps below cover **environment setup**, **bit‑for‑bit runs**, **sanity checks**, and **troubleshooting** for Linux, macOS (Intel/Apple Silicon), and Windows (via WSL recommended).

> **Scope**  
> Targets the course deliverables for **Vanderbilt CS‑3860‑01 Undergraduate Research**: evaluating *Neuromancer*, *NVIDIA PhysicsNeMo*, and *TorchPhysics* on small PDE/ODE demos; wiring up STL/STREL monitoring (RTAMT, MoonLight, SpaTiaL); and generating ablations/figures and an end‑of‑semester report. Optional components **skip gracefully** when not installed.

---

## 0) One‑shot quickstart (CPU‑only)

If you just want a working environment plus smoke tests:

```bash
# From the repo root
make quickstart
# Runs: create venv → install minimal + dev → install optional extras (CPU) → quick tests
```

This uses the `Makefile` recipes and mirrors what CI does. It produces a local virtual environment at `.venv/` and runs fast tests (optional stacks are skipped if not present).

---

## 1) Exact environment

### 1.1 Supported platforms

- **Python:** 3.10–3.12 (3.11 recommended)
- **OS:** Linux (x86‑64/arm64), macOS (Intel & Apple Silicon), Windows via **WSL**  
  *Windows native is fine for the core CPU demos, but some optional tools (e.g., SpaTiaL) are Linux/macOS only.*

### 1.2 Create an isolated environment (recommended)

**Option A — Standard `venv` + `pip` (works everywhere):**

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip wheel
pip install -r requirements.txt -r requirements-dev.txt
# Optional heavy stacks used by experiments/monitoring:
pip install -r requirements-extra.txt
# If you installed extras, install a CPU PyTorch wheel explicitly:
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

**Option B — Docker (bit‑for‑bit, CPU‑only by default):**

```bash
# Build: minimal runtime + tests; add WITH_EXTRAS=1 and/or WITH_JAVA=1 as needed
docker build -t physical-ai-stl:cpu .
docker run --rm -it -v "$PWD":/workspace -w /workspace physical-ai-stl:cpu bash
# Inside the container, tests run by default; you can also run scripts manually.
```

**Option C — `uv` (fast installer, optional):**

```bash
uv pip install --python 3.11 -r requirements.txt -r requirements-dev.txt
# Optional stacks
uv pip install -r requirements-extra.txt
uv pip install --index-url https://download.pytorch.org/whl/cpu torch
```

> **Java for MoonLight (STREL)**  
> MoonLight’s Python interface requires a JDK at runtime. If you’re not using Docker’s `WITH_JAVA=1`, install a recent JDK (e.g., OpenJDK 21) and ensure `java -version` works on your PATH.  

---

## 2) Sanity checks (fast)

From the repo root with your environment active:

```bash
# Print a concise environment probe (Python, NumPy, optional stacks, GPU visibility)
python scripts/check_env.py --md

# Run unit tests (fast; optional stacks auto‑skip if missing)
pytest -q
```

You can also emit JSON for archival:

```bash
python scripts/check_env.py --json > results/env_probe.json
```

---

## 3) Reproduce core experiments

All experiments write artifacts under `results/` and are **seeded** by default for stability (you can change `seed` in configs/CLI). The repository keeps CPU‑friendly defaults so these runs complete on laptops.

### 3.1 Diffusion‑1D (PINN) — baseline

```bash
python scripts/run_experiment.py --config configs/diffusion1d_baseline.yaml
```

**Outputs** (under `results/`):

- Scalar logs (CSV) and checkpoints for the MLP field
- A saved field tensor (e.g., `*_field.pt`) on a regular grid for later auditing/plotting

### 3.2 Diffusion‑1D with soft‑STL penalty

```bash
python scripts/run_experiment.py --config configs/diffusion1d_stl.yaml
```

The config enables a differentiable **`always`** upper‑bound penalty on the temperature field. You can adjust weight/temperature in the YAML.

**Evaluate STL robustness with RTAMT** on the saved field:

```bash
# Point --ckpt to the *_field.pt produced above
python scripts/eval_diffusion_rtamt.py \
  --ckpt results/diffusion1d_*_field.pt \
  --spec upper --u-max 1.0 \
  --json-out results/rtamt_summary.json
```

The script prints a scalar robustness and whether the spec is satisfied. The JSON mirrors these values and records parameters used.

### 3.3 Heat‑2D with STREL (MoonLight)

Train a small heat‑equation model and **audit** a spatio‑temporal specification using the included MoonLight script:

```bash
# Trains a compact model; then runs STREL audit with the provided spec
python scripts/train_heat2d_strel.py --epochs 200 --audit \
  --mls scripts/specs/contain_hotspot.mls \
  --out results/heat2d
```

Alternatively, audit exported frames or a single field tensor explicitly:

```bash
# Example: audit frames dumped as .npy slices (one per time step)
python scripts/eval_heat2d_moonlight.py \
  --frames-dir results/heat2d/frames --glob "frame_*.npy" \
  --quantile 0.95 \
  --out-json results/heat2d_moonlight.json
```

> **MoonLight notes**  
> Requires a working Java runtime. If you see a Java‑related import error, install a JDK or rebuild the Docker image with `--build-arg WITH_JAVA=1`.

### 3.4 Ablation: STL weight sweep (figure)

Run a small sweep over the STL penalty weight λ and plot the robustness curve:

```bash
# Write λ vs. robustness to a CSV
python scripts/run_ablations_diffusion.py \
  --weights 0.0 0.1 1.0 3.0 10.0 \
  --epochs 100 \
  --repeats 3 \
  --out results/ablations_diffusion.csv

# Make a publication‑quality plot (PNG/PDF)
python scripts/plot_ablations.py \
  --csv results/ablations_diffusion.csv \
  --out figs/ablations_diffusion.png \
  --title "Diffusion‑1D: STL λ sweep"
```

### 3.5 Diffusion 1D experiment (PDE + STL)

To reproduce the diffusion‑1D results and figures end‑to‑end:

1. **Train baseline and STL models**

   ```bash
   python scripts/run_experiment.py --config configs/diffusion1d_baseline.yaml
   python scripts/run_experiment.py --config configs/diffusion1d_stl.yaml
   ```

2. **Run STL‑weight ablations**

   ```bash
   python scripts/run_ablations_diffusion.py --weights 0:10:6 --out results/diffusion1d_ablations.csv
   ```

3. **Evaluate robustness with RTAMT**

   ```bash
   # Replace the paths with the actual *_field.pt checkpoints produced above
   python scripts/eval_diffusion_rtamt.py --ckpt <baseline_field_ckpt> --spec upper --u-max 1.0 --json-out results/diffusion1d_baseline_rtamt.json
   python scripts/eval_diffusion_rtamt.py --ckpt <stl_field_ckpt>      --spec upper --u-max 1.0 --json-out results/diffusion1d_stl_rtamt.json
   ```

4. **Generate plots**

   ```bash
   python scripts/make_diffusion_plots.py
   ```

This creates:

- `assets/diffusion1d_baseline_field.png`
- `assets/diffusion1d_stl_field.png`
- `assets/diffusion1d_training_loss.png`
- `assets/diffusion1d_training_robustness.png`
- `assets/diffusion1d_training_loss_components_stl.png`
- `assets/diffusion1d_robust_vs_lambda.png`

### 3.6 Framework hello‑worlds (optional)

Confirm the physics‑ML stacks import and run simple demos:

```bash
# Neuromancer and TorchPhysics: tiny "hello" demos
pytest -q tests/test_neuromancer_hello.py -q
pytest -q tests/test_torchphysics_hello.py -q

# PhysicsNeMo: import‑only smoke test
pytest -q tests/test_physicsnemo_hello.py -q
```

These tests auto‑skip if the corresponding optional dependency is not installed.

---

## 4) Determinism & seeds

We take a **best‑effort determinism** approach suited for research prototypes:

- **Explicit seeds** are set in scripts and configs (`seed: 0` by default).
- **Threading** is constrained for stability (defaults set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` in `Makefile`).
- **PyTorch** is kept on CPU by default; GPU/MPS work but may introduce minor nondeterminism (e.g., CUDA atomics).  
  If you need stricter determinism on CUDA, consider setting:
  ```bash
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  ```
- **Soft semantics** (e.g., soft mins/maxes for STL) are numerically stable across runs given the same seed.

> **Tip**  
> For archival, capture `pip freeze > results/pip-lock.txt` and `python -V && pip -V` along with `results/env_probe.json`.

---

## 5) Results & artifact layout

By default, artifacts go to `results/`:

```
results/
├── diffusion1d_.../                # run directories (tagged)
│   ├── logs.csv
│   ├── model.ckpt
│   └── field.pt                    # grid tensor for auditing
├── heat2d/
│   ├── frames/                     # optional .npy frames for STREL
│   └── audit.json                  # MoonLight audit summary (if --audit)
├── ablations_diffusion.csv
├── env_probe.json
└── rtamt_summary.json
```

Figures are written under `figs/` and, for the core diffusion‑1D PDE + STL plots, under `assets/` by plot scripts.

---

## 6) Troubleshooting

- **`ModuleNotFoundError: torch` after installing extras**  
  Install an explicit CPU wheel:  
  `pip install --index-url https://download.pytorch.org/whl/cpu torch`

- **MoonLight import error / Java not found**  
  Install a JDK (e.g., OpenJDK 21), ensure `java -version` works, or use Docker with `--build-arg WITH_JAVA=1`.

- **SpaTiaL / MONA on Windows**  
  Not supported natively. Use Linux/macOS or WSL.

- **Slow plots or MKL warnings**  
  Set `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` (already the defaults in `Makefile`/Dockerfile).

- **`pytest` skips many tests**  
  That is expected when optional stacks are not installed. Install `-r requirements-extra.txt` to enable more tests.

---

## 7) Exact reproduction checklists

When you publish or hand off results, include:

- **Git**: repository URL + commit SHA (or a release tag/zip)
- **Config**: copy of YAML / CLI flags used
- **Environment**: `results/env_probe.json` + `pip freeze`
- **Hardware**: CPU model, RAM; note if GPU/MPS was used
- **Artifacts**: `results/` subfolder(s) and `figs/` outputs

> **CI parity:**  
> You can approximate CI locally with `make ci` (installs minimal deps via `uv` if available and runs `pytest -q` on Python 3.11).

---

## 8) Clean teardown

```bash
deactivate || true
rm -rf .venv results figs
# Optional: prune Docker image
docker rmi physical-ai-stl:cpu || true
```

---

### Appendix A — Minimal dependency matrix

- **Always:** `numpy`  
- **Dev/tests:** `pytest`, `pytest-cov`  
- **Optional experiments/monitoring:** `rtamt`, `moonlight` (needs Java), `spatial-spec`, `matplotlib`, `tqdm`, `pyyaml`, `scipy`, `torch` (installed explicitly)

> Exact versions are governed by `requirements*.txt`. Heavy stacks are purposefully **not** in the minimal runtime set.

---

### Appendix B — Reproducibility design choices

- **Skip‑graceful optional deps:** tests and scripts detect missing stacks and provide fallbacks where feasible.
- **Config‑first experiments:** everything critical is driven by small YAMLs under `configs/` or explicit CLI flags.
- **Small, self‑contained demos:** PINN demos use synthetic PDEs (diffusion/heat) to avoid fragile external datasets.
- **Container parity:** the Dockerfile mirrors CI and can include extras and Java via build args.

---

If anything here diverges from your platform, please open an issue with your OS/Python, the command you ran, and the stderr/stdout. Reproducibility bugs are treated as first‑class.
