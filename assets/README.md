# `assets/` — tiny, reproducible demo data

This directory holds **small, deterministic sample assets** used in quick examples and tests (e.g., MoonLight/STREL monitoring of 2‑D heat diffusion). It is **git‑tracked only for tiny files**; anything large should be regenerated locally using the provided scripts.


## What belongs here (and what doesn’t)

- ✅ **Tiny samples** (a few hundred KB total) that make tutorials and unit tests runnable out‑of‑the‑box.
- ✅ Provenance files (e.g., `meta.json`) that document how to regenerate each sample exactly.
- ❌ **Do not** commit large or externally sourced datasets here. Regenerate from scripts or download at use time.
- ❌ Do not place model checkpoints or long videos in this repo.


## Standard layout

We keep a predictable structure so scripts and notebooks can find assets without extra configuration.

```
assets/
  README.md              # this file
  heat2d_scalar/         # small demo for 2‑D heat equation (scalar field)
    frame_0000.npy       # individual frames: shape (nx, ny), dtype float32 (by default)
    frame_0001.npy
    ...
    meta.json            # full provenance for reproducibility
    field_xy_t.npy       # optional packed tensor, shape (nx, ny, nt) if requested
```

> **Why `.npy` instead of images?** Monitors like STL/STREL work over *numeric signals*; saving full‑precision arrays avoids quantization and color‑map artifacts that can flip robustness at thresholds.


---

## Quickstart: generate a minimal 2‑D heat sequence

This uses the self‑contained script in `scripts/` (no external deps beyond NumPy).

```bash
# 50 frames on a 32×32 grid (≈0.2 MB total)
python scripts/gen_heat2d_frames.py   --nx 32 --ny 32 --nt 50   --outdir assets/heat2d_scalar
```

You can customize physics and numerics:

- **Boundary conditions:** `--bc {periodic,neumann,dirichlet}` (Dirichlet value via `--dirichlet-value`).
- **Integrator:** `--method {ftcs,fft}`. `fft` is unconditionally stable but requires periodic BC.
- **Time step:** set `--dt` manually, or let the script pick a stable step with `--auto-dt`.
- **Initial condition:** `--init {gaussian,two_gaussians,ring,checker}`, width via `--sigma`, amplitude via `--amplitude`, noise via `--noise`.
- **Packing & dtype:** add `--also-pack` to save a single tensor (`--layout {xy_t,t_xy}`); choose storage `--dtype` (e.g., `float16` to halve disk use).
- **Reproducibility:** `--seed` controls the random initial condition.

Examples:

```bash
# Periodic BC + FFT solver (unconditionally stable), also pack a single 3‑D tensor
python scripts/gen_heat2d_frames.py   --nx 64 --ny 64 --nt 80 --bc periodic --method fft --also-pack   --dtype float32 --outdir assets/heat2d_scalar

# No‑flux (Neumann) BC with a double‑hotspot initial condition, auto‑chosen stable dt
python scripts/gen_heat2d_frames.py   --nx 48 --ny 48 --nt 60 --bc neumann --init two_gaussians --auto-dt   --outdir assets/heat2d_scalar
```

The script writes a **`meta.json`** that captures all knobs (grid, dt, BC, solver, seed, dtype, tool versions) so anyone can reproduce the exact sequence.


---

## One‑minute STL/STREL check (optional but recommended)

Once you have a tiny sequence, you can sanity‑check it against a spatio‑temporal property with **MoonLight** (STREL) or a temporal property with **RTAMT** (STL).

### A. Spatio‑temporal monitoring with MoonLight (STREL)

MoonLight exposes STREL in Python (requires a recent Java runtime) and supports spatial operators (“surround”, “reach”, etc.). Install the Python wrapper:

```bash
pip install moonlight
# MoonLight requires Java 21+ on your system PATH.
# On macOS with Homebrew: brew install openjdk@21 && sudo ln -sfn $(/usr/libexec/java_home -v 21)/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-21.jdk
```

Evaluate the provided STREL spec (`scripts/specs/contain_hotspot.mls`) on the packed field:

```bash
# First generate a packed tensor (see --also-pack above), then:
python scripts/eval_heat2d_moonlight.py   --field assets/heat2d_scalar/field_xy_t.npy   --mls   scripts/specs/contain_hotspot.mls   --formula contain_hotspot
```

This prints the robustness over time and a pass/fail summary for the named formula.  *(MoonLight project & STREL details:)* 


### B. Temporal monitoring with RTAMT (STL)

For 1‑D diffusion demos (e.g., bounding a scalar state over a horizon), use the RTAMT‑based script:

```bash
pip install rtamt
python scripts/eval_diffusion_rtamt.py   --ckpt results/diffusion1d_week2_field.pt   --spec upper --u-max 0.8   --agg mean --temp 0.1
```

RTAMT provides offline and online monitors for STL with quantitative robustness semantics and an optimized C++ backend for discrete‑time online monitoring.


---

## Naming & size guidance (keep the repo light)

- Prefer **tiny grids** (`32×32`) and short **clips** (`nt≲50`). A packed `float32` tensor with `(32,32,50)` is ~0.2 MB; `float16` halves that.
- Save fewer frames with `--save-every K` to thin time sampling when you only need a coarse trajectory.
- If you need image previews for slides, generate them **locally** from `.npy` (don’t commit), e.g., with `scripts/utils_plot.py`.
- Large assets (movies, long runs, external datasets) should live outside the repo or be re‑generated via scripts.


---

## Reproducibility checklist

- Commit the **`meta.json`** alongside any sample you check in.
- Record the **command line** you used (or add it as a comment in `meta.json`).
- Keep **random seeds** fixed for demos to ensure stable robustness values.
- If you change grid resolution or layout, **update downstream script flags** (`--layout {xy_t,t_xy}`) accordingly.


---

## References & context (for this project’s scope)

- **Neuromancer (SciML / physics‑based ML)** — PyTorch library for modeling, optimization, and control with physics‑informed components. Useful for neural ODE/PDE baselines that we may pair with monitoring. 
- **MoonLight (STREL)** — spatio‑temporal monitoring with spatial operators; Python package `moonlight` and Java 21+ runtime required. 
- **RTAMT (STL)** — real‑time monitoring library with offline/online robustness semantics; Python package `rtamt`. 
- **PhysicsNeMo** (NVIDIA) and **TorchPhysics** (Bosch) — complementary physics‑ML frameworks we are surveying for PDE models/datasets. 
- **STLnet (NeurIPS’20)** — example of enforcing temporal‑logic properties in learning; inspiration for logic‑aware training objectives. 
- **NNV 2.0 (CAV’23)** — broader verification context (neural ODEs, NN controllers). 

If you need a public PDE dataset to try larger‑scale experiments locally (not committed here), consider **PDEBench** or framework examples in PhysicsNeMo/TorchPhysics docs. 


---

## Troubleshooting

- **MoonLight not found / Java errors.** Ensure `pip install moonlight` succeeded and Java 21+ is on your PATH. 
- **Unstable time stepping with `ftcs`.** Use `--auto-dt` or switch to `--method fft` with `--bc periodic` (unconditionally stable).
- **Robustness flips at thresholds.** Avoid converting `.npy` to images before monitoring; stay in floating‑point space.
- **Repo getting large.** Prefer packed tensors with `float16` and sub‑sampled time; never commit long runs.

---

## Attributions & licenses

- MoonLight is Apache‑licensed; see its repo for details. 
- RTAMT is BSD‑3‑Clause; see its repo. 
- PhysicsNeMo and TorchPhysics are Apache‑licensed; see their repos/docs. 
