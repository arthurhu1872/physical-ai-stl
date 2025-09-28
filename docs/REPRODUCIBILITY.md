# Reproducibility & Environment Notes

## Base
- Python ≥ 3.10
- `pip install -r requirements.txt`
- For tests: `pip install -r requirements-dev.txt` (pytest ≥ 8)

## Optional stacks (install only what you need)

### RTAMT (STL monitoring)
- `pip install rtamt`
- Offline/online monitors; discrete/dense time.
- API changed across versions; repo helpers handle common variants.

### MoonLight (STREL, spatio‑temporal)
- Requires **Java (JDK 17–21+)**.
- Python wrapper: `pip install moonlight`
- If Java is missing locally, install or use Docker with `WITH_JAVA=1` build arg.
- Spec files (`.mls`) live under `scripts/specs/`.

### SpaTiaL
- `pip install spatial-spec`
- Automaton-based planning requires **MONA** via `ltlf2dfa`.
- Windows is not supported for this part; Linux/macOS recommended.

### Neuromancer / TorchPhysics / PhysicsNeMo
- Neuromancer: `pip install neuromancer`
- TorchPhysics: `pip install torchphysics`
- PhysicsNeMo (Linux-first): `pip install nvidia-physicsnemo`
  - Some examples require CUDA or NVIDIA indices.
  - Prefer the repo’s CPU-only paths unless you have GPUs.

## Containers
- `Dockerfile` supports light CI and optional extras:
  - `--build-arg WITH_EXTRAS=1` to include heavy deps
  - `--build-arg WITH_JAVA=1` to install a JDK for MoonLight

## Pinning
- After picking versions, create a lock (e.g., `uv pip compile` or `conda-lock`)
  and commit it for longevity (CPU and GPU variants).
