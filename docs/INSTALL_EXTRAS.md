# Installing optional stacks (STL/STREL + physics-ML)

This repo keeps the **base install lean**. Optional stacks live in `requirements-extra.txt`.

## 0) Create/activate a virtual environment
```bash
python -m venv .venv && source .venv/bin/activate   # macOS/Linux/WSL
# Windows Powershell:  py -m venv .venv; .venv\Scripts\Activate.ps1
```

## 1) Base + dev tools
```bash
pip install -r requirements.txt -r requirements-dev.txt
```

## 2) Optional stacks
```bash
pip install -r requirements-extra.txt
```

Notes:
- **MoonLight (STREL)** requires **Java 21+** at runtime. Install a JDK (e.g., Temurin 21)
  and make sure `java -version` prints 21.x before using MoonLight.
- **NVIDIA PhysicsNeMo** is installed from PyPI as `nvidia-physicsnemo` (and optionally
  `nvidia-physicsnemo.sym`). CPU wheels are available; CUDA wheels require NVIDIA drivers.
- **TorchPhysics** is on PyPI as `torchphysics`.
- **Neuromancer** is on PyPI as `neuromancer`.

## 3) Sanity check
```bash
python scripts/check_env.py --md
```

## 4) MoonLight quick smoke
Generate a tiny 2D heat rollout and evaluate a simple STREL formula:

```bash
# generate frames
python scripts/gen_heat2d_frames.py --nx 32 --ny 32 --nt 50 --dt 0.05 --alpha 0.5 --outdir assets/heat2d_scalar

# (after installing MoonLight + Java 21)
python scripts/eval_heat2d_moonlight.py --frames-dir assets/heat2d_scalar --formula "G[0,10](rho < 0.8)"
```

## 5) RTAMT quick smoke
```bash
python scripts/eval_diffusion_rtamt.py --ckpt results/diffusion1d_field.pt --stl "G[0,1](u <= 1.0)"
```
