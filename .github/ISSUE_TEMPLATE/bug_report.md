---
name: Bug report
about: Report a **reproducible** problem in the Physical‑AI‑STL repo (frameworks, STL/STREL monitoring, datasets, or specifications)
title: "[BUG] <concise summary>"
labels: ["bug"]
assignees: []
---

<!--
READ THIS FIRST — to help us triage quickly

• Keep it concise but complete; fill all **Required** fields.
• Provide a **CPU‑friendly** Minimal Reproducible Example (MRE) that runs from a clean env.
• Do **NOT** include secrets (API keys, tokens, private data).
• Prefer commands that mirror CI: `make quickstart` or steps from docs/REPRODUCIBILITY.md.
-->

## 1) Summary **Required**
One or two sentences that crisply describe the problem.

## 2) Impact / Severity **Required**
- [ ] **Blocks progress / hard failure** (cannot run or evaluate)
- [ ] **Incorrect results** (e.g., wrong **STL/STREL** robustness, false satisfaction/violation)
- [ ] **Crash / exception**
- [ ] **Nondeterministic / flaky** (seed‑dependent or CI‑only)
- [ ] **Numerical instability / divergence** (NaN/Inf/overflow/underflow)
- [ ] **Performance regression** (slower or higher memory)
- [ ] **Documentation / typo**
- [ ] **Security / privacy**
- [ ] Other: _<describe>_

## 3) Affected Area(s) **Required**

**Framework(s):**
- [ ] Neuromancer
- [ ] NVIDIA PhysicsNeMo
- [ ] Bosch TorchPhysics
- [ ] Internal glue / integration code
- [ ] Other: _<name & link>_

**STL tooling:**
- [ ] RTAMT
- [ ] MoonLight (STREL / spatial)
- [ ] SpaTiaL
- [ ] Other: _<name & link>_

**Dataset / problem space:**
- Name & version: _<e.g., Diffusion‑1D synthetic v0.2>_
- Source link: _<URL>_  ·  License: _<SPDX or name>_  ·  Size/shape: _<e.g., 10k samples, 64×64×T>_

**Model / problem class (check all that apply):**
- [ ] ODE
- [ ] PDE
- [ ] Hybrid / CPS
- [ ] PINN / physics‑ML
- [ ] Neural operator (e.g., FNO / DeepONet)
- [ ] Controller / closed loop
- [ ] Other: _<describe>_

**Task phase:**
- [ ] Training
- [ ] Inference / simulation
- [ ] Monitoring / specification checking
- [ ] Post‑processing / reporting
- [ ] CI / Docker / environment

**Precision / device:**
- [ ] CPU‑only
- [ ] CUDA GPU (arch & count): _<e.g., sm_89 · 1×RTX 4090>_
- [ ] Apple M‑series / MPS
- [ ] Other: _<describe>_

## 4) Expected vs. Actual **Required**

**Expected:** _What should happen?_ Include the intended STL/STREL property in plain English and, if helpful, its formula.  
**Actual:** _What happens instead?_ If results are incorrect, include observed robustness values, signal traces, or counterexamples.

**STL/STREL semantics used in this report (be precise):**
- **Time domain:** _<discrete vs dense>_; sample rate = _<Hz>_; interpolation = _<zero‑order / linear / other>_  
- **Time units:** _<e.g., seconds / steps>_  · **Horizon/window:** _<e.g., [0,5] s or 50 steps>_  
- **Spatial domain:** _<grid / graph / mesh>_; neighborhood metric & radius: _<e.g., Euclidean, r=1>_; boundary handling: _<fixed / wrap / ignore>_  
- **Robust semantics:** _<quantitative vs boolean>_; aggregation for **always/until**: _<min/max or soft‑min with temperature τ>_

**Numerics & solvers (if relevant):**
- `dtype` = _<fp32/fp64>_; integrator = _<Euler/RK4/RK45/etc>_ with **dt**/**tols** = _<values>_  
- Loss weights / penalties = _<values>_  · Gradient clipping = _<on/off & value>_  
- Any smoothing/softening of STL operators (e.g., log‑sum‑exp soft‑min with τ) = _<details>_

## 5) Minimal Reproducible Example (MRE) **Required**
Provide the **smallest** code/config that reproduces the issue. Prefer a **CPU‑only** repro under the **minimal** requirements (`requirements.txt`; optional stacks may be skipped).

**Code (trim to minimal):**
```python
# minimal.py
# (Keep only what’s necessary to reproduce)
```

**Config / spec (trim to minimal):**
```yaml
# e.g., configs/<name>.yaml
```

**STL / spatial spec (if applicable):**
```txt
# e.g., phi := always_[0,5](temp(x) < 70)
# or STREL example using spatial until / reach operators
```

**Exact command(s) to run from repo root:**
```bash
# Prefer a clean env. Examples:
make quickstart
# or, step‑by‑step:
python -m pip install -r requirements.txt -r requirements-dev.txt            # lean base
python -m pip install -r requirements-extra.txt                               # optional stacks (STL/STREL, frameworks)
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch  # CPU‑only PyTorch
pytest -q -k "<focused test>"                                                 # if relevant
python -m <module.or.script> --config <path>  # and flags used

# If Java‑backed tools are involved (MoonLight), include how you installed Java/JDK and version.
```

## 6) Environment & Versions **Required**
> Exact versions are critical for reproducibility (and CI parity).

- Repo commit: `git rev-parse HEAD` = `<hash>`
- Install method(s): _conda/pip/source_ (exact commands)
- OS: _<e.g., Ubuntu 22.04 / macOS 14 / Windows 11>_  · Kernel: _<optional>_
- Python: _<e.g., 3.11.7>_
- PyTorch: _<e.g., 2.4.x>_  | CUDA Toolkit / Driver: _<e.g., 12.4 / 555.xx>_  | cuDNN: _<e.g., 9.x>_
- GPUs: _<model & count>_  | CPU: _<model>_  | RAM: _<GB>_
- Java (if used): _<OpenJDK 21.x>_  | `JAVA_HOME` set? _<yes/no>_
- Key libraries (exact versions or commits):
  - Neuromancer: _<ver / commit>_
  - PhysicsNeMo: _<ver / commit>_
  - TorchPhysics: _<ver / commit>_
  - RTAMT: _<ver>_
  - MoonLight: _<ver>_
  - SpaTiaL: _<ver>_
  - Others: _<names & versions>_
- Reproducibility: seed(s) used = _<int>_; determinism flags set? _<yes/no>_  
  - If **yes**, list flags (e.g., `torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`).

## 7) Logs / Stack Traces / Artifacts
Paste the most relevant lines. Attach files if needed (plots, traces, CSVs, screenshots).
For performance regressions, include timings/memory **before vs after** under identical conditions.

<details><summary>Logs (collapse/expand)</summary>

```
# paste here
```
</details>

## 8) Regression?
- First known **good** commit / version: _<hash/version>_
- First known **bad** commit / version: _<hash/version>_
- Changes between them (if known): _<links or summary>_

## 9) Additional Context
Links to related issues/PRs/papers/datasets; workarounds tried; hypotheses.

---

### Reporter Checklist (please confirm)
- [ ] I searched existing issues and discussions.
- [ ] I can reproduce this on the latest `main` (or I included commit info).
- [ ] I provided a minimal **CPU‑friendly** reproducible example and exact commands.
- [ ] I included precise version info (OS, Python, PyTorch, CUDA, Java, library versions).
- [ ] I attached logs or artifacts that show the problem.
- [ ] (Optional) I tried `make quickstart` or followed `docs/REPRODUCIBILITY.md` for a clean env.
