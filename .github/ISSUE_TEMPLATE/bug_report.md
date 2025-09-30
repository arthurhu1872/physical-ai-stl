---
name: Bug report
about: Report a reproducible problem in the Physical‑AI‑STL repo (frameworks, integrations, datasets, or specifications)
title: "[BUG] <concise summary>"
labels: ["bug"]
assignees: []
---

<!--
Thanks for filing a high‑quality bug! Please keep this report concise but complete.
Fields marked with **Required** must be filled for us to reproduce quickly.
-->

## 1) Summary **Required**
A clear, concise description of the problem.

## 2) Impact / Severity **Required**
- [ ] **Blocks progress** (cannot run or evaluate)
- [ ] **Incorrect results** (e.g., wrong STL robustness, false satisfaction/violation)
- [ ] **Crash/exception**
- [ ] **Performance regression** (slower or higher memory)
- [ ] **Documentation/typo**
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

**Model / problem class (check all that apply):**
- [ ] ODE
- [ ] PDE
- [ ] Hybrid / CPS
- [ ] PINN
- [ ] Controller / closed loop
- [ ] Other: _<describe>_

**Task phase:**
- [ ] Training
- [ ] Inference/simulation
- [ ] Monitoring / specification checking
- [ ] Post‑processing / reporting
- [ ] CI / Docker / environment

## 4) Expected vs. Actual **Required**
**Expected:** _What should happen? Include the intended STL/STREL property in plain English._
**Actual:** _What happens instead? If incorrect results, include observed robustness values, traces, or counterexamples._

## 5) Minimal Reproducible Example (MRE) **Required**
Provide the **smallest** code/config that reproduces the issue. Prefer a self‑contained script or a single config + command.

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
# e.g., phi := always_[0,5](temp(x) < 70)  or  spatial until, STREL, etc.
```

**Exact command(s) to run:**
```bash
# from repo root
python -m <module or script> --config <path>  # and flags
```

## 6) Environment & Versions **Required**
> Exact versions are critical for reproducibility.

- Repo commit: `git rev-parse HEAD` = `<hash>`
- Install method(s): _conda/pip/source_ (exact commands)
- OS: _<e.g., Ubuntu 22.04 / macOS 14 / Windows 11>_
- Python: _<e.g., 3.11.6>_
- PyTorch: _<e.g., 2.4.0>_  | CUDA/CuDNN: _<e.g., 12.1 / 9.x>_
- GPUs: _<model & count>_  | CPU: _<model>_  | RAM: _<GB>_
- Key libraries (exact versions):
  - Neuromancer: _<ver / commit>_
  - PhysicsNeMo: _<ver / commit>_
  - TorchPhysics: _<ver / commit>_
  - RTAMT: _<ver>_
  - MoonLight: _<ver>_
  - SpaTiaL: _<ver>_
  - Others: _<names & versions>_
- Reproducibility: seed(s) used = _<int>_; determinism flags set? _<yes/no>_

## 7) Logs / Stack Traces / Artifacts
Paste the most relevant lines. Attach files if needed (traces, CSVs, screenshots).
For performance regressions, include timings/memory before vs after.

<details><summary>Logs (collapse/expand)</summary>

```
# paste here
```
</details>

## 8) Regression?
- First known good commit / version: _<hash/version>_
- First known bad commit / version: _<hash/version>_
- Changes between them (if known): _<links or summary>_

## 9) Additional Context
Links to related issues/PRs/papers/datasets; workarounds tried; hypotheses.

---

### Reporter Checklist (please confirm)
- [ ] I searched existing issues and discussions.
- [ ] I can reproduce this on the latest `main` (or I included commit info).
- [ ] I provided a minimal reproducible example and exact commands.
- [ ] I included precise version info (OS, Python, PyTorch, CUDA, library versions).
- [ ] I attached logs or artifacts that show the problem.
