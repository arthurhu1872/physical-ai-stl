---
name: "Feature / Experiment request (Physical‑AI × STL/STREL)"
about: "Propose a tightly‑scoped feature or experiment advancing Physical‑AI with STL/STREL monitoring for CS‑3860‑01."
title: "[feat] <short, imperative title>"
labels: ["enhancement","research"]
assignees: []
---

<!--
READ FIRST — to keep this repo lean, reproducible, and aligned with CS‑3860‑01:

1) Read README (Goals, Quickstart) and CONTRIBUTING.
2) Keep base installs lean; gate heavy stacks in requirements-extra.txt only.
3) Prefer fast CPU‑first demos; add tests that skip gracefully if optional deps are missing.
4) Ensure all datasets are public and license‑compatible; include citations/links.
5) Aim at the end‑of‑semester report; propose concrete milestones & Friday check‑ins.
6) Provide a CPU‑only path (or clear fallback) for everything you propose.
-->

## 1) Summary

**One‑liner:**  
<What is the feature/experiment?>

**Problem & motivation (why now):**  
<What research/engineering gap does this close? Why is STL/STREL the right tool?>

## 2) Research alignment

- **Framework(s):** ☐ Neuromancer ☐ PhysicsNeMo ☐ TorchPhysics ☐ Other: <name>  
- **STL tooling:** ☐ RTAMT (STL) ☐ MoonLight (STREL) ☐ SpaTiaL ☐ Other: <name>  
- **Category:** ☐ Monitor‑only ☐ Train‑time soft enforcement ☐ Post‑hoc evaluation ☐ Dataset/problem integration

<Briefly state how this advances the Physical‑AI + STL/STREL agenda for the course.>

## 3) Problem space / dataset

- **System / PDE/ODE/CPS:** <e.g., 1D diffusion, 2D heat, Burgers, cart‑pole, traffic sensors, etc.>  
- **Dataset/source & license:** <URL + license>; size & shape: <e.g., 10k samples, 64×64×T>  
- **Spatial layout / graph / mesh:** <grid, irregular, adjacency/neighbor definition>  
- **Relevance of STL/STREL:** <what properties matter? bounds, reach/escape, smoothness, propagation speed, etc.>

## 4) Specification(s)

Provide the temporal / spatio‑temporal properties you will monitor/enforce.

- **Informal:** <plain English description>  
- **Formal STL/STREL:**  
  ```txt
  <phi := G_[t1,t2](a → F_[τ1,τ2] b) ; use STREL reach/escape for spatial fields if needed>
  ```  
- **Robustness semantics & thresholds:** <how evaluated; normalization/margins; units>  
- **Monitor runtime mode:** ☐ offline (batch) ☐ online (bounded‑future) ☐ streaming

## 5) Proposed approach

- **Integration points:** <where monitors plug into training/inference/logging>  
- **Model/architecture:** <PINN/DeepONet/FNO/Neural ODE/PDE/etc.>  
- **Enforcement strategy (if any):** <loss shaping with soft robustness, penalties, shields, projection, etc.>  
- **Compute plan:** <CPU/GPU; est. memory/runtime on CPU; include CPU‑only fallback>  
- **Complexity budget:** <max wall‑clock per run, target ≤ minutes on CPU for smoke tests>

## 6) Acceptance criteria (Definition of Done)

- [ ] Reproducible script(s) under `scripts/` with argparse + README usage
- [ ] STL/STREL specs encoded and unit‑tested (robustness values sanity‑checked)
- [ ] At least **one baseline** (no STL) vs **one STL/STREL** variant, with metrics (task + robustness)
- [ ] Results table/figure with seeds noted (CSV or Markdown)
- [ ] Minimal deps in `requirements.txt`; heavy extras in `requirements-extra.txt`
- [ ] Tests (`pytest`) are fast and **skip** when optional stacks are absent
- [ ] Short docs: what it does, how to run, expected time/resources
- [ ] (Optional) Ablations: spec strength, monitor/enforcement variants
- [ ] (Optional) CI hook: add/extend a tiny skip‑aware test

## 7) Plan & milestones

- **Weekly cadence & load:** <hrs/week; guidance: 2–3 hrs per credit ⇒ ~6–9 hrs/wk for 3 credits>  
- **Milestones & dates (Fri check‑ins):**  
  - M1: <spec draft + tiny prototype> — <date> (share trace plots + initial robustness)  
  - M2: <baseline + monitor wired> — <date> (CPU run; table of metrics)  
  - M3: <enforcement/ablation> — <date> (show effect on task + robustness)  
  - M4: <results + short write‑up> — <date> (draft sections for final report)

## 8) Tasks (tick as you go)

- [ ] Survey prior art / alternatives (brief bullets)
- [ ] Data prep & licensing checks
- [ ] Implement monitors/specs
- [ ] Hook into training/eval loop
- [ ] Unit tests & quick integration test
- [ ] Docs & example configs
- [ ] Results & figures

## 9) Risks & mitigations

<List likely blockers (e.g., monitor runtime, gradient stability, dataset quality, Java availability for MoonLight, OS limits for SpaTiaL) **and** concrete fallbacks.>

## 10) Reproducibility details

- **Env/Deps:**  
  ```bash
  pip install -r requirements.txt                 # base (tiny, CI‑friendly)
  pip install -r requirements-extra.txt           # optional STL/STREL + physics stacks
  ```
- **Seeds & determinism:** <list seed(s); note any nondeterminism>  
- **How to run:**  
  ```bash
  python scripts/run_experiment.py --config <...>  # or a dedicated train/eval script
  ```

## 11) References (short list)

<Links to papers/code/datasets/specs you will use. Prefer primary sources.>

---

**Submitter:** @<you> • **Reviewers:** @maintainers  
<!-- Maintainer notes: label appropriately; confirm scope, compute, and alignment with course/report deliverables. -->
