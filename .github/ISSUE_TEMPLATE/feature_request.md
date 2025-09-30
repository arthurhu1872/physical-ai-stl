---
name: "Feature / Experiment request (Physical-AI + STL)"
about: "Propose a new feature or experiment that advances Physical‑AI with STL/STREL monitoring in this repo (CS‑3860‑01)."
title: "[feat] <short, imperative title>"
labels: ["enhancement","research"]
assignees: []
---

<!--
Before filing, please:
1) Read README goals and CONTRIBUTING.
2) Keep base installs lean; gate heavy deps in requirements-extra.txt.
3) Prefer fast, CPU‑friendly demos; add tests that skip gracefully if optional deps missing.
4) Ensure datasets are public and license‑compatible; include citations/links.
5) Target an end‑of‑semester report; propose concrete milestones/check‑ins.
-->

## 1) Summary

**One‑liner:**  
<What is the feature/experiment?>

**Problem it addresses (why now):**  
<What research/engineering gap does this close?>

## 2) Research alignment

- **Framework(s):** ☐ Neuromancer ☐ PhysicsNeMo ☐ TorchPhysics ☐ Other: <name>  
- **STL tooling:** ☐ RTAMT (STL) ☐ MoonLight (STREL) ☐ SpaTiaL ☐ Other: <name>  
- **Category:** ☐ Monitor only ☐ Train‑time enforcement ☐ Post‑hoc evaluation ☐ Dataset/problem integration

<Briefly state how this advances the Physical‑AI + STL/STREL agenda for the course.>

## 3) Problem space / dataset

- **System / PDE/ODE:** <e.g., 1D diffusion, 2D heat, Burgers, cart‑pole, etc.>  
- **Dataset/source & license:** <URL + license>; size & shape: <e.g., 10k samples, 64×64×T>  
- **Spatial layout / graph / mesh:** <grid, irregular, neighbors definition>  
- **Relevance of STL/STREL here:** <what properties matter?>

## 4) Specification(s)

Provide the temporal / spatio‑temporal properties you will monitor/enforce.

- **Informal:** <plain English description>  
- **Formal STL/STREL:**  
  ```txt
  <phi := G_[t1,t2](a → F_[τ1,τ2] b) ; STREL reach/escape if spatial>
  ```
- **Robustness semantics & thresholds:** <how evaluated, normalization, margins>  
- **Monitor runtime mode:** ☐ offline (batch) ☐ online (bounded‑future) ☐ streaming

## 5) Proposed approach

- **Integration points:** <where monitors plug into training/inference>  
- **Model/architecture:** <PINN/DeepONet/FNO/Neural ODE/PDE, etc.>  
- **Enforcement strategy (if any):** <loss shaping, penalties, shields, projection, etc.>  
- **Compute plan:** <CPU/GPU, est. memory/runtime; include CPU‑only fallback if possible>

## 6) Acceptance criteria (Definition of Done)

- [ ] Reproducible script(s) under `scripts/` with argparse and README notes
- [ ] STL/STREL specs encoded and unit‑tested (robustness values sanity‑checked)
- [ ] Results table/figure with metrics (task + robustness), seeds noted
- [ ] Minimal dependencies in `requirements.txt`; heavy extras in `requirements-extra.txt`
- [ ] Tests (`pytest`) are fast and skip when optional stacks are absent
- [ ] Short docs: what it does, how to run, expected time/resources
- [ ] (Optional) Ablations: spec strength, monitor/enforcement variants

## 7) Plan & milestones

- **Weekly cadence & load:** <hrs/week; default guidance: 2–3 hrs per credit (≈6–9 hrs for 3 cr.)>  
- **Milestones & dates:**  
  - M1: <spec draft + tiny prototype> — <date>
  - M2: <baseline + monitor> — <date>
  - M3: <enforcement/ablation> — <date>
  - M4: <results + short write‑up> — <date>

## 8) Tasks

- [ ] Survey prior art / alternatives (brief bullets)
- [ ] Data prep & licensing checks
- [ ] Implement monitors/specs
- [ ] Hook into training/eval loop
- [ ] Unit tests & quick integration test
- [ ] Docs & example configs
- [ ] Results & figures

## 9) Risks & mitigations

<List likely blockers (e.g., monitor runtime, gradient stability, dataset quality) and concrete fallbacks.>

## 10) Reproducibility details

- **Env/Deps:**  
  ```bash
  pip install -r requirements.txt    # base
  pip install -r requirements-extra.txt  # optional heavy stacks
  ```
- **Seeds & determinism:** <list seed(s) and any nondeterminism caveats>  
- **How to run:**  
  ```bash
  python scripts/run_experiment.py --config <...>  # or specific train/eval script
  ```

## 11) References

<Links to papers/code/datasets/specs you will use (short list).>

---

**Submitter:** @<you> • **Reviewers:** @maintainers  
<!-- Maintainer notes: label appropriately; confirm scope, resources, and alignment with course/report deliverables. -->
