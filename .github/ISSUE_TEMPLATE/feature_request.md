---
name: "Feature / Experiment (Physical‑AI × STL/STREL)"
about: "Propose a tightly‑scoped, reproducible feature or experiment advancing Physical‑AI with STL/STREL monitoring for CS‑3860‑01."
title: "[feat] <short, imperative title>"
labels: ["enhancement","research"]
assignees: []
---

<!--
READ FIRST — to keep this repo lean, reproducible, and aligned with CS‑3860‑01:

1) Skim **README** (Goals, Quickstart) and **CONTRIBUTING** before filing.
2) Keep the **base** environment tiny; gate heavy stacks (PyTorch, Java toolchains, vendor SDKs) behind `requirements-extra.txt` only.
3) Prefer fast **CPU‑first** demos; all tests must **skip gracefully** when optional deps are absent.
4) Use **public, license‑compatible** datasets only; include links + licenses; never commit large data.
5) Aim at the **end‑of‑semester report**; propose concrete **Friday** check‑ins and milestones.
6) Provide a **CPU‑only path** or clear fallback for everything you propose (smoke test ≤ ~5 minutes, ≤ ~4 GB RAM).
7) Results must be **reproducible** (fixed seeds, logged configs); document any nondeterminism.
-->

### ✅ Pre‑flight checklist (tick before submitting)
- [ ] CPU‑only **smoke test** exists (≤ ~5 min, ≤ ~4 GB RAM) and is described below
- [ ] Datasets are **public** and **license‑compatible** (links + license noted)
- [ ] STL/STREL specification(s) include **units**, horizons, and robustness thresholds
- [ ] Heavy deps isolated in `requirements-extra.txt`; base `requirements.txt` stays **small**
- [ ] At least one **baseline** (no STL) and one **STL/STREL** variant are planned
- [ ] Plan includes **Friday** cadence & end‑of‑semester deliverable

## 1) Summary

**One‑liner**  
<What is the feature/experiment?>

**Problem & motivation (why now?)**  
<What research/engineering gap does this close? Why are STL/STREL the right tools?>

**Measurable outcome / hypothesis**  
<What should improve? e.g., task error ↓, robustness margin ↑, constraint violations ↓>

## 2) Research alignment

- **Framework(s):** ☐ Neuromancer ☐ PhysicsNeMo ☐ TorchPhysics ☐ Other: <name>  
- **STL tooling:** ☐ RTAMT (STL) ☐ MoonLight (STREL) ☐ SpaTiaL ☐ Other: <name>  
- **Category:** ☐ Monitor‑only ☐ Train‑time soft enforcement ☐ Post‑hoc evaluation ☐ Dataset/problem integration

<Briefly state how this advances the Physical‑AI + STL/STREL agenda for the course.>

## 3) Problem space / dataset

- **System / PDE/ODE/CPS:** <e.g., 1D diffusion, 2D heat, Burgers, cart‑pole, traffic sensors, etc.>  
- **Dataset/source & license:** <URL + license>; **size/shape:** <e.g., 10k samples, 64×64×T>  
- **Spatial layout / graph / mesh:** <grid, irregular, adjacency/neighbor definition>  
- **Why STL/STREL here:** <which properties matter? bounds, reach/escape, smoothness, propagation speed, causality, etc.>

## 4) Specification(s)

Provide the temporal or spatio‑temporal properties you will monitor/enforce.

- **Informal (plain English):**  
  <e.g., “temperature stays in [0,1] until the source is off; any hotspot must dissipate within 5 s at most 2 cells away.”>  
- **Formal STL/STREL:**  
  ```txt
  <phi := G_[t1,t2](a → F_[τ1,τ2] b)    # use STREL reach/escape for spatial fields if needed>
  ```
- **Robustness semantics & thresholds:** <how evaluated; normalization/margins; units>  
- **Monitor runtime mode:** ☐ offline (batch) ☐ online (bounded‑future) ☐ streaming  
- **Differentiable proxy (if training‑time):** <e.g., smooth min/log‑sum‑exp; temperature; β/τ value>

## 5) Proposed approach

- **Integration points:** <where monitors plug into training/inference/logging>  
- **Model/architecture:** <PINN/DeepONet/FNO/Neural ODE/PDE/etc.>  
- **Enforcement strategy (if any):** <loss shaping with soft robustness, penalties, shields, projection, etc.>  
- **Compute plan:** <CPU/GPU; est. memory/runtime on CPU; include CPU‑only fallback>  
- **Complexity budget:** <max wall‑clock per run; target ≤ minutes on CPU for smoke tests>  
- **Artifacts to log:** <configs, seeds, metrics, robustness traces, figures>

## 6) Acceptance criteria — Definition of Done (DoD)

- [ ] Reproducible script(s) under `scripts/` with argparse + README usage
- [ ] STL/STREL specs encoded and **unit‑tested** (robustness values sanity‑checked)
- [ ] ≥1 **baseline** (no STL) vs ≥1 **STL/STREL** variant, with metrics (task + robustness)
- [ ] Results table/figure with seeds noted (CSV or Markdown)
- [ ] Minimal deps in `requirements.txt`; heavy extras in `requirements-extra.txt`
- [ ] Tests (`pytest`) are **fast** and **skip** when optional stacks are absent
- [ ] Short docs: what it does, how to run, expected time/resources
- [ ] (Optional) Ablations: spec strength, monitor/enforcement variants
- [ ] (Optional) CI hook: tiny skip‑aware test

## 7) Plan & milestones

- **Weekly cadence & load:** <hrs/week; guidance: 2–3 hrs/credit ⇒ ~6–9 hrs/wk for 3 credits>  
- **Milestones & dates (Fri check‑ins):**  
  - M1: <spec draft + tiny prototype> — <date> (trace plots + initial robustness)  
  - M2: <baseline + monitor wired> — <date> (CPU run; table of metrics)  
  - M3: <enforcement/ablation> — <date> (effect on task + robustness)  
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

<List likely blockers (e.g., monitor runtime, gradient stability, dataset quality, **Java availability for MoonLight**, OS limits for SpaTiaL, vendor SDK quirks) **and** concrete fallbacks (alternate tool, weaker spec, smaller domain, synthetic data).>

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
