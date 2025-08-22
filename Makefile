# ============================================================================
# physical-ai-stl — Makefile
# ----------------------------------------------------------------------------
# Purpose
#   • Give a *one-command* developer UX for this repository: create a venv,
#     install deps (lean or full), run tests, and execute the small CPU‑friendly
#     demos required for CS‑3860‑01 (Neuromancer/TorchPhysics/PhysicsNeMo +
#     RTAMT/MoonLight/SpaTiaL). The heavy stacks are optional.
#   • Everything here is *safe* to run on a laptop (CPU by default), and mirrors
#     CI behavior. Optional CUDA installs are available when desired.
#
# Design notes
#   • Cross‑platform: Linux/macOS, and Windows via Git Bash.
#   • Defensive shell: stop on errors; propagate failures through pipes.
#   • Tools like ruff/mypy/pre-commit are *optional* – recipes won’t fail if
#     they are not installed.
#   • Prefer `uv` when available for fast, hermetic installs; fall back to pip.
# ============================================================================
SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables

# ---- Python & package managers -------------------------------------------------
OS_NAME        := $(shell uname -s 2>/dev/null || echo Unknown)
PY             := $(shell command -v python3 >/dev/null 2>&1 && echo python3 || echo python)
PIP            := $(PY) -m pip
UV             := $(shell command -v uv 2>/dev/null || true)
HAVE_UV        := $(shell test -n "$(UV)" && echo 1 || echo 0)

# ---- Virtual environment -------------------------------------------------------
VENV_DIR       ?= .venv

# Cross‑platform activation: use bin/ on POSIX, Scripts/ on Windows (Git Bash).
# We *don’t* hardcode the path – we check both every time to avoid foot‑guns.
define ACTIVATE
if [ -f "$(VENV_DIR)/bin/activate" ]; then \
  . "$(VENV_DIR)/bin/activate"; \
elif [ -f "$(VENV_DIR)/Scripts/activate" ]; then \
  . "$(VENV_DIR)/Scripts/activate"; \
else \
  echo "❌ No virtualenv at '$(VENV_DIR)'. Run 'make venv' first." >&2; \
  exit 1; \
fi
endef

# ---- Torch channel selection ---------------------------------------------------
# cpu (default), or CUDA wheels e.g. cu118, cu121, cu124
TORCH_CHANNEL  ?= cpu

# ---- Paths --------------------------------------------------------------------
PY_SRC         := src
TESTS_DIR      := tests
RESULTS_DIR    ?= results
PLOTS_DIR      ?= plots
LOGS_DIR       ?= logs

# ---- Defaults for quick demos --------------------------------------------------
SEED           ?= 0
DEVICE         ?= cpu

# ---- Helper: quiet, deterministic pip output ----------------------------------
PIP_QUIET      := -q
export PIP_DISABLE_PIP_VERSION_CHECK := 1
export PYTHONDONTWRITEBYTECODE       := 1
export PYTHONUNBUFFERED              := 1
export PYTHONNOUSERSITE              := 1
export MPLBACKEND                    := Agg
export OMP_NUM_THREADS               := 1
export MKL_NUM_THREADS               := 1

# ---- Meta targets --------------------------------------------------------------
.DEFAULT_GOAL := help

## —— Help ——
.PHONY: help
help: ## Show this help
	@grep -E '(^[a-zA-Z0-9_.-]+:.*##)|(^##)' Makefile | \
		awk 'BEGIN {FS = ":.*?## "}; \
		     /^[^#].*:.*##/ {printf "\033[36m%-28s\033[0m %s\n", $$1, $$2}; \
		     /^##/ {gsub(/^##[ ]?/, "", $$0); print $$0}'

## —— Environment ——
.PHONY: env
env: ## Print environment & dependency summary
	@echo "OS           : $(OS_NAME)"
	@echo "Python       : $$($(PY) -V 2>&1)"
	@echo "Using uv     : $(HAVE_UV) ($(UV))"
	@echo "Venv         : $(VENV_DIR)"
	@echo "Torch channel: $(TORCH_CHANNEL)"
	@echo "—— Python executable paths ——"
	@which $(PY) || true
	@which $(PIP) || true
	@echo "—— Installed top‑level packages (short) ——"
	@$(PY) - <<'PY'
import importlib.util as I
def have(m): return I.find_spec(m) is not None
print("torch:", have("torch"),
      "| neuromancer:", have("neuromancer"),
      "| torchphysics:", have("torchphysics"),
      "| physicsnemo:", have("nvidia.physicsnemo") or have("physicsnemo"),
      "| rtamt:", have("rtamt"),
      "| moonlight:", have("moonlight"),
      "| spatial-spec:", have("spatial_spec"))
PY
	@echo "Tip: run 'make check' for a deeper probe."

.PHONY: check
check: ## Deep environment probe (runs scripts/check_env.py if available)
	@if [ -f scripts/check_env.py ]; then \
	  PYTHONPATH=$(PY_SRC) $(PY) scripts/check_env.py --summary; \
	else \
	  echo "scripts/check_env.py not found; skipping."; \
	fi

## —— Virtualenv ————————————————————————————————————————————————————————————
.PHONY: venv
venv: ## Create a local virtualenv in $(VENV_DIR)
	@if [ ! -d "$(VENV_DIR)" ]; then \
	  if [ "$(HAVE_UV)" = "1" ]; then \
	    $(UV) venv "$(VENV_DIR)" --python=$(PY); \
	  else \
	    $(PY) -m venv "$(VENV_DIR)"; \
	  fi; \
	fi
	@$(call ACTIVATE); $(PY) -m pip $(PIP_QUIET) install -U pip wheel

.PHONY: dirs
dirs: ## Create standard output directories
	@mkdir -p "$(RESULTS_DIR)" "$(PLOTS_DIR)" "$(LOGS_DIR)"

## —— Installation ————————————————————————————————————————————————————————
.PHONY: install
install: venv ## Install minimal runtime (requirements.txt)
	@$(call ACTIVATE); \
	if [ "$(HAVE_UV)" = "1" ]; then \
	  $(UV) pip install --python $(PY) -r requirements.txt; \
	else \
	  $(PIP) install -r requirements.txt; \
	fi

.PHONY: install-extras
install-extras: venv ## Install heavy extras (STL/STREL + physics‑ML toolkits)
	@$(call ACTIVATE); \
	if [ "$(HAVE_UV)" = "1" ]; then \
	  $(UV) pip install --python $(PY) -r requirements-extra.txt; \
	else \
	  $(PIP) install -r requirements-extra.txt; \
	fi

.PHONY: install-dev
install-dev: venv ## Install developer/test deps
	@$(call ACTIVATE); \
	if [ "$(HAVE_UV)" = "1" ]; then \
	  $(UV) pip install --python $(PY) -r requirements-dev.txt; \
	else \
	  $(PIP) install -r requirements-dev.txt; \
	fi
	@$(call ACTIVATE); $(PY) -m pip $(PIP_QUIET) install -U ruff mypy pre-commit || true
	@$(call ACTIVATE); $(PY) -m pre_commit install || true

.PHONY: install-all
install-all: install install-extras install-dev ## Install everything (runtime + extras + dev)

## ---- Torch wheels (explicit channel to avoid huge CUDA by default) ----------
.PHONY: install-torch-cpu
install-torch-cpu: venv ## Install PyTorch CPU wheels
	@$(call ACTIVATE); \
	if [ "$(HAVE_UV)" = "1" ]; then \
	  $(UV) pip install --python $(PY) --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio; \
	else \
	  $(PIP) install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio; \
	fi

.PHONY: install-torch-cuda
install-torch-cuda: venv ## Install PyTorch CUDA wheels (set TORCH_CHANNEL=cu121, cu118, cu124, ...)
	@if [ "$(TORCH_CHANNEL)" = "cpu" ]; then \
	  echo "Set TORCH_CHANNEL=cu121 (or cu118, cu124, ...) for CUDA wheels." && exit 1; \
	fi
	@$(call ACTIVATE); \
	if [ "$(HAVE_UV)" = "1" ]; then \
	  $(UV) pip install --python $(PY) --index-url https://download.pytorch.org/whl/$(TORCH_CHANNEL) torch torchvision torchaudio; \
	else \
	  $(PIP) install --index-url https://download.pytorch.org/whl/$(TORCH_CHANNEL) torch torchvision torchaudio; \
	fi

## —— Developer quality gates ————————————————————————————————————————————
.PHONY: format
format: ## Auto-format with Ruff (if installed)
	-@$(call ACTIVATE); $(PY) -m ruff format .

.PHONY: lint
lint: ## Lint with Ruff (if installed)
	-@$(call ACTIVATE); $(PY) -m ruff check .

.PHONY: typecheck
typecheck: ## Static type check with mypy (if installed)
	-@$(call ACTIVATE); $(PY) -m mypy src || true

.PHONY: precommit
precommit: ## Run pre-commit hooks across repo (if installed)
	-@$(call ACTIVATE); $(PY) -m pre_commit run --all-files || true

## —— Tests ———————————————————————————————————————————————————————————————
PYTEST_ARGS ?= -q
.PHONY: test
test: ## Run full test suite (CPU‑friendly; optional deps skip automatically)
	@$(call ACTIVATE); PYTHONPATH=$(PY_SRC) $(PY) -m pytest $(PYTEST_ARGS)

.PHONY: test-fast
test-fast: ## Run the quick smoke tests (hello demos + core utilities)
	@$(call ACTIVATE); PYTHONPATH=$(PY_SRC) $(PY) -m pytest -q -k "hello or pde"

.PHONY: coverage
coverage: ## Run tests with coverage
	@$(call ACTIVATE); PYTHONPATH=$(PY_SRC) $(PY) -m pytest --cov=physical_ai_stl --cov-report=term-missing -q

## —— Experiments & monitoring ————————————————————————————————————————————
EXTRA_FLAGS ?=
.PHONY: diffusion1d
diffusion1d: dirs ## Train & evaluate the 1D diffusion PINN baseline + STL audit
	@$(call ACTIVATE); PYTHONPATH=$(PY_SRC) $(PY) scripts/run_experiment.py -c configs/diffusion1d_baseline.yaml $(EXTRA_FLAGS)
	@$(call ACTIVATE); PYTHONPATH=$(PY_SRC) $(PY) scripts/run_experiment.py -c configs/diffusion1d_stl.yaml $(EXTRA_FLAGS)

.PHONY: heat2d
heat2d: dirs ## Train & evaluate the 2D heat demo + STREL audit (MoonLight)
	@$(call ACTIVATE); PYTHONPATH=$(PY_SRC) $(PY) scripts/run_experiment.py -c configs/heat2d_baseline.yaml $(EXTRA_FLAGS)

# Neuromancer/Torch tiny demo: sine fit with an STL-style upper bound
EPOCHS ?= 200
LR     ?= 1e-3
BOUND  ?= 0.8
WEIGHT ?= 10.0
N      ?= 1024
MODE   ?= torch        # torch | neuromancer (if available)
.PHONY: neuromancer-sine
neuromancer-sine: dirs ## Tiny Neuromancer/Torch demo with soft STL-style bound
	@$(call ACTIVATE); PYTHONPATH=$(PY_SRC) $(PY) scripts/train_neuromancer_stl.py \
	  --epochs $(EPOCHS) --lr $(LR) --bound $(BOUND) --weight $(WEIGHT) \
	  --n $(N) --seed $(SEED) --device $(DEVICE) --mode $(MODE) --out $(RESULTS_DIR)

.PHONY: rtamt-eval
rtamt-eval: ## Evaluate STL robustness using RTAMT on saved diffusion field
	@$(call ACTIVATE); PYTHONPATH=$(PY_SRC) $(PY) scripts/eval_diffusion_rtamt.py --ckpt $(RESULTS_DIR)/diffusion1d_field.pt || true

.PHONY: moonlight-eval
moonlight-eval: ## Evaluate STREL properties with MoonLight on heat2d outputs
	@$(call ACTIVATE); PYTHONPATH=$(PY_SRC) $(PY) scripts/eval_heat2d_moonlight.py --ckpt $(RESULTS_DIR)/heat2d_field.pt || true

.PHONY: ablations
ablations: dirs ## Run light ablations for diffusion1d (epochs/penalty sweeps)
	@$(call ACTIVATE); PYTHONPATH=$(PY_SRC) $(PY) scripts/run_ablations_diffusion.py --out $(RESULTS_DIR)/ablations.csv
	@$(call ACTIVATE); PYTHONPATH=$(PY_SRC) $(PY) scripts/plot_ablations.py --csv $(RESULTS_DIR)/ablations.csv --out $(PLOTS_DIR)

.PHONY: survey
survey: ## Summarize framework feature survey to docs/framework_survey.md
	@$(call ACTIVATE); PYTHONPATH=$(PY_SRC) $(PY) scripts/framework_survey.py --out docs/framework_survey.md || true

## —— Java (MoonLight) ————————————————————————————————————————————————————
.PHONY: java-check
java-check: ## Check if Java is available (required for MoonLight/STREL)
	@java -version || echo "Java not found; consider 'pip install install-jdk' or a system JDK (see requirements-extra.txt)."

## —— Containers ———————————————————————————————————————————————————————————
.PHONY: docker-build
docker-build: ## Build minimal CPU image (no heavy extras)
	@docker build -t pai-stl .

.PHONY: docker-build-full
docker-build-full: ## Build full image with extras (STL/STREL + physics‑ML); add WITH_DEV=1 for dev tools
	@docker build --build-arg WITH_EXTRAS=1 --build-arg WITH_JAVA=1 -t pai-stl:full .

.PHONY: docker-run
docker-run: ## Run the image interactively mounting the repo
	@docker run --rm -it -v "$$(pwd)":/work -w /work pai-stl:full bash

## —— Housekeeping ———————————————————————————————————————————————————————
.PHONY: clean
clean: ## Remove Python caches and temporary files
	@find . -name "__pycache__" -type d -exec rm -rf {} + || true
	@rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov || true

.PHONY: dist-clean
dist-clean: clean ## Also remove venv, build artifacts, and results
	@rm -rf "$(VENV_DIR)" build dist *.egg-info "$(RESULTS_DIR)" "$(PLOTS_DIR)" "$(LOGS_DIR)" || true

## —— One‑shot bootstrap for newcomers ————————————————————————————————
.PHONY: quickstart
quickstart: venv install-all install-torch-cpu dirs env test-fast ## Create venv, install everything (CPU), smoke-test
	@echo "✅ Quickstart complete."

## —— CI parity (mirrors .github/workflows/ci.yml) ————————————————
.PHONY: ci
ci: ## Minimal steps used in GitHub Actions CI
	@if [ "$(HAVE_UV)" = "1" ]; then \
	  $(UV) pip install --python $(PY) -r requirements.txt -r requirements-dev.txt; \
	else \
	  $(PIP) install -r requirements.txt -r requirements-dev.txt; \
	fi
	@PYTHONPATH=$(PY_SRC) $(PY) -m pytest -q
