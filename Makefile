# ------------------------------------------------------------
# physical-ai-stl — Project Makefile
# Goals:
#   • Reproducible, fast developer UX (venv, install, lint, test)
#   • Turnkey experiments (diffusion1d, heat2d) + audits (RTAMT/MoonLight)
#   • One-line ablations/plots and config-driven runs
#   • Cross‑platform (Linux/macOS; Windows via Git Bash)
# ------------------------------------------------------------

# --- Strict shell ------------------------------------------------------------
SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables

# --- Paths & tools -----------------------------------------------------------
VENV_DIR       ?= .venv
# Prefer venv Python if present
PY             := $(shell [ -x "$(VENV_DIR)/bin/python" ] && printf "$(VENV_DIR)/bin/python" || command -v python3 || command -v python)
PIP            := $(PY) -m pip
UV             ?= uv                         # Optional: ultra‑fast installer if available
HAVE_UV        := $(shell command -v $(UV) >/dev/null 2>&1 && echo 1 || echo 0)

RESULTS_DIR    ?= results
FIGS_DIR       ?= figs
CFG_DIR        ?= configs
SCRIPTS_DIR    ?= scripts

# CPU-friendly defaults for laptop dev; override as needed
export OMP_NUM_THREADS ?= 1
export MKL_NUM_THREADS ?= 1

# --- Default target ----------------------------------------------------------
.DEFAULT_GOAL := help

# Helper to print pretty help from '##' comments.
_help = grep -E '^[a-zA-Z0-9_.-]+:.*##' $(firstword $(MAKEFILE_LIST)) | \
        sed -E 's/:.*##/:/g' | \
        awk -F':' '{ printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 }'

# --- Meta --------------------------------------------------------------------
.PHONY: help
help: ## Show this help
	@echo "Targets:"
	@$(call _help)

.PHONY: all
all: format lint test ## Format, lint, then test

# --- Environment setup -------------------------------------------------------
.PHONY: venv
venv: ## Create a local virtual environment in $(VENV_DIR)
	$(shell command -v python3 >/dev/null 2>&1 || { echo "python3 not found"; exit 1; })
	python3 -m venv $(VENV_DIR)
	$(PIP) install -U pip wheel

.PHONY: install-min
install-min: ## Install minimal runtime (fast): requirements.txt + package
	$(if $(filter 1,$(HAVE_UV)),\
		$(UV) pip install --python $(PY) -r requirements.txt;\
		$(UV) pip install --python $(PY) -e .,\
		$(PIP) install -r requirements.txt && $(PIP) install -e .)

.PHONY: install-dev
install-dev: ## Install dev tooling (pytest/coverage) from requirements-dev.txt
	$(if $(filter 1,$(HAVE_UV)),\
		$(UV) pip install --python $(PY) -r requirements-dev.txt,\
		$(PIP) install -r requirements-dev.txt)

.PHONY: install-extras
install-extras: ## Install heavy extras (STL/STREL, frameworks, plotting utils)
	$(if $(filter 1,$(HAVE_UV)),\
		$(UV) pip install --python $(PY) -r requirements-extra.txt,\
		$(PIP) install -r requirements-extra.txt)

.PHONY: install-all
install-all: install-min install-dev install-extras ## Install everything (CPU)

.PHONY: install-torch-cpu
install-torch-cpu: ## Install CPU PyTorch wheels directly (avoids CUDA downloads)
	$(PIP) install --index-url https://download.pytorch.org/whl/cpu torch

.PHONY: install-torch-cu121
install-torch-cu121: ## Install CUDA 12.1 PyTorch (Linux, if you have NVIDIA CUDA)
	$(PIP) install --index-url https://download.pytorch.org/whl/cu121 torch

.PHONY: env
env: ## Print availability of optional deps (rtamt, moonlight, spatial, neuromancer, ...)
	$(PY) $(SCRIPTS_DIR)/check_env.py

# --- Code quality ------------------------------------------------------------
.PHONY: format
format: ## Auto-format with Ruff (if installed)
	-$(PY) -m ruff format .

.PHONY: lint
lint: ## Lint with Ruff (if installed)
	-$(PY) -m ruff check .

.PHONY: typecheck
typecheck: ## Static type check with mypy (if installed)
	-$(PY) -m mypy src || true

.PHONY: precommit
precommit: ## Run pre-commit hooks across repo (if installed)
	-$(PY) -m pre_commit run --all-files || true

# --- Testing -----------------------------------------------------------------
.PHONY: test
test: ## Run the full test suite (pytest -q)
	$(PY) -m pytest -q

.PHONY: test-fast
test-fast: ## Run only “hello/smoke” tests (fast)
	$(PY) -m pytest -q tests/test_*hello.py -q

.PHONY: coverage
coverage: ## Run tests with coverage report
	$(PY) -m pytest --cov=src --cov-report=term-missing

# --- Config-driven experiments ----------------------------------------------
CFG ?=

.PHONY: run
run: ## Run an experiment from a YAML config, e.g., `make run CFG=configs/diffusion1d_stl.yaml`
	@test -n "$(CFG)" || (echo "Set CFG=path/to/config.yaml"; exit 2)
	$(PY) $(SCRIPTS_DIR)/run_experiment.py --cfg $(CFG)

# --- 1D Diffusion PINN -------------------------------------------------------
ARGS ?=
.PHONY: train1d
train1d: ## Train diffusion1d (STL penalty available) — pass ARGS='--epochs 200 --weight 0.5'
	$(PY) $(SCRIPTS_DIR)/train_diffusion_stl.py $(ARGS)

.PHONY: audit1d
audit1d: ## Audit diffusion1d checkpoint with RTAMT (robustness of G(mean_x u ≤ u_max))
	-$(PY) $(SCRIPTS_DIR)/eval_diffusion_rtamt.py $(ARGS) || { \
		echo "Hint: install RTAMT (pip install rtamt) to enable this audit."; }

# --- 2D Heat PINN ------------------------------------------------------------
.PHONY: heat2d
heat2d: ## Train heat2d PINN (no MoonLight required)
	$(PY) $(SCRIPTS_DIR)/train_heat2d_strel.py $(ARGS)

.PHONY: audit2d
audit2d: ## Audit heat2d frames with MoonLight STREL spec (requires Java 21+)
	-$(PY) $(SCRIPTS_DIR)/eval_heat2d_moonlight.py $(ARGS) || { \
		echo "Hint: install moonlight (pip install moonlight) and Java 21+."; }

# --- Ablations & plots -------------------------------------------------------
.PHONY: ablate
ablate: ## Sweep STL weights for diffusion1d and log CSV
	$(PY) $(SCRIPTS_DIR)/run_ablations_diffusion.py --weights 0.0 0.1 0.5 1.0 --epochs 100

.PHONY: plot
plot: ## Plot ablation CSV → PNG
	$(PY) $(SCRIPTS_DIR)/plot_ablations.py --csv $(RESULTS_DIR)/ablations_diffusion.csv --out $(FIGS_DIR)/ablations_diffusion.png

# --- Surveys / utilities -----------------------------------------------------
.PHONY: survey
survey: ## Print versions of frameworks & STL tooling discovered
	$(PY) $(SCRIPTS_DIR)/framework_survey.py

.PHONY: neuromancer-demo
neuromancer-demo: ## Tiny Neuromancer training demo w/ STL-style bound (writes JSON)
	$(PY) $(SCRIPTS_DIR)/train_neuromancer_stl.py --config $(CFG_DIR)/neuromancer_sine_bound.yaml --out $(RESULTS_DIR)/neuromancer_sine.json

.PHONY: torchphysics-burgers
torchphysics-burgers: ## Placeholder Burgers' PINN with TorchPhysics (writes a dummy ckpt)
	$(PY) $(SCRIPTS_DIR)/train_burgers_torchphysics.py --results $(RESULTS_DIR) --tag demo

# --- Housekeeping ------------------------------------------------------------
.PHONY: dirs
dirs: ## Create result/fig directories
	mkdir -p $(RESULTS_DIR) $(FIGS_DIR)

.PHONY: clean
clean: ## Remove caches and temporary artifacts
	rm -rf .pytest_cache .ruff_cache __pycache__ **/__pycache__ $(RESULTS_DIR)/*.tmp

.PHONY: distclean
distclean: clean ## Remove build artifacts, venv, and generated outputs
	rm -rf build dist *.egg-info $(VENV_DIR) $(RESULTS_DIR) $(FIGS_DIR)

# --- Quickstart recipes ------------------------------------------------------
.PHONY: quickstart
quickstart: venv install-all install-torch-cpu dirs env test-fast ## Create venv, install everything (CPU), smoke-test
	@echo "✅ Quickstart complete."

# --- CI parity (mirrors .github/workflows/ci.yml) ---------------------------
.PHONY: ci
ci: ## Minimal steps used in GitHub Actions CI
	@if [ "$(HAVE_UV)" = "1" ]; then \
	  $(UV) pip install --python $(PY) -r requirements.txt -r requirements-dev.txt; \
	else \
	  $(PIP) install -r requirements.txt -r requirements-dev.txt; \
	fi
	$(PY) -m pytest -q
