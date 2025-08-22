.PHONY: help test lint train1d audit1d heat2d ablate plot env survey

help:
        @echo "Targets: test, lint, train1d, audit1d, heat2d, ablate, plot, env, survey"

test: ; pytest -q
lint: ; ruff check .
train1d: ; python scripts/train_diffusion_stl.py
audit1d: ; python scripts/eval_diffusion_rtamt.py
heat2d: ; python scripts/train_heat2d_strel.py
ablate: ; python scripts/run_ablations_diffusion.py --weights 0.0 0.1 0.5 1.0 --epochs 100
plot: ; python scripts/plot_ablations.py --csv results/ablations_diffusion.csv --out figs/ablations_diffusion.png
env: ; python scripts/check_env.py
survey: ; python scripts/framework_survey.py
