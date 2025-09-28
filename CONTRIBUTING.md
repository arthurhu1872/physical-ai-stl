# Contributing

## Dev quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest -q   # pytest >= 8 is expected
```

## Optional stacks
Install heavy deps only when needed:
```bash
pip install -r requirements-extra.txt
```

## Style & tests
- Lint/format: ruff (via pre-commit), mypy (non-blocking)
- Tests should *skip gracefully* when optional deps aren’t present.
- Keep unit tests fast (seconds).

## Branch/PR
- Feature branches: `feature/<topic>`
- Small, reviewable PRs with a clear checklist and CI green.
