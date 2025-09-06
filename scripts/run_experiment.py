"""Generic wrapper to run experiments from YAML configuration files."""

from __future__ import annotations

from typing import Any, Dict
import argparse
import os
def _load_yaml(path: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        raise SystemExit(
            "Missing dependency: pyyaml. Install it with 'pip install pyyaml' "
            "or 'pip install -r requirements-extra.txt'."
        ) from e
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    exp = str(cfg.get("experiment", "")).lower().strip()
    os.makedirs(cfg.get("io", {}).get("results_dir", "results"), exist_ok=True)

    if exp == "diffusion1d":
        from physical_ai_stl.experiments.diffusion1d import run_diffusion1d
        out = run_diffusion1d(cfg)
        print(f"[diffusion1d] done → {out}")
    elif exp == "heat2d":
        from physical_ai_stl.experiments.heat2d import run_heat2d
        out = run_heat2d(cfg)
        print(f"[heat2d] done → {out}")
    else:
        raise SystemExit(f"Unknown experiment: {exp}")

if __name__ == "__main__":
    main()