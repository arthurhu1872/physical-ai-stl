#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import pkgutil
import sys
import time
from collections.abc import Iterable, Iterator
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# YAML loader (with a crisp error if pyyaml isn't installed)
# ---------------------------------------------------------------------------


def _require_yaml():
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover - friendly fatal
        raise SystemExit(
            "Missing dependency: pyyaml. Install it with:\n"
            "  pip install pyyaml\n"
            "or via extras:\n"
            "  pip install -r requirements-extra.txt"
        ) from e
    return yaml


def _read_text(path: str) -> str:
    path = os.path.expanduser(os.path.expandvars(path))
    with open(path, encoding="utf-8") as f:  # UP015: no redundant mode="r"
        return f.read()


def load_yaml(path: str) -> dict[str, Any]:
    yaml = _require_yaml()
    raw = _read_text(path)

    # Expand env vars *inside* scalar strings post-parse to avoid YAML surprises
    data = yaml.safe_load(raw) or {}

    def _expand(obj: Any) -> Any:
        if isinstance(obj, str):
            return os.path.expandvars(os.path.expanduser(obj))
        if isinstance(obj, list | tuple):  # UP038: union in isinstance
            return type(obj)(_expand(v) for v in obj)
        if isinstance(obj, dict):
            return {k: _expand(v) for k, v in obj.items()}
        return obj

    return _expand(data)


# ---------------------------------------------------------------------------
# Small, dependency-free experiment registry using module discovery
# ---------------------------------------------------------------------------


def _ensure_src_on_path() -> None:
    try:
        import physical_ai_stl  # noqa: F401
        return
    except Exception:
        pass
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, os.pardir))
    src = os.path.join(repo_root, "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


@dataclass(frozen=True)
class ExpInfo:
    name: str  # e.g. 'diffusion1d'
    module: str  # 'physical_ai_stl.experiments.diffusion1d'
    run_candidates: tuple[str, ...]  # ordered possible run function names


def discover_experiments() -> list[ExpInfo]:
    _ensure_src_on_path()
    try:
        pkg = importlib.import_module("physical_ai_stl.experiments")
    except Exception as e:
        raise SystemExit(
            "Cannot import 'physical_ai_stl.experiments'. If running from a clone,\n"
            "ensure the repository root's 'src/' is on PYTHONPATH or install the\n"
            "package (e.g., 'pip install -e .').\n\n"
            f"Original error: {e}"
        ) from e

    infos: list[ExpInfo] = []
    for modinfo in pkgutil.iter_modules(pkg.__path__):  # type: ignore[attr-defined]
        name = modinfo.name
        module = f"physical_ai_stl.experiments.{name}"  # consider only modules for now
        # Conventional run function order: run_<name>, run
        candidates = (f"run_{name}", "run")
        infos.append(ExpInfo(name=name, module=module, run_candidates=candidates))
    # Stable alphabetical order for --list
    return sorted(infos, key=lambda i: i.name)


def get_runner(exp_name: str):
    infos = {i.name: i for i in discover_experiments()}
    if exp_name not in infos:
        available = ", ".join(sorted(infos))
        raise SystemExit(f"Unknown experiment '{exp_name}'. Available: [{available}]")
    info = infos[exp_name]
    mod = importlib.import_module(info.module)
    for fn in info.run_candidates:
        if hasattr(mod, fn):
            return getattr(mod, fn)
    raise SystemExit(
        f"No runnable function found in {info.module}. Tried: {info.run_candidates}"
    )


# ---------------------------------------------------------------------------
# Config utilities: dotted overrides and tiny sweep helper
# ---------------------------------------------------------------------------


def _parse_override(s: str) -> tuple[list[str], Any]:
    if "=" not in s:
        raise argparse.ArgumentTypeError("--set expects KEY=VALUE (use quotes for lists)")
    key, val = s.split("=", 1)
    key_parts = [k for k in key.split(".") if k]
    if not key_parts:
        raise argparse.ArgumentTypeError(f"Invalid key in override: {s}")
    # Reuse YAML parser to get numbers/bools/lists right
    yaml = _require_yaml()
    value = yaml.safe_load(val)
    return key_parts, value


def _set_nested(d: dict[str, Any], keys: list[str], value: Any) -> None:
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]  # type: ignore[assignment]
    cur[keys[-1]] = value


def apply_overrides(cfg: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    cfg = deepcopy(cfg)
    for o in overrides:
        keys, value = _parse_override(o)
        _set_nested(cfg, keys, value)
    return cfg


def iter_sweep_cfgs(base: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    sweep = base.get("sweep")
    if not sweep:
        yield "", base
        return

    # Normalize: keys (dotted) -> lists of values
    items: list[tuple[list[str], list[Any]]] = []
    for k, v in sweep.items():
        keys = [p for p in str(k).split(".") if p]
        if not isinstance(v, list) or len(v) == 0:
            raise SystemExit(f"Each sweep entry must be a non-empty list: {k}")
        items.append((keys, v))

    # Cartesian product
    from itertools import product

    for combo in product(*[vals for _, vals in items]):
        cfg = deepcopy(base)
        parts: list[str] = []
        for (keys, _), v in zip(items, combo, strict=True):
            _set_nested(cfg, keys, v)
            # Build a compact, file-system-safe suffix
            sval = repr(v).replace(" ", "")
            sval = sval.replace("/", "-").replace(os.sep, "-")
            parts.append(f"{'.'.join(keys)}={sval}")
        yield "__".join(parts), cfg


# ---------------------------------------------------------------------------
# Seeding and run directory handling (lightweight / optional)
# ---------------------------------------------------------------------------


def try_set_seed(seed: int | None) -> None:
    if seed is None:
        return
    try:
        import random

        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # no-op if CUDA absent
        # Optional deterministic flags if available
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(False)  # safer default for speed
    except Exception:
        pass


def make_run_dir(cfg: dict[str, Any]) -> str:
    io_cfg = cfg.setdefault("io", {})
    results_dir = str(io_cfg.get("results_dir", "results"))
    os.makedirs(results_dir, exist_ok=True)
    exp = str(cfg.get("experiment", "")).strip() or "exp"
    tag = str(cfg.get("tag", "run")).strip() or "run"
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(results_dir, f"{exp}--{tag}--{ts}")
    os.makedirs(run_dir, exist_ok=True)
    io_cfg["run_dir"] = run_dir
    return run_dir


def dump_effective_config(run_dir: str, cfg: dict[str, Any]) -> None:
    yaml = _require_yaml()
    path = os.path.join(run_dir, "config.effective.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generic runner for Physical-AI–STL experiments (YAML-driven).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", "-c", required=False, help="Path to YAML config.")
    p.add_argument(
        "--list",
        action="store_true",
        help="List available experiments and exit.",
    )
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help=(
            "Override config with KEY=VALUE (YAML values). "
            "Can be repeated."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print the resolved experiment without running.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)

    if args.list:
        infos = discover_experiments()
        print("Available experiments:")
        for i in infos:
            print(f"  - {i.name}  (module: {i.module})")
        return

    if not args.config:
        raise SystemExit("--config is required unless using --list")

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.overrides)

    # Experiment name
    exp = str(cfg.get("experiment", "")).strip().lower()
    if not exp:
        # Small convenience: infer from config file name (e.g., diffusion1d_baseline.yaml)
        base = os.path.basename(os.path.splitext(args.config)[0])
        exp = base.split("_")[0].lower()
        cfg["experiment"] = exp

    # Seeding (best effort, optional)
    try_set_seed(cfg.get("seed"))

    # Optional sweep support
    ran_any = False
    for suffix, subcfg in iter_sweep_cfgs(cfg):
        run_dir = make_run_dir(subcfg)
        if suffix:
            subcfg.setdefault("io", {})["run_dir"] = os.path.join(run_dir, suffix)
            os.makedirs(subcfg["io"]["run_dir"], exist_ok=True)
            run_dir = subcfg["io"]["run_dir"]
        dump_effective_config(run_dir, subcfg)

        runner = get_runner(exp)
        if args.dry_run:
            tag = subcfg.get("tag", "")
            print(f"[DRY‑RUN] Would run: {exp} with tag='{tag}' in {run_dir}")
            continue

        out = runner(subcfg)  # type: ignore[misc]
        print(f"[{exp}] done → {out if out is not None else run_dir}")
        ran_any = True

    if args.dry_run and not ran_any:
        print("[DRY‑RUN] No runs were scheduled (empty sweep?).")


if __name__ == "__main__":  # pragma: no cover
    main()
