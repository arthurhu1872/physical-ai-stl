"""Print a quick summary of optional dependencies and their availability."""

from __future__ import annotations

import importlib
import sys
def _check(name: str) -> str:
    """Return a ✓/✗ string indicating whether ''name'' imports successfully."""
    try:
        importlib.import_module(name)
        return "✅"
    except Exception:  # pragma: no cover - purely diagnostic
        return f"❌  ({e.__class__.__name__}: {e})"

def main() -> None:
    print("Environment check:")
    pkgs = [
        "torch",
        "rtamt",
        "moonlight",
        "matplotlib",
        "tqdm",
        "pyyaml",
        "neuromancer",
        "torchphysics",
        "physicsnemo",
        "spatial_spec",
    ]
    for pkg in pkgs:
        print(f"  {pkg:>12}: {_check(pkg)}")
    print("\nPython:", sys.version)

if __name__ == "__main__":
    main()