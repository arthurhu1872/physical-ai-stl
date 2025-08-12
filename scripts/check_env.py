from __future__ import annotations
import importlib, sys

def _check(name: str) -> str:
    try:
        importlib.import_module(name); return "✅"
    except Exception as e:
        return f"❌  ({e.__class__.__name__}: {e})"

def main() -> None:
    print("Environment check:")
    for pkg in ["torch", "rtamt", "moonlight", "matplotlib", "tqdm", "pyyaml"]:
        print(f"  {pkg:>10}: {_check(pkg)}")
    print("\nPython:", sys.version)

if __name__ == "__main__":
    main()
