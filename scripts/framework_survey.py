"""Summarize available physical-AI frameworks and STL tooling versions."""
from __future__ import annotations

import importlib.metadata as metadata

FRAMEWORKS = {
    "neuromancer": "PyTorch-based differentiable programming for physics-informed optimization",
    "physicsnemo": "NVIDIA's toolkit for physics-ML models and PDE solvers",
    "torchphysics": "Bosch mesh-free physics learning library",
}

STL_LIBS = {
    "rtamt": "Runtime STL monitoring",
    "moonlight": "Java-based STREL monitoring (via Python bindings)",
    "spatial_spec": "SpaTiaL: spatial-temporal specification framework",
}

def _version(name: str) -> str:
    try:
        return metadata.version(name)
    except Exception:
        return "not installed"

def main() -> None:
    print("Framework survey:\n")
    for name, desc in FRAMEWORKS.items():
        print(f"{name:>12} : {_version(name)} - {desc}")
    print("\nSTL tooling:\n")
    for name, desc in STL_LIBS.items():
        print(f"{name:>12} : {_version(name)} - {desc}")

if __name__ == "__main__":
    main()
