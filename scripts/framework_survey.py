#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ------------------------------ spec -----------------------------------------


@dataclass(frozen=True)
class Pkg:
    name: str                       # Human-friendly name for display
    pip_names: Tuple[str, ...]      # One or more possible distribution names on PyPI
    import_name: Optional[str]      # Python import name (may differ from pip name)
    desc: str                       # One-line description
    category: str                   # 'framework' | 'stl'


PKGS: List[Pkg] = [
    Pkg(
        name="Neuromancer",
        pip_names=("neuromancer",),
        import_name="neuromancer",
        desc="PyTorch-based differentiable programming for physics-informed optimization and control",
        category="framework",
    ),
    Pkg(
        name="PhysicsNeMo",
        pip_names=("nvidia-physicsnemo",),
        import_name="physicsnemo",
        desc="NVIDIA’s toolkit for physics-ML models and PDE solvers (formerly Modulus)",
        category="framework",
    ),
    Pkg(
        name="TorchPhysics",
        pip_names=("torchphysics",),
        import_name="torchphysics",
        desc="Bosch mesh-free physics learning library for PINNs/DeepRitz/DeepONets",
        category="framework",
    ),
    Pkg(
        name="RTAMT",
        pip_names=("rtamt",),
        import_name="rtamt",
        desc="Runtime STL monitoring (discrete & dense time)",
        category="stl",
    ),
    Pkg(
        name="MoonLight",
        pip_names=("moonlight",),
        import_name="moonlight",
        desc="STREL/STL monitoring (Java engine with Python bindings)",
        category="stl",
    ),
    Pkg(
        name="SpaTiaL",
        pip_names=("spatial-spec",),
        import_name="spatial",  # PyPI name != import name
        desc="Spatio-temporal specification framework (import: 'spatial')",
        category="stl",
    ),
]


# ---------------------------- helpers ----------------------------------------


def _dist_version_for_names(names: Tuple[str, ...]) -> Optional[str]:
    for n in names:
        try:
            return metadata.version(n)
        except Exception:
            # Try normalized form (PEP 503 normalization)
            try:
                return metadata.version(n.replace("-", "_"))
            except Exception:
                continue
    return None


def _import_version(import_name: str) -> Optional[str]:
    try:
        mod = importlib.import_module(import_name)
    except Exception:
        return None
    for attr in ("__version__", "version", "VERSION"):
        v = getattr(mod, attr, None)
        if isinstance(v, str):
            return v
        try:
            if v is not None:
                return str(v)
        except Exception:
            pass
    return None


def _check_java_version(timeout: float = 3.0) -> Optional[str]:
    try:
        # 'java -version' prints to stderr on many JDKs
        proc = subprocess.run(
            ["java", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=timeout,
        )
        blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
        # extract version "21.0.2" etc.
        import re

        m = re.search(r'version\s+"([^"]+)"', blob)
        return m.group(1) if m else blob.strip() or None
    except Exception:
        return None


def _check_torch() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "installed": False,
        "version": None,
        "cuda_available": None,
        "cuda_version": None,
    }
    try:
        import torch  # type: ignore

        info["installed"] = True
        info["version"] = getattr(torch, "__version__", None)
        try:
            info["cuda_available"] = torch.cuda.is_available()
            info["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
        except Exception:
            pass
    except Exception:
        pass
    return info


def _survey(deep: bool = False) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for pkg in PKGS:
        dist_version = _dist_version_for_names(pkg.pip_names)
        imp_version = _import_version(pkg.import_name) if (deep and pkg.import_name) else None
        installed = (dist_version is not None) or (imp_version is not None)

        notes = ""
        if pkg.name == "PhysicsNeMo":
            notes = "Import name: physicsnemo; pip: nvidia-physicsnemo"
        elif pkg.name == "MoonLight":
            if deep:
                jv = _check_java_version()
                notes = f"Java {jv} detected" if jv else "Requires Java 21+ runtime (not detected)"
            else:
                notes = "Requires Java 21+ runtime"
        elif pkg.name == "SpaTiaL":
            notes = "Automaton planning uses MONA via ltlf2dfa (Linux-only support)"

        rows.append(
            {
                "name": pkg.name,
                "pip": pkg.pip_names[0],
                "import": pkg.import_name or "",
                "installed": bool(installed),
                "version": dist_version or imp_version or "not installed",
                "desc": pkg.desc,
                "category": pkg.category,
                "notes": notes,
            }
        )

    sysinfo: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": _check_torch(),
    }
    if deep:
        sysinfo["java"] = _check_java_version() or "not detected"
    return {"rows": rows, "sys": sysinfo}


def _format_text_table(rows: List[Dict[str, Any]]) -> str:
    headers = ["Package", "Installed", "Version", "Import", "PyPI", "Notes"]
    max_widths = [18, 9, 18, 18, 26, 48]
    data = [
        [
            r["name"],
            "yes" if r["installed"] else "no",
            r["version"],
            r["import"],
            r["pip"],
            r["notes"],
        ]
        for r in rows
    ]
    col_w = [
        min(
            max(len(headers[i]), max((len(str(row[i])) for row in data), default=0)),
            max_widths[i],
        )
        for i in range(len(headers))
    ]

    def trunc(s: str, w: int) -> str:
        return s if len(s) <= w else s[: w - 1] + "…"

    def line(cells: List[str]) -> str:
        return "  ".join(trunc(str(cells[i]), col_w[i]).ljust(col_w[i]) for i in range(len(headers)))

    out = [line(headers), line(["-" * w for w in col_w])]
    out.extend(line(row) for row in data)
    return "\n".join(out)


def _format_md_table(rows: List[Dict[str, Any]]) -> str:
    headers = ["Package", "Installed", "Version", "Import", "PyPI", "Notes"]
    md = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        md.append(
            "| "
            + " | ".join(
                [
                    r["name"],
                    "✅" if r["installed"] else "❌",
                    r["version"],
                    f"`{r['import']}`" if r["import"] else "",
                    f"`{r['pip']}`",
                    r["notes"],
                ]
            )
            + " |"
        )
    return "\n".join(md)


# ------------------------------ CLI ------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Summarize available physical‑AI frameworks and STL tooling versions."
    )
    ap.add_argument("--format", choices=["text", "md", "json"], default="text", help="Output format")
    ap.add_argument(
        "--deep",
        action="store_true",
        help="Import packages and check runtime dependencies (Java/CUDA).",
    )
    ap.add_argument(
        "--only",
        choices=["all", "framework", "stl"],
        default="all",
        help="Filter rows by category.",
    )
    ap.add_argument(
        "--show-install",
        action="store_true",
        help="Print pip install lines for any missing packages.",
    )
    args = ap.parse_args(argv)

    result = _survey(deep=args.deep)
    rows = result["rows"]
    sysinfo = result["sys"]

    if args.only != "all":
        want_framework = args.only == "framework"
        rows = [r for r in rows if (r["category"] == "framework") == want_framework]

    if args.format == "json":
        import json

        print(json.dumps({"rows": rows, "sys": sysinfo}, indent=2))
        return

    # Render text or Markdown
    if args.format == "md":
        if args.only in ("all", "framework"):
            print("### Physical‑AI frameworks\n")
            print(_format_md_table([r for r in rows if r["category"] == "framework"]))
            print()
        if args.only in ("all", "stl"):
            print("### STL tooling\n")
            print(_format_md_table([r for r in rows if r["category"] == "stl"]))
            print()
        torch = sysinfo.get("torch", {})
        print("**Environment**")
        print(f"- Python: {sysinfo['python']}  ")
        print(f"- Platform: {sysinfo['platform']}  ")
        print(f"- Torch: {torch.get('version') or 'not installed'}  ")
        if torch.get("installed"):
            print(
                f"- CUDA available: {torch.get('cuda_available')} (CUDA {torch.get('cuda_version')})  "
            )
        if args.deep and "java" in sysinfo:
            print(f"- Java: {sysinfo['java']}  ")
    else:
        if args.only in ("all", "framework"):
            print("Physical‑AI frameworks:\n")
            print(_format_text_table([r for r in rows if r["category"] == "framework"]))
            print()
        if args.only in ("all", "stl"):
            print("STL tooling:\n")
            print(_format_text_table([r for r in rows if r["category"] == "stl"]))
            print()
        print("Environment:")
        torch = sysinfo.get("torch", {})
        print(f"  Python {sysinfo['python']} on {sysinfo['platform']}")
        if torch.get("installed"):
            print(
                f"  Torch {torch['version']} | CUDA available: {torch.get('cuda_available')} (CUDA {torch.get('cuda_version')})"
            )
        else:
            print("  Torch not installed")
        if args.deep and "java" in sysinfo:
            print(f"  Java: {sysinfo['java']}")

    if args.show_install:
        missing = [r for r in rows if not r["installed"]]
        if missing:
            print("\nInstall commands for missing packages:")
            for r in missing:
                print(f"  pip install {r['pip']}")
        else:
            print("\nAll listed packages are installed.")


if __name__ == "__main__":
    main()
