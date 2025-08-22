#!/usr/bin/env python3

from __future__ import annotations

import argparse
import dataclasses
import importlib
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from typing import Callable, Dict, Iterable, List, Optional, Tuple

try:
    # Python 3.8+: importlib.metadata in stdlib; fall back for older
    from importlib import metadata as md  # type: ignore
except Exception:  # pragma: no cover
    import importlib_metadata as md  # type: ignore

# ------------------------------- helpers ------------------------------------

@dataclasses.dataclass
class ProbeResult:
    present: bool
    imported: bool
    version: Optional[str]
    message: str  # human string (OK/err details)
    extra: Dict[str, str]  # any extra diagnostics


@dataclasses.dataclass
class Dep:
    display: str                # friendly name to show
    modules: Tuple[str, ...]    # python import names to probe (first is canonical)
    dist: Optional[str] = None  # PyPI distribution name for version lookup / pip hint
    required: bool = False      # for exit code; 'core optional' set True
    post_check: Optional[Callable[[ProbeResult, bool], None]] = None  # augment diagnostics
    note: Optional[str] = None  # extra note to display

    def canonical_module(self) -> str:
        return self.modules[0]


def _find_spec(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def _safe_import(mod: str) -> Tuple[bool, Optional[BaseException]]:
    try:
        importlib.import_module(mod)
        return True, None
    except BaseException as e:
        return False, e


def _version_for(dist: Optional[str], module: Optional[str]) -> Optional[str]:
    if dist:
        try:
            return md.version(dist)  # type: ignore[arg-type]
        except md.PackageNotFoundError:
            pass
        except Exception:
            # e.g., broken metadata — continue
            pass
    if module:
        try:
            mod = sys.modules.get(module) or importlib.import_module(module)
            v = getattr(mod, "__version__", None)
            if isinstance(v, str):
                return v
        except Exception:
            pass
    return None


def _run(cmd: Iterable[str]) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        out = proc.stdout.decode("utf-8", errors="replace")
        err = proc.stderr.decode("utf-8", errors="replace")
        return proc.returncode, out, err
    except FileNotFoundError:
        return 127, "", ""
    except Exception as e:
        return 1, "", f"{e.__class__.__name__}: {e}"


def _parse_java_version(text: str) -> Optional[str]:
    m = re.search(r'version\s+"([\d.]+)"', text)
    return m.group(1) if m else None


def _cuda_extra(result: ProbeResult, do_import: bool) -> None:
    if not do_import:
        result.extra["hint"] = "run with --import to gather CUDA/GPU details"
        return
    try:
        import torch  # type: ignore
    except Exception as e:
        result.extra["torch_error"] = f"{e.__class__.__name__}: {e}"
        return
    try:
        cuda_ok = torch.cuda.is_available()
        result.extra["cuda_available"] = str(cuda_ok)
        result.extra["torch_version"] = getattr(torch, "__version__", "?")
        result.extra["cuda_version"] = getattr(torch.version, "cuda", None) or ""
        cudnn_v = getattr(torch.backends, "cudnn", None)
        if cudnn_v is not None and hasattr(cudnn_v, "version"):
            try:
                result.extra["cudnn_version"] = str(torch.backends.cudnn.version())  # type: ignore
            except Exception:
                pass
        if cuda_ok:
            try:
                n = torch.cuda.device_count()
                names = []
                for i in range(n):
                    try:
                        names.append(torch.cuda.get_device_name(i))
                    except Exception:
                        names.append(f"cuda:{i}")
                result.extra["gpus"] = " | ".join(names)
            except Exception:
                pass
    except Exception as e:
        result.extra["cuda_probe_error"] = f"{e.__class__.__name__}: {e}"


def _moonlight_extra(result: ProbeResult, do_import: bool) -> None:
    java = shutil.which("java")
    if not java:
        result.extra["java"] = "not found in PATH"
        return
    code, out, err = _run([java, "-version"])
    text = (out + "\n" + err).strip()
    ver = _parse_java_version(text) if code == 0 or text else None
    result.extra["java"] = f"{java}"
    if ver:
        result.extra["java_version"] = ver
        try:
            major = int(ver.split(".", 1)[0])
            result.extra["java_ok_for_moonlight"] = str(major >= 21)
        except Exception:
            pass


def _spatial_extra(result: ProbeResult, do_import: bool) -> None:
    # Python binding
    try:
        import ltlf2dfa  # type: ignore
        result.extra["ltlf2dfa"] = getattr(ltlf2dfa, "__version__", "present")
    except Exception:
        result.extra["ltlf2dfa"] = "missing"
    # MONA binary
    mona = shutil.which("mona")
    result.extra["mona"] = mona or "not found in PATH"


def _probe(dep: Dep, do_import: bool) -> ProbeResult:
    present = any(_find_spec(m) for m in dep.modules)
    imported = False
    msg = "OK" if present else "not found"
    exc: Optional[BaseException] = None

    if do_import and present:
        imported = False
        # Try importing the canonical module only, to avoid heavy side effects across aliases
        imported, exc = _safe_import(dep.canonical_module())
        if imported:
            msg = "import ok"
        else:
            msg = f"import failed: {exc.__class__.__name__}: {exc}"

    version = _version_for(dep.dist, dep.canonical_module() if imported else None)
    extra: Dict[str, str] = {}

    result = ProbeResult(present=present, imported=imported, version=version, message=msg, extra=extra)

    # Attach domain-specific diagnostics
    if dep.post_check and present:
        try:
            dep.post_check(result, do_import)
        except Exception as e:
            result.extra["post_check_error"] = f"{e.__class__.__name__}: {e}"

    return result


# ------------------------------- inventory ----------------------------------

CORE: List[Dep] = [
    Dep("PyTorch", modules=("torch",), dist="torch", required=True, post_check=_cuda_extra),
    Dep("RTAMT (STL)", modules=("rtamt",), dist="rtamt", required=True),
    Dep("MoonLight (STREL)", modules=("moonlight",), dist="moonlight", required=True, post_check=_moonlight_extra),
    Dep("Neuromancer", modules=("neuromancer",), dist="neuromancer", required=True),
    Dep("TorchPhysics", modules=("torchphysics",), dist="torchphysics", required=True),
    Dep("PhysicsNeMo", modules=("physicsnemo",), dist="nvidia-physicsnemo", required=True),
    Dep("SpaTiaL (spatial-spec)", modules=("spatial_spec",), dist="spatial-spec", required=True, post_check=_spatial_extra),
]

EXTRA: List[Dep] = [
    Dep("matplotlib", modules=("matplotlib",), dist="matplotlib"),
    Dep("tqdm", modules=("tqdm",), dist="tqdm"),
    Dep("PyYAML", modules=("yaml",), dist="PyYAML"),
    Dep("NumPy", modules=("numpy",), dist="numpy"),
    Dep("SciPy", modules=("scipy",), dist="scipy"),
    # Optional SpaTiaL subpackage from source (may coexist)
    Dep("SpaTiaL (spatial-lib)", modules=("spatial",)),
]

ALL_DEPS: List[Dep] = CORE + EXTRA


def _row(dep: Dep, pr: ProbeResult, ascii_only: bool) -> List[str]:
    ok = pr.present
    check = ("OK" if ascii_only else "✅") if ok else ("MISSING" if ascii_only else "❌")
    ver = pr.version or ""
    msg = pr.message
    return [dep.display, check, ver, msg]


def _format_table(rows: List[List[str]], headers: List[str]) -> str:
    # Simple fixed-width table without extra deps; keep dependencies zero.
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    def fmt_row(r: List[str]) -> str:
        return "  " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(r))
    sep = "  " + "-+-".join("-" * w for w in widths)
    lines = [fmt_row(headers), sep]
    lines.extend(fmt_row(r) for r in rows)
    return "\n".join(lines)


def _print_human(results: Dict[str, Tuple[Dep, ProbeResult]], ascii_only: bool, extended: bool) -> None:
    print("Environment check:\n")
    headers = ["Package", "OK", "Version", "Notes"]
    core_rows = [_row(dep, pr, ascii_only) for dep, pr in (results[n] for n in (d.display for d in CORE))]
    print(_format_table(core_rows, headers))
    if extended:
        extra_rows = [_row(dep, pr, ascii_only) for dep, pr in (results[n] for n in (d.display for d in EXTRA))]
        print("\nExtras:\n")
        print(_format_table(extra_rows, headers))

    # Selected extra diagnostics
    torch_res = results.get("PyTorch")[1]
    if torch_res.present:
        extras = torch_res.extra
        if extras:
            print("\nPyTorch details:")
            for k in ("torch_version", "cuda_available", "cuda_version", "cudnn_version", "gpus"):
                if k in extras and extras[k]:
                    print(f"  {k.replace('_', ' ').title():<16}: {extras[k]}")

    moon_res = results.get("MoonLight (STREL)")[1]
    if moon_res.present:
        extras = moon_res.extra
        if extras:
            print("\nMoonLight / Java:")
            for k in ("java", "java_version", "java_ok_for_moonlight"):
                if k in extras and extras[k]:
                    print(f"  {k.replace('_', ' ').title():<16}: {extras[k]}")

    spat_res = results.get("SpaTiaL (spatial-spec)")[1]
    if spat_res.present:
        extras = spat_res.extra
        if extras:
            print("\nSpaTiaL extras:")
            for k in ("ltlf2dfa", "mona"):
                if k in extras and extras[k]:
                    print(f"  {k:<9}: {extras[k]}")

    # Python/platform
    print("\nPython:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())


def _print_markdown(results: Dict[str, Tuple[Dep, ProbeResult]], extended: bool) -> None:
    def md_row(dep: Dep, pr: ProbeResult) -> str:
        check = "✅" if pr.present else "❌"
        ver = pr.version or ""
        return f"| `{dep.display}` | {check} | `{ver}` | {pr.message} |"
    print("### Environment check\n")
    print("| Package | OK | Version | Notes |")
    print("|---|:--:|:--:|---|")
    for d in CORE:
        dep, pr = results[d.display]
        print(md_row(dep, pr))
    if extended:
        print("\n**Extras**\n")
        print("| Package | OK | Version | Notes |")
        print("|---|:--:|:--:|---|")
        for d in EXTRA:
            dep, pr = results[d.display]
            print(md_row(dep, pr))

    # Append additional diagnostics in fenced blocks
    torch_res = results.get("PyTorch")[1]
    if torch_res.present and torch_res.extra:
        print("\n<details><summary>PyTorch details</summary>\n\n```text")
        for k, v in torch_res.extra.items():
            print(f"{k}: {v}")
        print("```\n</details>")

    moon_res = results.get("MoonLight (STREL)")[1]
    if moon_res.present and moon_res.extra:
        print("\n<details><summary>MoonLight / Java</summary>\n\n```text")
        for k, v in moon_res.extra.items():
            print(f"{k}: {v}")
        print("```\n</details>")

    spat_res = results.get("SpaTiaL (spatial-spec)")[1]
    if spat_res.present and spat_res.extra:
        print("\n<details><summary>SpaTiaL extras</summary>\n\n```text")
        for k, v in spat_res.extra.items():
            print(f"{k}: {v}")
        print("```\n</details>")

    print("\n```text")
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    print("```")


def _print_json(results: Dict[str, Tuple[Dep, ProbeResult]]) -> None:
    payload = {}
    for name, (dep, pr) in results.items():
        payload[name] = {
            "present": pr.present,
            "imported": pr.imported,
            "version": pr.version,
            "message": pr.message,
            "extra": pr.extra,
            "dist": dep.dist,
            "modules": dep.modules,
            "required": dep.required,
        }
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Quick summary of optional dependencies and their availability.")
    p.add_argument("--md", action="store_true", help="print a Markdown table")
    p.add_argument("--json", action="store_true", help="print JSON")
    p.add_argument("--import", dest="do_import", action="store_true", help="actually import modules (slower, more robust)")
    p.add_argument("--extended", action="store_true", help="also check extra convenience dependencies")
    p.add_argument("--plain", action="store_true", help="ASCII only (no emoji)")
    args = p.parse_args(argv)

    deps = CORE + (EXTRA if args.extended else [])
    results: Dict[str, Tuple[Dep, ProbeResult]] = {}

    for dep in deps:
        pr = _probe(dep, do_import=args.do_import)
        results[dep.display] = (dep, pr)

    if args.json:
        _print_json(results)
    elif args.md:
        _print_markdown(results, extended=args.extended)
    else:
        _print_human(results, ascii_only=args.plain, extended=args.extended)

    # Exit code: 0 if all required are present, else 1
    missing = [d.display for d in CORE if not results[d.display][1].present]
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
