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
from collections.abc import Callable, Iterable

try:
    # Python 3.8+: importlib.metadata in stdlib
    from importlib import metadata as md  # type: ignore
except Exception:  # pragma: no cover
    import importlib_metadata as md  # type: ignore


# ------------------------------- data types ----------------------------------


@dataclasses.dataclass
class ProbeResult:
    present: bool
    imported: bool
    version: str | None
    message: str             # human string (OK/err details)
    extra: dict[str, str]    # any extra diagnostics


@dataclasses.dataclass
class Dep:
    display: str                 # friendly name to show
    modules: tuple[str, ...]     # python import names to probe (first is canonical)
    dist: str | None = None      # PyPI distribution name for version lookup / pip hint
    required: bool = False       # if True, contributes to exit code
    post_check: Callable[[ProbeResult, bool], None] | None = None  # augment diagnostics

    def canonical_module(self) -> str:
        return self.modules[0]


# ------------------------------- utilities -----------------------------------


def _find_spec(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def _safe_import(mod: str) -> tuple[bool, BaseException | None]:
    try:
        importlib.import_module(mod)
        return True, None
    except BaseException as e:
        return False, e


def _version_for(dist: str | None, module: str | None, imported: bool) -> str | None:
    # Prefer PyPI distribution metadata (works even without import)
    if dist:
        try:
            return md.version(dist)  # type: ignore[arg-type]
        except md.PackageNotFoundError:
            pass
        except Exception:
            pass
    # Fallback to __version__ attribute if already imported
    if imported and module:
        try:
            mod_obj = sys.modules.get(module) or importlib.import_module(module)
            v = getattr(mod_obj, "__version__", None)
            if isinstance(v, str):
                return v
        except Exception:
            pass
    return None


def _run(cmd: Iterable[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(list(cmd), capture_output=True, check=False, text=True)
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        return 127, "", ""
    except Exception as e:  # pragma: no cover
        return 1, "", f"{e.__class__.__name__}: {e}"


def _parse_java_version(text: str) -> str | None:
    """
    Parses: openjdk version "21.0.2" 2024-01-16 / java version "17.0.9" ...
    """
    m = re.search(r'version\s+"([\d.]+)"', text)
    return m.group(1) if m else None


def _install_hint(dep: Dep) -> str | None:
    """
    Friendly, minimal instructions the user can paste in *any* Python (no bash).
    """
    if dep.dist:
        if dep.dist == "torch":
            return "CPU: python -m pip install torch   (for GPU, see https://pytorch.org/get-started/)"
        if dep.dist == "moonlight":
            return "python -m pip install moonlight   (requires Java 21+ on PATH: check with `java -version`)"
        if dep.dist == "nvidia-physicsnemo":
            return "python -m pip install nvidia-physicsnemo   (optional: add [all] extras)"
        if dep.dist == "spatial-spec":
            return "python -m pip install spatial-spec   (optional automata: install MONA + `python -m pip install ltlf2dfa`)"
        # default
        return f"python -m pip install {dep.dist}"
    return None


def _env_add(d: dict[str, str], key: str, val: str | None) -> None:
    if val:
        d[key] = val


# ----------------------------- post checks -----------------------------------


def _cuda_extra(result: ProbeResult, do_import: bool) -> None:
    # Quick NVIDIA/CUDA signal without Python imports
    smi = shutil.which("nvidia-smi")
    if smi:
        code, out, _ = _run([smi, "--query-gpu=name,driver_version,cuda_version", "--format=csv,noheader"])
        if code == 0 and out:
            names, drivers, cudas = [], set(), set()
            for ln in (ln.strip() for ln in out.splitlines() if ln.strip()):
                parts = [p.strip() for p in ln.split(",")]
                if parts:
                    names.append(parts[0])
                if len(parts) > 1 and parts[1]:
                    drivers.add(parts[1])
                if len(parts) > 2 and parts[2]:
                    cudas.add(parts[2])
            if names:
                result.extra["gpus"] = "; ".join(names)
            if drivers:
                result.extra["nvidia_driver"] = ", ".join(sorted(drivers))
            if cudas:
                result.extra["nvidia_cuda"] = ", ".join(sorted(cudas))
        else:
            result.extra["nvidia_smi"] = f"error (code {code})"
    else:
        result.extra["nvidia_smi"] = "not found"

    nvcc = shutil.which("nvcc")
    if nvcc:
        code, out, err = _run([nvcc, "--version"])
        text = (out or err or "").strip()
        m = re.search(r"release\s+([\d.]+)", text)
        result.extra["nvcc"] = m.group(1) if m else "present"
    else:
        result.extra["nvcc"] = "not found"

    if not do_import:
        result.extra["hint"] = "add --import to query CUDA/MPS via torch"
        return

    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        result.extra["torch_error"] = f"{e.__class__.__name__}: {e}"
        return

    try:
        result.extra["torch_version"] = getattr(torch, "__version__", "?")
        _env_add(result.extra, "cuda_available", str(torch.cuda.is_available()))
        _env_add(result.extra, "cuda_version", getattr(torch.version, "cuda", None) or "")
        # cuDNN
        try:
            import torch.backends.cudnn as cudnn  # type: ignore
            _env_add(result.extra, "cudnn_version", str(cudnn.version()))
        except Exception:
            pass
        # Apple MPS (Metal) backend
        try:
            mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()  # type: ignore[attr-defined]
            _env_add(result.extra, "mps_available", str(bool(mps_ok)))
        except Exception:
            pass
        # List devices if CUDA is ready
        if torch.cuda.is_available():
            try:
                n = torch.cuda.device_count()
                names = [torch.cuda.get_device_name(i) for i in range(n)]
                result.extra["gpus"] = "; ".join(names)
            except Exception:
                pass
    except Exception as e:  # pragma: no cover
        result.extra["torch_cuda_error"] = f"{e.__class__.__name__}: {e}"


def _moonlight_extra(result: ProbeResult, do_import: bool) -> None:  # noqa: ARG001
    # Java presence / version for MoonLight (requires Java 21+)
    java = shutil.which("java")
    if not java:
        result.extra["java"] = "not found in PATH"
        return
    code, out, err = _run([java, "-version"])
    text = (out + "\n" + err).strip()
    ver = _parse_java_version(text) if (code == 0 or text) else None
    result.extra["java"] = java
    if ver:
        result.extra["java_version"] = ver
        try:
            major = int(ver.split(".", 1)[0])
            result.extra["java_ok_for_moonlight"] = str(major >= 21)
        except Exception:
            pass
    if "JAVA_HOME" in os.environ:
        result.extra["JAVA_HOME"] = os.environ.get("JAVA_HOME", "")


def _spatial_extra(result: ProbeResult, do_import: bool) -> None:  # noqa: ARG001
    # Optional external tools SpaTiaL may call
    mona = shutil.which("mona")
    result.extra["mona"] = mona or "not found in PATH"
    try:
        import ltlf2dfa  # type: ignore
        result.extra["ltlf2dfa"] = getattr(ltlf2dfa, "__version__", "present")
    except Exception:
        result.extra["ltlf2dfa"] = "missing"


# ------------------------------- probing -------------------------------------


def _probe(dep: Dep, do_import: bool) -> ProbeResult:
    present = any(_find_spec(m) for m in dep.modules)
    imported = False
    msg = "OK" if present else "not found"
    if not present:
        hint = _install_hint(dep)
        if hint:
            msg = f"not found — {hint}"
    exc: BaseException | None = None

    if do_import and present:
        imported, exc = _safe_import(dep.canonical_module())
        msg = "import ok" if imported else f"import failed: {exc.__class__.__name__}: {exc}"

    version = _version_for(dep.dist, dep.canonical_module(), imported)
    extra: dict[str, str] = {}

    result = ProbeResult(
        present=present,
        imported=imported,
        version=version,
        message=msg,
        extra=extra,
    )

    if dep.post_check and present:
        try:
            dep.post_check(result, do_import)
        except Exception as e:  # pragma: no cover
            result.extra.setdefault("post_check_error", f"{e.__class__.__name__}: {e}")

    return result


# ------------------------------- inventory ----------------------------------


CORE: list[Dep] = [
    Dep("PyTorch", modules=("torch",), dist="torch", required=True, post_check=_cuda_extra),
    Dep("RTAMT (STL)", modules=("rtamt",), dist="rtamt", required=True),
    Dep(
        "MoonLight (STREL)",
        modules=("moonlight",),
        dist="moonlight",
        required=True,
        post_check=_moonlight_extra,
    ),
    Dep("Neuromancer", modules=("neuromancer",), dist="neuromancer", required=True),
    Dep("TorchPhysics", modules=("torchphysics",), dist="torchphysics", required=True),
    Dep("PhysicsNeMo", modules=("physicsnemo",), dist="nvidia-physicsnemo", required=True),
    Dep(
        "SpaTiaL (spatial-spec)",
        modules=("spatial_spec",),
        dist="spatial-spec",
        required=True,
        post_check=_spatial_extra,
    ),
]

EXTRA: list[Dep] = [
    Dep("NumPy", modules=("numpy",), dist="numpy"),
    Dep("SciPy", modules=("scipy",), dist="scipy"),
    Dep("matplotlib", modules=("matplotlib",), dist="matplotlib"),
    Dep("tqdm", modules=("tqdm",), dist="tqdm"),
    Dep("PyYAML", modules=("yaml",), dist="PyYAML"),
    # Optional SpaTiaL subpackage from source (may coexist)
    Dep("SpaTiaL (spatial-lib)", modules=("spatial",)),
]


# ------------------------------- rendering -----------------------------------


def _row(dep: Dep, pr: ProbeResult, ascii_only: bool) -> list[str]:
    check = ("OK" if ascii_only else "✅") if pr.present else ("MISSING" if ascii_only else "❌")
    ver = pr.version or ""
    msg = pr.message
    return [dep.display, check, ver, msg]


def _format_table(rows: list[list[str]], headers: list[str]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def fmt_row(r: list[str]) -> str:
        return "  " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(r))

    sep = "  " + "-+-".join("-" * w for w in widths)
    lines = [fmt_row(headers), sep]
    lines.extend(fmt_row(r) for r in rows)
    return "\n".join(lines)


def _print_human(
    results: dict[str, tuple[Dep, ProbeResult]],
    ascii_only: bool,
    extended: bool,
) -> None:
    print("Environment check:\n")
    headers = ["Package", "OK", "Version", "Notes"]

    core_rows: list[list[str]] = []
    for d in CORE:
        dep, pr = results[d.display]
        core_rows.append(_row(dep, pr, ascii_only))
    print(_format_table(core_rows, headers))

    if extended:
        extra_rows: list[list[str]] = []
        for d in EXTRA:
            dep, pr = results[d.display]
            extra_rows.append(_row(dep, pr, ascii_only))
        print("\nExtras:\n")
        print(_format_table(extra_rows, headers))

    # Diagnostics
    _, torch_res = results["PyTorch"]
    if torch_res.present and torch_res.extra:
        print("\nPyTorch details:")
        for k in sorted(torch_res.extra.keys()):
            print(f"  {k:<18}: {torch_res.extra[k]}")

    _, moon_res = results["MoonLight (STREL)"]
    if moon_res.present and moon_res.extra:
        print("\nMoonLight extras:")
        for k in ("java", "java_version", "JAVA_HOME", "java_ok_for_moonlight"):
            if k in moon_res.extra and moon_res.extra[k]:
                print(f"  {k:<18}: {moon_res.extra[k]}")

    _, spat_res = results["SpaTiaL (spatial-spec)"]
    if spat_res.present and spat_res.extra:
        print("\nSpaTiaL extras:")
        for k in ("ltlf2dfa", "mona"):
            if k in spat_res.extra and spat_res.extra[k]:
                print(f"  {k:<10}: {spat_res.extra[k]}")

    # Python/platform
    print("\nPython:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())


def _print_markdown(results: dict[str, tuple[Dep, ProbeResult]], extended: bool) -> None:
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

    # Append diagnostics in fenced blocks
    _, torch_res = results["PyTorch"]
    if torch_res.present and torch_res.extra:
        print("\n<details><summary>PyTorch details</summary>\n\n```text")
        for k, v in torch_res.extra.items():
            print(f"{k}: {v}")
        print("```\n</details>")

    _, moon_res = results["MoonLight (STREL)"]
    if moon_res.present and moon_res.extra:
        print("\n<details><summary>MoonLight extras</summary>\n\n```text")
        for k, v in moon_res.extra.items():
            print(f"{k}: {v}")
        print("```\n</details>")

    _, spat_res = results["SpaTiaL (spatial-spec)"]
    if spat_res.present and spat_res.extra:
        print("\n<details><summary>SpaTiaL extras</summary>\n\n```text")
        for k, v in spat_res.extra.items():
            print(f"{k}: {v}")
        print("```\n</details>")

    print("\n```text")
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    print("```")


def _print_json(results: dict[str, tuple[Dep, ProbeResult]]) -> None:
    payload: dict[str, dict[str, object]] = {}
    for name, (dep, pr) in results.items():
        payload[name] = {
            "present": pr.present,
            "imported": pr.imported,
            "version": pr.version,
            "message": pr.message,
            "extra": pr.extra,
        }
    payload["_env"] = {"python": sys.version, "platform": platform.platform()}
    print(json.dumps(payload, indent=2, sort_keys=True))


# ------------------------------- installer -----------------------------------


def _attempt_install(missing: list[Dep]) -> dict[str, str]:
    """
    Try to install missing Python packages using `python -m pip install <dist>`.
    Returns a map of dist -> outcome text.
    """
    results: dict[str, str] = {}
    for dep in missing:
        if not dep.dist:
            results[dep.display] = "skipped (no PyPI dist)"
            continue
        cmd = [sys.executable, "-m", "pip", "install", dep.dist]
        # Special cases where extras are useful
        if dep.dist == "nvidia-physicsnemo":
            cmd[-1] = "nvidia-physicsnemo"
        code, out, err = _run(cmd)
        if code == 0:
            results[dep.dist] = "installed"
        else:
            results[dep.dist] = f"failed (code {code})"
    return results


# ---------------------------------- main -------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Check presence of core frameworks & STL tooling for Physical AI experiments."
    )
    out = p.add_mutually_exclusive_group()
    out.add_argument("--md", action="store_true", help="print a Markdown table")
    out.add_argument("--json", action="store_true", help="print JSON")

    p.add_argument("--extended", action="store_true", help="also check extra convenience dependencies")
    p.add_argument(
        "--import",
        dest="do_import",
        action="store_true",
        help="actually import modules (slower) and probe CUDA/MPS/Java details",
    )
    p.add_argument("--plain", action="store_true", help="ASCII only (no emoji)")
    p.add_argument(
        "--attempt-install",
        action="store_true",
        help="try to `python -m pip install` any missing Python packages (no Bash needed)",
    )

    args = p.parse_args(argv)

    # Probe everything
    results: dict[str, tuple[Dep, ProbeResult]] = {}
    for dep in CORE + EXTRA:
        results[dep.display] = (dep, _probe(dep, args.do_import))

    # Optionally try to install what's missing (Python packages only)
    if args.attempt_install:
        to_install = [d for d in CORE if not results[d.display][1].present]
        if to_install:
            print("\nAttempting to install missing core Python packages ...\n")
            outcomes = _attempt_install(to_install)
            for dist, status in outcomes.items():
                print(f"  {dist:<22} {status}")
            print("\nRe-running checks after installation...\n")
            # Re-probe core only, without imports, to update status quickly
            for dep in CORE:
                results[dep.display] = (dep, _probe(dep, False))
        else:
            print("\nAll core Python packages already present. Nothing to install.\n")

    # Render
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
