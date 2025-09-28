#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import importlib
import importlib.util
import json
import platform
import re
import shutil as _shutil
import subprocess
import sys
from collections.abc import Callable, Iterable

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
    version: str | None
    message: str             # human string (OK/err details)
    extra: dict[str, str]    # any extra diagnostics


@dataclasses.dataclass
class Dep:
    display: str                      # friendly name to show
    modules: tuple[str, ...]          # python import names to probe (first is canonical)
    dist: str | None = None           # PyPI distribution name for version lookup / pip hint
    required: bool = False            # whether this is part of the core set
    platforms: tuple[str, ...] | None = None  # limit requirement to specific OSes (e.g., ("Linux",))
    post_check: Callable[[ProbeResult, bool], None] | None = None  # augment diagnostics
    note: str | None = None           # extra note to display

    def canonical_module(self) -> str:
        return self.modules[0]


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


def _version_for(dist: str | None, module: str | None) -> str | None:
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
    except Exception as e:  # pragma: no cover - extremely rare on CI
        return 1, "", f"{e.__class__.__name__}: {e}"


def _parse_java_version(text: str) -> str | None:
    m = re.search(r'version\s+"([\d.]+)"', text)
    return m.group(1) if m else None


def _version_tuple(s: str) -> tuple[int, ...]:
    nums = re.findall(r"\d+", s)
    return tuple(int(n) for n in nums) if nums else (0,)


def _compare_versions(found: str, required: str) -> int:
    f, r = _version_tuple(found), _version_tuple(required)
    # Compare lexicographically with padding
    L = max(len(f), len(r))
    f += (0,) * (L - len(f))
    r += (0,) * (L - len(r))
    return (f > r) - (f < r)


def _install_hint(dep: Dep) -> str | None:
    if dep.dist:
        # Special-cases with extra context
        if dep.dist == "torch":
            return ("pip install torch  "
                    "(or use Makefile: `make install-torch-cpu` / `make install-torch-cu121`; "
                    "see https://pytorch.org/get-started/locally/)")
        if dep.dist == "moonlight":
            return "pip install moonlight  (requires Java 21+; verify `java -version`)"
        if dep.dist == "spatial-spec":
            return "pip install spatial-spec  (optional: install MONA + `ltlf2dfa`; Windows not supported)"
        if dep.dist == "nvidia-physicsnemo":
            return "pip install nvidia-physicsnemo  (use `[all]` extras if needed)"
        # Default hint
        return f"pip install {dep.dist}"
    return None


# -------------------------- domain-specific checks --------------------------


def _accel_extra(result: ProbeResult, do_import: bool) -> None:
    # NVIDIA
    smi = _shutil.which("nvidia-smi")
    if smi:
        code, out, err = _run([smi, "--query-gpu=name,driver_version,cuda_version", "--format=csv,noheader"])
        if code == 0:
            lines = [ln.strip() for ln in (out or "").splitlines() if ln.strip()]
            if lines:
                names: list[str] = []
                drivers: set[str] = set()
                cudas: set[str] = set()
                for ln in lines:
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

    nvcc = _shutil.which("nvcc")
    if nvcc:
        code, out, err = _run([nvcc, "--version"])
        text = (out or err or "").strip()
        m = re.search(r"release\s+([\d.]+)", text)
        result.extra["nvcc"] = m.group(1) if m else "present"
    else:
        result.extra["nvcc"] = "not found"

    # AMD ROCm
    rocmsmi = _shutil.which("rocm-smi")
    if rocmsmi:
        code, out, err = _run([rocmsmi, "--showdriverversion"])
        ver = None
        for line in (out or "").splitlines():
            m = re.search(r"Driver version:\s*([\w.:-]+)", line)
            if m:
                ver = m.group(1)
                break
        result.extra["rocm_smi"] = ver or "present"
    else:
        result.extra["rocm_smi"] = "not found"

    hipcc = _shutil.which("hipcc")
    if hipcc:
        code, out, err = _run([hipcc, "--version"])
        m = re.search(r"HIP version:\s*([\d.]+)", (out or err or ""))
        result.extra["hipcc"] = m.group(1) if m else "present"
    else:
        result.extra["hipcc"] = "not found"

    if not do_import:
        result.extra.setdefault("hint", "run with --import to gather CUDA/ROCm device details via torch")
        return

    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        result.extra["torch_error"] = f"{e.__class__.__name__}: {e}"
        return

    try:
        result.extra["torch_version"] = getattr(torch, "__version__", "?")
        # CUDA or ROCm build string (e.g., '12.1' or '6.2' for ROCm)
        cu = getattr(torch.version, "cuda", None)
        hip = getattr(torch.version, "hip", None)
        if cu:
            result.extra["torch_cuda"] = cu
        if hip:
            result.extra["torch_rocm"] = hip

        # Device availability and names
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            try:
                n = torch.cuda.device_count()
                names = [torch.cuda.get_device_name(i) for i in range(n)]
                result.extra["cuda_available"] = "True"
                result.extra["gpus"] = "; ".join(names)
            except Exception:
                result.extra["cuda_available"] = "True"
        else:
            result.extra["cuda_available"] = "False"
    except Exception as e:  # pragma: no cover
        result.extra["torch_cuda_error"] = f"{e.__class__.__name__}: {e}"


def _moonlight_extra(result: ProbeResult, do_import: bool) -> None:  # noqa: ARG001
    java = _shutil.which("java")
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
            major_token = ver.split(".", 1)[0]
            major = int(major_token)
            # Handle legacy '1.8.0_xx' style -> treat as 8
            if major == 1:
                try:
                    legacy_minor = int(ver.split(".")[1])
                    major = legacy_minor
                except Exception:
                    pass
            result.extra["java_ok_for_moonlight"] = str(major >= 21)
            result.extra["java_major"] = str(major)
        except Exception:
            pass


def _spatial_extra(result: ProbeResult, do_import: bool) -> None:  # noqa: ARG001
    # Python binding used by SpaTiaL's automaton planning
    try:
        import ltlf2dfa  # type: ignore
        result.extra["ltlf2dfa"] = getattr(ltlf2dfa, "__version__", "present")
    except Exception:
        result.extra["ltlf2dfa"] = "missing"
    # MONA binary
    mona = _shutil.which("mona")
    result.extra["mona"] = mona or "not found in PATH"
    # Windows caveat per project docs
    if platform.system() == "Windows":
        result.extra["windows_note"] = "ltlf2dfa currently not supported on Windows"


def _probe(dep: Dep, do_import: bool) -> ProbeResult:
    present = any(_find_spec(m) for m in dep.modules)
    imported = False
    msg = "OK" if present else "not found"
    exc: BaseException | None = None

    if not present:
        hint = _install_hint(dep)
        if hint:
            msg = f"not found — {hint}"

    if do_import and present:
        imported = False
        # Try importing the canonical module only to avoid heavy side effects across aliases.
        imported, exc = _safe_import(dep.canonical_module())
        if imported:
            msg = "import ok"
        else:
            msg = f"import failed: {exc.__class__.__name__}: {exc}"

    version = _version_for(dep.dist, dep.canonical_module() if imported else None)
    extra: dict[str, str] = {}

    result = ProbeResult(
        present=present,
        imported=imported,
        version=version,
        message=msg,
        extra=extra,
    )

    # Attach domain-specific diagnostics
    if dep.post_check and present:
        try:
            dep.post_check(result, do_import)
        except Exception as e:  # pragma: no cover
            result.extra.setdefault("post_check_error", f"{e.__class__.__name__}: {e}")

    return result


# ------------------------------- inventory ----------------------------------

# Minimum versions aligned with this repo's requirements files.
MIN_VERS = {
    "rtamt": "0.3.5",
    "moonlight": "0.3.1",
    "neuromancer": "1.5.4",
    "torchphysics": "1.0.4",
    "nvidia-physicsnemo": "1.2.0",  # Linux-only in requirements-extra
    "spatial-spec": "0.1.1",
    # Convenience libraries
    "numpy": "1.26.0",
    "matplotlib": "3.10.0",
    "tqdm": "4.67.1",
    "PyYAML": "6.0.2",
    # SciPy min is Python-dependent; checked later
}

# Python version required by pyproject.toml
PY_MIN = (3, 10)

CORE_BASE: list[Dep] = [
    Dep("PyTorch", modules=("torch",), dist="torch", required=True, post_check=_accel_extra),
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
    Dep(
        "PhysicsNeMo",
        modules=("physicsnemo",),
        dist="nvidia-physicsnemo",
        required=True,
        platforms=("Linux",),  # required only on Linux in this repo
    ),
    Dep(
        "SpaTiaL (spatial-spec)",
        modules=("spatial_spec",),
        dist="spatial-spec",
        required=True,
        post_check=_spatial_extra,
    ),
]

EXTRA: list[Dep] = [
    Dep("matplotlib", modules=("matplotlib",), dist="matplotlib"),
    Dep("tqdm", modules=("tqdm",), dist="tqdm"),
    Dep("PyYAML", modules=("yaml",), dist="PyYAML"),
    Dep("NumPy", modules=("numpy",), dist="numpy"),
    Dep("SciPy", modules=("scipy",), dist="scipy"),
    # Optional SpaTiaL subpackage from source (may coexist)
    Dep("SpaTiaL (spatial-lib)", modules=("spatial",)),
]


def _effective_core() -> list[Dep]:
    osname = platform.system()
    out: list[Dep] = []
    for d in CORE_BASE:
        if d.platforms is None or osname in d.platforms:
            out.append(d)
    return out


def _row(dep: Dep, pr: ProbeResult, ascii_only: bool, required_here: bool, min_ver: str | None) -> list[str]:
    ok = pr.present
    check = ("OK" if ascii_only else "✅") if ok else ("MISSING" if ascii_only else "❌")
    ver = pr.version or ""
    msg = pr.message

    # Version ceiling/floor checks (floor only for now)
    if ok and min_ver and pr.version:
        try:
            if _compare_versions(pr.version, min_ver) < 0:
                check = "⚠️" if not ascii_only else "WARN"
                msg = f"outdated (< {min_ver}); consider upgrade"
        except Exception:
            pass

    # If not required on this OS, annotate
    if dep.platforms and platform.system() not in dep.platforms:
        msg = (msg + " [not required on this OS]").strip()

    return [dep.display, check, ver, msg]


def _format_table(rows: list[list[str]], headers: list[str]) -> str:
    # Simple fixed-width table without extra deps; keep dependencies zero.
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

    core = _effective_core()
    core_rows: list[list[str]] = []
    for d in CORE_BASE:
        dep, pr = results[d.display]
        min_ver = MIN_VERS.get(dep.dist or "", None)
        core_rows.append(_row(dep, pr, ascii_only, d in core, min_ver))
    print(_format_table(core_rows, headers))

    if extended:
        extra_rows: list[list[str]] = []
        for d in EXTRA:
            dep, pr = results[d.display]
            min_ver = MIN_VERS.get(dep.dist or "", None)
            extra_rows.append(_row(dep, pr, ascii_only, True, min_ver))
        print("\nExtras:\n")
        print(_format_table(extra_rows, headers))

    # Selected extra diagnostics
    _, torch_res = results["PyTorch"]
    if torch_res.present and torch_res.extra:
        print("\nPyTorch details:")
        for k in sorted(torch_res.extra.keys()):
            print(f"  {k:<18}: {torch_res.extra[k]}")

    _, moon_res = results["MoonLight (STREL)"]
    if moon_res.present and moon_res.extra:
        print("\nMoonLight extras:")
        for k in ("java", "java_version", "java_ok_for_moonlight"):
            if k in moon_res.extra and moon_res.extra[k]:
                print(f"  {k:<18}: {moon_res.extra[k]}")

    _, spat_res = results["SpaTiaL (spatial-spec)"]
    if spat_res.present and spat_res.extra:
        print("\nSpaTiaL extras:")
        for k in ("ltlf2dfa", "mona", "windows_note"):
            if k in spat_res.extra and spat_res.extra[k]:
                print(f"  {k:<18}: {spat_res.extra[k]}")

    # Python/platform
    print("\nPython:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())

    # Simple "what next" suggestions for missing core pieces
    missing_cmds: list[str] = []
    for d in core:
        dep, pr = results[d.display]
        if not pr.present:
            hint = _install_hint(dep)
            if hint:
                missing_cmds.append(hint)
    # System prereqs for MoonLight / SpaTiaL
    if results["MoonLight (STREL)"][1].present:
        mres = results["MoonLight (STREL)"][1]
        if mres.extra.get("java_ok_for_moonlight") == "False":
            missing_cmds.append("Install Java 21+ (e.g., Temurin/OpenJDK 21) and ensure `java -version` reports 21 or higher.")
    if results["SpaTiaL (spatial-spec)"][1].present:
        sres = results["SpaTiaL (spatial-spec)"][1]
        if sres.extra.get("mona", "") in ("", "not found in PATH"):
            missing_cmds.append("Install MONA (e.g., `sudo apt install mona`) and ensure `mona` is in PATH.")
        if sres.extra.get("ltlf2dfa") == "missing" and platform.system() != "Windows":
            missing_cmds.append("pip install ltlf2dfa")

    if missing_cmds:
        print("\nNext steps:\n  - " + "\n  - ".join(missing_cmds))


def _print_markdown(results: dict[str, tuple[Dep, ProbeResult]], extended: bool) -> None:
    def md_row(dep: Dep, pr: ProbeResult) -> str:
        min_ver = MIN_VERS.get(dep.dist or "", None)
        check = "✅" if pr.present else "❌"
        ver = pr.version or ""
        note = pr.message
        if pr.present and min_ver and pr.version and _compare_versions(pr.version, min_ver) < 0:
            note = f"outdated (< {min_ver}); consider upgrade"
            check = "⚠️"
        if dep.platforms and platform.system() not in dep.platforms:
            note = (note + " [not required on this OS]").strip()
        return f"| `{dep.display}` | {check} | `{ver}` | {note} |"

    print("### Environment check\n")
    print("| Package | OK | Version | Notes |")
    print("|---|:--:|:--:|---|")
    for d in CORE_BASE:
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
    _, torch_res = results["PyTorch"]
    if torch_res.present and torch_res.extra:
        print("\n<details><summary>PyTorch details</summary>\n")
        print("```text")
        for k in sorted(torch_res.extra):
            print(f"{k}: {torch_res.extra[k]}")
        print("```")
        print("</details>")

    _, moon_res = results["MoonLight (STREL)"]
    if moon_res.present and moon_res.extra:
        print("\n<details><summary>MoonLight extras</summary>\n")
        print("```text")
        for k in sorted(moon_res.extra):
            print(f"{k}: {moon_res.extra[k]}")
        print("```")
        print("</details>")

    _, spat_res = results["SpaTiaL (spatial-spec)"]
    if spat_res.present and spat_res.extra:
        print("\n<details><summary>SpaTiaL extras</summary>\n")
        print("```text")
        for k in sorted(spat_res.extra):
            print(f"{k}: {spat_res.extra[k]}")
        print("```")
        print("</details>")

    print("\n```text")
    print("Python:", sys.version.replace("\\n", " "))
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
    # Attach environment basics
    payload["_env"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "requires_python_min": PY_MIN,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Quick summary of dependencies and their availability for physical-ai-stl."
    )
    p.add_argument("--md", action="store_true", help="print a Markdown table")
    p.add_argument("--json", action="store_true", help="print JSON")
    p.add_argument(
        "--import",
        dest="do_import",
        action="store_true",
        help="actually import modules (slower, enables GPU details via torch)",
    )
    p.add_argument("--extended", action="store_true", help="also check extra convenience dependencies")
    p.add_argument("--plain", action="store_true", help="ASCII only (no emoji)")
    args = p.parse_args(argv)

    # First, check Python version against repo's requirement
    py_ok = sys.version_info >= PY_MIN
    if not py_ok:
        print(f"⚠️  Python {PY_MIN[0]}.{PY_MIN[1]}+ is recommended (found {sys.version.split()[0]}).")

    # Probe everything
    results: dict[str, tuple[Dep, ProbeResult]] = {}
    for dep in CORE_BASE + EXTRA:
        results[dep.display] = (dep, _probe(dep, args.do_import))

    if args.json:
        _print_json(results)
    elif args.md:
        _print_markdown(results, extended=args.extended)
    else:
        _print_human(results, ascii_only=args.plain, extended=args.extended)

    # Exit code: 0 if all *effective* core deps are present, else 1
    core = _effective_core()
    missing = [d.display for d in core if not results[d.display][1].present]
    return 0 if (not missing and py_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
