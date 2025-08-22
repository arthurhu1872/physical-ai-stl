from __future__ import annotations

import builtins
import sys
from types import ModuleType

import pytest


def _make_dummy_spatial_spec(*, version: str | None) -> ModuleType:
    mod = ModuleType("spatial_spec")
    if version is not None:
        mod.__version__ = version  # type: ignore[attr-defined]
    return mod


def test_spatial_spec_version_real_or_skip() -> None:
    try:
        from physical_ai_stl.frameworks.spatial_spec_hello import (
            spatial_spec_version,
        )
    except Exception:
        pytest.skip("SpaTiaL helper missing")
        return

    try:
        v = spatial_spec_version()
    except ImportError:
        pytest.skip("spatial-spec not installed")
        return

    assert isinstance(v, str)
    assert v != ""


def test_spatial_spec_version_uses_module_dunder_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        from physical_ai_stl.frameworks.spatial_spec_hello import (
            spatial_spec_version,
        )
    except Exception:
        pytest.skip("SpaTiaL helper missing")
        return

    dummy = _make_dummy_spatial_spec(version="9.9.9-test")
    # Ensure our dummy is used regardless of whether the real package exists.
    monkeypatch.setitem(sys.modules, "spatial_spec", dummy)
    assert spatial_spec_version() == "9.9.9-test"


def test_spatial_spec_version_falls_back_to_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        from physical_ai_stl.frameworks.spatial_spec_hello import (
            spatial_spec_version,
        )
    except Exception:
        pytest.skip("SpaTiaL helper missing")
        return

    dummy = _make_dummy_spatial_spec(version=None)
    monkeypatch.setitem(sys.modules, "spatial_spec", dummy)
    assert spatial_spec_version() == "unknown"


def test_spatial_spec_version_raises_importerror_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        from physical_ai_stl.frameworks.spatial_spec_hello import (
            spatial_spec_version,
        )
    except Exception:
        pytest.skip("SpaTiaL helper missing")
        return

    real_import = builtins.__import__

    def _block_spatial_spec(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "spatial_spec":
            raise ModuleNotFoundError("No module named 'spatial_spec'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_spatial_spec)
    with pytest.raises(ImportError):
        spatial_spec_version()
