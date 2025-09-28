# tests/test_torchphysics_hello.py
from __future__ import annotations

import importlib.util
import pytest

MOD = "physical_ai_stl.frameworks.torchphysics_hello"

def _import_or_skip():
    spec = importlib.util.find_spec(MOD)
    if spec is None:
        pytest.skip("module not importable")
    mod = importlib.import_module(MOD)
    return mod

def test_import_and_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _import_or_skip()
    assert hasattr(mod, "torchphysics_version")
    assert callable(mod.torchphysics_version)

def test_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _import_or_skip()
    # If TorchPhysics is installed, skip this negative path
    if importlib.util.find_spec("torchphysics") is not None:  # pragma: no cover
        pytest.skip("TorchPhysics installed; skipping negative path")
    with pytest.raises(ImportError):
        _ = mod.torchphysics_version()
