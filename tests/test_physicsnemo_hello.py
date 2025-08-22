# tests/test_physicsnemo_hello.py
# High-signal, zero-dependency tests for the PhysicsNeMo helper.
# Goals: correctness, speed, and environment independence.
from __future__ import annotations

import importlib
import importlib.util
import sys
import types
import pytest

HELPER_MOD = "physical_ai_stl.frameworks.physicsnemo_hello"


def _import_helper_or_skip(monkeypatch: pytest.MonkeyPatch):
    # Ensure a clean import of the helper itself
    monkeypatch.delitem(sys.modules, HELPER_MOD, raising=False)
    try:
        mod = importlib.import_module(HELPER_MOD)
    except Exception:
        pytest.skip("PhysicsNeMo helper missing")
        raise  # for type checkers
    return mod


def test_helper_import_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    # Make sure there's no lingering entry
    monkeypatch.delitem(sys.modules, "physicsnemo", raising=False)
    helper = _import_helper_or_skip(monkeypatch)
    assert "physicsnemo" not in sys.modules, "Helper import should be lazy"
    # Sanity: exported API exists
    assert hasattr(helper, "physicsnemo_version")


def test_version_uses_dunder_version_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper_or_skip(monkeypatch)

    # Inject a tiny stand‑in module
    dummy = types.ModuleType("physicsnemo")
    dummy.__version__ = "9.8.7"  # arbitrary sentinel
    monkeypatch.setitem(sys.modules, "physicsnemo", dummy)

    v = helper.physicsnemo_version()
    assert isinstance(v, str)
    assert v == "9.8.7"


def test_version_falls_back_to_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper_or_skip(monkeypatch)

    # Inject a module without __version__
    dummy = types.ModuleType("physicsnemo")
    monkeypatch.setitem(sys.modules, "physicsnemo", dummy)

    v = helper.physicsnemo_version()
    assert v == "unknown"


def test_raises_when_physicsnemo_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper_or_skip(monkeypatch)

    # If PhysicsNeMo can be found, skip this specific negative-path test.
    if importlib.util.find_spec("physicsnemo") is not None:  # pragma: no cover
        pytest.skip("PhysicsNeMo is installed; skipping 'not installed' path")

    # Ensure no cached dummy module slips through
    monkeypatch.delitem(sys.modules, "physicsnemo", raising=False)

    with pytest.raises(ImportError):
        _ = helper.physicsnemo_version()
