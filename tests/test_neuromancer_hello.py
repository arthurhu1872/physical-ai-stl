# tests/test_neuromancer_hello.py
# High-signal, zero-dependency tests for the Neuromancer helper.
# Goals: correctness, speed, and environment independence.
from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
import types
import builtins
import pytest

HELPER_MOD = "physical_ai_stl.frameworks.neuromancer_hello"


def _import_helper_or_skip(monkeypatch: pytest.MonkeyPatch):
    # Ensure a clean import of the helper itself
    monkeypatch.delitem(sys.modules, HELPER_MOD, raising=False)
    try:
        mod = importlib.import_module(HELPER_MOD)
    except Exception:
        pytest.skip("Neuromancer helper missing")
        raise  # for type checkers
    return mod


def test_helper_import_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure neither the helper nor neuromancer are cached
    monkeypatch.delitem(sys.modules, HELPER_MOD, raising=False)
    monkeypatch.delitem(sys.modules, "neuromancer", raising=False)

    _ = _import_helper_or_skip(monkeypatch)

    # Simply importing the helper should not have pulled in `neuromancer`
    assert "neuromancer" not in sys.modules


def test_version_is_string_if_neuromancer_present(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper_or_skip(monkeypatch)

    if importlib.util.find_spec("neuromancer") is None:
        pytest.skip("Neuromancer not installed in this environment")

    v = helper.neuromancer_version()
    assert isinstance(v, str)
    assert v.strip() != ""
    # Cross-check with the installed package's own version attribute, if present.
    nm = importlib.import_module("neuromancer")
    expected = getattr(nm, "__version__", "unknown")
    assert v == expected
    # Also ensure it *looks* like a version when not 'unknown'.
    assert (v == "unknown") or any(ch.isdigit() for ch in v)


def test_returns_unknown_when_dunder_version_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper_or_skip(monkeypatch)

    # Replace any existing neuromancer module with a tiny dummy that lacks __version__
    monkeypatch.setitem(sys.modules, "neuromancer", types.SimpleNamespace())
    v = helper.neuromancer_version()
    assert v == "unknown"


def test_raises_importerror_when_neuromancer_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper_or_skip(monkeypatch)

    # Ensure no cached module leaks through
    monkeypatch.delitem(sys.modules, "neuromancer", raising=False)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # type: ignore[override]
        if name == "neuromancer":
            raise ModuleNotFoundError("No module named 'neuromancer'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError):
        _ = helper.neuromancer_version()


def test_public_api_and_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper_or_skip(monkeypatch)
    assert hasattr(helper, "__all__")
    assert "neuromancer_version" in getattr(helper, "__all__")
    sig = inspect.signature(helper.neuromancer_version)
    assert len(sig.parameters) == 0
