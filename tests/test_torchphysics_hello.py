# tests/test_torchphysics_hello.py
# High-signal, zero-dependency tests for the TorchPhysics helper.
# Goals: correctness, speed, and environment independence.
from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
import types
from pathlib import Path

import pytest

# Make the in-repo package importable whether or not it has been installed yet.
# (tests/ is a sibling of src/, so we prepend src/ to sys.path)
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

MOD = "physical_ai_stl.frameworks.torchphysics_hello"


def _import_helper_or_skip(monkeypatch: pytest.MonkeyPatch):
    """Import the helper module or skip if the package path is unresolved.

    We keep this tiny wrapper so each test can import the same helper
    without duplicating skip logic or polluting globals.
    """
    spec = importlib.util.find_spec(MOD)
    if spec is None:  # pragma: no cover - environment dependent
        pytest.skip("helper module not importable")
    helper = importlib.import_module(MOD)
    return helper


def test_public_api_and_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper_or_skip(monkeypatch)
    # __all__ exists and includes the main query function
    assert hasattr(helper, "__all__")
    assert "torchphysics_version" in getattr(helper, "__all__")
    assert "torchphysics_available" in getattr(helper, "__all__")
    # Signature is 0-arg callable
    assert hasattr(helper, "torchphysics_version")
    assert callable(helper.torchphysics_version)
    sig = inspect.signature(helper.torchphysics_version)
    assert len(sig.parameters) == 0


def test_lazy_import_has_no_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing the helper should *not* import TorchPhysics eagerly."""
    was_present = "torchphysics" in sys.modules
    # Make sure there's no lingering injected module from other tests
    if not was_present:
        monkeypatch.delitem(sys.modules, "torchphysics", raising=False)

    helper = _import_helper_or_skip(monkeypatch)

    # If TorchPhysics wasn't present before, the helper import must not add it.
    if not was_present:
        assert "torchphysics" not in sys.modules, "helper import should be lazy"


def test_version_uses_dunder_version_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper_or_skip(monkeypatch)

    # Inject a tiny stand‑in module that exposes __version__
    dummy = types.ModuleType("torchphysics")
    dummy.__version__ = "9.8.7"  # arbitrary sentinel
    monkeypatch.setitem(sys.modules, "torchphysics", dummy)

    v = helper.torchphysics_version()
    assert isinstance(v, str)
    assert v == "9.8.7"


def test_version_falls_back_to_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper_or_skip(monkeypatch)

    # Inject a module without __version__ so helper must ask metadata.
    dummy = types.ModuleType("torchphysics")
    monkeypatch.setitem(sys.modules, "torchphysics", dummy)

    # Ensure metadata path raises to force the graceful 'unknown' fallback,
    # regardless of whether a real distribution may be installed.
    class _Meta:
        @staticmethod
        def version(name: str) -> str:
            raise RuntimeError("boom")  # force fallback

    monkeypatch.setattr(helper, "_metadata", _Meta, raising=True)

    v = helper.torchphysics_version()
    assert v == "unknown"


def test_available_truthiness(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper_or_skip(monkeypatch)

    # (A) With a dummy module present, available() must be True
    dummy = types.ModuleType("torchphysics")
    monkeypatch.setitem(sys.modules, "torchphysics", dummy)
    assert helper.torchphysics_available() is True

    # (B) When import fails, available() must be False
    monkeypatch.delitem(sys.modules, "torchphysics", raising=False)

    def _fake_import(name: str, *args, **kwargs):
        if name == "torchphysics":
            raise ModuleNotFoundError("No module named 'torchphysics'")
        return importlib.import_module(name, *args, **kwargs)

    # Patch the helper-local import function only.
    monkeypatch.setattr(helper, "import_module", _fake_import, raising=True)
    assert helper.torchphysics_available() is False


def test_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _import_helper_or_skip(monkeypatch)

    # If TorchPhysics is actually installed, skip this negative path.
    if importlib.util.find_spec("torchphysics") is not None:  # pragma: no cover
        pytest.skip("TorchPhysics installed; skipping negative path")

    # Ensure no cached dummy module slips through
    monkeypatch.delitem(sys.modules, "torchphysics", raising=False)

    with pytest.raises(ImportError):
        _ = helper.torchphysics_version()
