# tests/test_rtamt_hello.py
from __future__ import annotations

import pathlib
import sys
import pytest

# Lightweight fallback so the tests run whether or not the package has been installed.
_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Skip all tests here quickly if RTAMT isn't available (optional dependency).
pytest.importorskip("rtamt", reason="RTAMT not installed; skipping RTAMT tests")


def test_rtamt_hello_returns_expected_scalar() -> None:
    from physical_ai_stl.monitors.rtamt_hello import stl_hello_offline

    rob = stl_hello_offline()
    assert isinstance(rob, (int, float))
    # Numerical correctness for G(u <= 1) on [0.2, 0.4, 1.1].
    assert rob == pytest.approx(-0.1, abs=1e-9)


def test_rtamt_monitor_helpers_match_hello() -> None:
    from physical_ai_stl.monitoring.rtamt_monitor import (
        stl_always_upper_bound,
        evaluate_series,
    )

    spec = stl_always_upper_bound("u", u_max=1.0)
    rob = evaluate_series(spec, "u", [0.2, 0.4, 1.1], dt=1.0)
    assert isinstance(rob, (int, float))
    assert rob == pytest.approx(-0.1, abs=1e-9)
