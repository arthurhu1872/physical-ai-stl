"""Minimal MoonLight 'hello' that can be skipped in CI if Java/MoonLight are missing."""
from __future__ import annotations

import numpy as np

def temporal_hello():
    # Import lazily to allow tests to skip when missing.
    try:
        from moonlight import ScriptLoader  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("moonlight not installed") from e

    t = np.arange(0.0, 1.0, 0.2)
    x = np.sin(t)
    y = np.cos(t)

    script = """
    signal { real x; real y; }
    domain boolean;
    formula future = globally [0, 0.2] (x > y);
    """
    mls = ScriptLoader.loadFromText(script)
    mon = mls.getMonitor("future")

    # Build the input matrix with (t, x, y) rows as expected by MoonLight
    data = np.vstack([t, x, y]).T.astype(float)
    out = mon.monitor(data)
    # Return something array-like with shape (N, 2) for the test assertions
    # MoonLight returns a list of (time, bool), convert to ndarray
    arr = np.array(out, dtype=float)
    return arr

__all__ = ["temporal_hello"]
