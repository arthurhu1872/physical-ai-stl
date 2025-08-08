"""Minimal MoonLight examples via its Python wrapper."""

import numpy as np
from moonlight import ScriptLoader  # type: ignore


def temporal_hello():
    """Compute a simple temporal property using MoonLight."""
    # Build a tiny signal with two real-valued variables x, y
    t = list(np.arange(0.0, 1.0, 0.2))
    x = np.sin(t)
    y = np.cos(t)

    # MoonLight script describing two formulas: 'future' and 'past'
    script = """
    signal { real x; real y; }
    domain boolean;
    formula future = globally [0, 0.2] (x > y);
    formula past   = historically [0, 0.2] (x > y);
    """
    mls = ScriptLoader.loadFromText(script)
    # Optionally use quantitative domain: mls.setMinMaxDomain()

    # Get a monitor for 'future'
    future_monitor = mls.getMonitor("future")

    # pair (x, y) values for each time
    sig = list(zip(x, y))
    res = np.array(future_monitor.monitor(t, sig))
    print("MoonLight temporal hello (time, value):")
    print(res)
    return res


def spatiotemporal_hello():
    """Compute a simple spatio-temporal property using MoonLight."""
    # Simple graph of 5 nodes with unit distances (static over time)
    graph = [[[0.0, 1.0, 1.0], [0.0, 3.0, 1.0], [0.0, 4.0, 1.0],
              [1.0, 0.0, 1.0], [1.0, 4.0, 1.0], [1.0, 2.0, 1.0],
              [2.0, 1.0, 1.0], [2.0, 4.0, 1.0], [2.0, 3.0, 1.0],
              [3.0, 0.0, 1.0], [3.0, 2.0, 1.0], [3.0, 4.0, 1.0],
              [4.0, 0.0, 1.0], [4.0, 1.0, 1.0], [4.0, 2.0, 1.0],
              [4.0, 3.0, 1.0]]]
    # The graph is constant in time (one time point)
    loc_times = [0.0]

    # 1D signal on 5 locations (also time-invariant here)
    signal = [[[1.0]], [[3.0]], [[3.0]], [[3.0]], [[3.0]]]

    # Load a simple STREL formula from a text string (or a file)
    script = """
    signal { real s; }
    domain boolean;
    formula MyFirstFormula = everywhere (s <= 3.0);
    """
    mls = ScriptLoader.loadFromText(script)
    mon = mls.getMonitor("MyFirstFormula")

    # When doing spatio-temporal monitoring, pass graph + times + signal
    res = mon.monitor(loc_times, graph, loc_times, signal)
    print("MoonLight spatiotemporal hello:", res)
    return res


if __name__ == "__main__":
    temporal_hello()
    spatiotemporal_hello()
