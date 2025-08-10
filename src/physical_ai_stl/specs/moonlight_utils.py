"""
Utilities for working with MoonLight STREL specifications.

Functions:
compile_strel(script_path, formula_name): load a MoonLight script and build a monitor for the given formula.
evaluate_strel(monitor, graph_times, graph, times, signals, *params): evaluate a STREL monitor.

Note: This module requires moonlight (Python wrapper) and Java.
"""


def compile_strel(script_path, formula_name):
    """Load a MoonLight specification script (.mls) and return a monitor for the specified formula.

    Parameters
    ----------
    script_path : str
        Path to the MoonLight specification file.
    formula_name : str
        Name of the formula defined in the script.

    Returns
    -------
    monitor : object
        The compiled monitor.

    Raises
    ------
    RuntimeError
        If moonlight is not installed or Java runtime is not available.
    """
    try:
        from moonlight import ScriptLoader
    except Exception as e:
        raise RuntimeError("MoonLight must be installed and Java must be available") from e
    script = ScriptLoader.loadFromFile(script_path)
    return script.getMonitor(formula_name)


def evaluate_strel(monitor, graph_times, graph, times, signals, *params):
    """Evaluate a MoonLight STREL monitor on provided signals and graph.

    Parameters
    ----------
    monitor : object
        Monitor returned by compile_strel.
    graph_times : list
        Location time array (list with one time value) for the static graph.
    graph : list
        Graph representation; list of edge lists as expected by MoonLight.
    times : list
        List of time points for the signal.
    signals : list
        Sequence of signal frames for each variable; e.g., list where each element is a list of node values.
    *params
        Additional parameters (numbers) passed to monitor.monitor.

    Returns
    -------
    list
        List of tuples (time_index, robustness_value).
    """
    try:
        return monitor.monitor(graph_times, graph, times, signals, *params)
    except Exception as e:
        raise RuntimeError("Error evaluating MoonLight STREL monitor") from e
