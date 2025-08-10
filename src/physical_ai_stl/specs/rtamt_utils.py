"""
Utilities for working with RTAMT specifications.

Functions:
compile_stl(spec_str): compile an STL string into a spec using rtamt.
evaluate_stl(spec, series): evaluate the spec on the given time series.

Note: This module requires rtamt to be installed.
"""

import importlib


def compile_stl(spec_str):
    """
    Compile an STL specification string using rtamt.

    Parameters
    ----------
    spec_str : str
        The STL formula string.

    Returns
    -------
    spec : rtamt.StlDiscreteTimeSpecification
        The compiled STL specification.

    Raises
    ------
    RuntimeError
        If rtamt is not installed.
    """
    try:
        import rtamt
    except ImportError as e:
        raise RuntimeError("rtamt is required for compile_stl") from e
    spec = rtamt.StlDiscreteTimeSpecification()
    # Assign the formula; the caller must ensure variables are declared before parse
    spec.spec = spec_str
    spec.parse()
    return spec


def evaluate_stl(spec, series):
    """
    Evaluate a compiled STL specification on a dictionary of time series.

    Parameters
    ----------
    spec : rtamt.StlDiscreteTimeSpecification
        Compiled specification.
    series : dict
        Mapping from variable name to a list of values (float or int).

    Returns
    -------
    float
        Robustness value at time 0.

    Raises
    ------
    RuntimeError
        If rtamt is not installed.
    """
    try:
        import rtamt
    except ImportError as e:
        raise RuntimeError("rtamt is required for evaluate_stl") from e
    tvs = []
    keys = sorted(series.keys())
    T = len(next(iter(series.values())))
    for k in keys:
        tvs.append([k, [[i, float(series[k][i])] for i in range(T)]])
    return float(spec.evaluate(*tvs))
