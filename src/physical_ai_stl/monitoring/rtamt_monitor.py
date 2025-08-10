"""Helpers for building and evaluating STL specifications with RTAMT.

This module provides small convenience functions to construct simple
discrete-time STL specifications using RTAMT and evaluate them on
vectorised signals.  The week-1 example uses these helpers for
post-training validation of a safety property.
"""

from __future__ import annotations

from typing import Dict, List, Union

import rtamt  # type: ignore


def stl_always_upper_bound(var: str = "u", u_max: float = 1.0) -> rtamt.StlDiscreteTimeSpecification:
    """Construct an STL spec enforcing ``var <= u_max`` for all time.

    Parameters
    ----------
    var : str
        Name of the signal variable in the specification.
    u_max : float
        Upper bound on the signal.

    Returns
    -------
    rtamt.StlDiscreteTimeSpecification
        Parsed STL specification.
    """
    spec = rtamt.StlDiscreteTimeSpecification()
    spec.declare_var(var, 'float')
    spec.spec = f"always ({var} <= {float(u_max)})"
    spec.parse()
    return spec


def stl_response(
    var: str = "u",
    boundary: str = "ub",
    theta: float = 0.5,
    tau: int = 10,
) -> rtamt.StlDiscreteTimeSpecification:
    """Construct a response property: if boundary triggers then var responds.

    The formula is ``always( (boundary >= θ) -> eventually[0:τ] (var >= θ) )``.

    Parameters
    ----------
    var : str
        Name of the controlled signal.
    boundary : str
        Name of the trigger signal.
    theta : float
        Threshold for triggering and satisfaction.
    tau : int
        Time horizon within which ``var`` must exceed ``theta`` after
        ``boundary`` exceeds ``theta``.

    Returns
    -------
    rtamt.StlDiscreteTimeSpecification
        Parsed STL specification.
    """
    spec = rtamt.StlDiscreteTimeSpecification()
    spec.declare_var(var, 'float')
    spec.declare_var(boundary, 'float')
    spec.spec = (
        f"always( ({boundary} >= {float(theta)}) -> eventually[0:{int(tau)}] "
        f"({var} >= {float(theta)}) )"
    )
    spec.parse()
    return spec


def evaluate_series(
    spec: rtamt.StlDiscreteTimeSpecification,
    series: Dict[str, List[Union[float, int]]],
) -> float:
    """Evaluate robustness of a discrete-time series against an STL spec.

    Parameters
    ----------
    spec : rtamt.StlDiscreteTimeSpecification
        Compiled STL specification.
    series : Dict[str, List[Union[float, int]]]
        Mapping from variable names to lists of values.  The series are
        assumed to be sampled at integer time steps ``0,1,...``.

    Returns
    -------
    float
        Robustness value at time ``0``.
    """
    # RTAMT expects a list of pairs (time, value) per variable
    keys = sorted(series.keys())
    tvs = []
    length = len(next(iter(series.values())))
    for k in keys:
        tvs.append([k, [[i, float(series[k][i])] for i in range(length)]])
    robustness = spec.evaluate(*tvs)
    return float(robustness)
