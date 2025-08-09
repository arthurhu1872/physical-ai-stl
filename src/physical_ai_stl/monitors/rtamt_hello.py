"""Minimal offline STL robustness example using RTAMT."""

from rtamt.spec.stl.discrete_time.specification import StlDiscreteTimeSpecification  # type: ignore


def stl_hello_offline() -> float:
    """Compute robustness of a simple STL formula offline."""
    # Define a simple property: for the next 3 steps, x - y must stay > 0
    spec = StlDiscreteTimeSpecification()
    spec.name = "hello_offline"
    spec.declare_var("x", "float")
    spec.declare_var("y", "float")
    spec.spec = "always[0,3] (x - y > 0)"
    spec.parse()

    # Discrete-time signals as (time, value) pairs per variable
    x = [(0, 1.0), (1, 2.0), (2, 2.5), (3, 3.0), (4, 4.0)]
    y = [(0, 0.5), (1, 1.5), (2, 1.9), (3, 1.0), (4, 2.0)]

    # Evaluate robustness (scalar robustness at time 0 for the full trace)
    rob = spec.evaluate({"x": x, "y": y})
    print(f"RTAMT hello (offline) robustness: {rob}")
    return float(rob)


if __name__ == "__main__":
    stl_hello_offline()
