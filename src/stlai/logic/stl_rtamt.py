import rtamt


class StlMonitor:
    """Wrapper for RTAMT STL monitor to evaluate specifications on traces."""

    def __init__(self, spec_str: str, discrete: bool = True) -> None:
        """
        Initialize the STL monitor with a specification string.

        Args:
            spec_str: STL specification in RTAMT syntax.
            discrete: Whether to use discrete time semantics (True) or dense time (False).
        """
        self.discrete = discrete
        # Create a discrete or dense time specification
        if discrete:
            self.spec = rtamt.StlDiscreteTimeSpecification()
        else:
            self.spec = rtamt.StlDenseTimeSpecification()
        self.spec.spec = spec_str

    def compile(self, var_types: dict[str, str]) -> None:
        """
        Declare variables and parse the specification.

        Args:
            var_types: A mapping of signal names to their types (e.g. {'x': 'float'}).
        """
        for name, typ in var_types.items():
            # Declare each variable before parsing
            self.spec.declare_var(name, typ)
        # Parse the spec to finalize internal structure
        self.spec.parse()

    def evaluate(self, dataset: dict[str, list[tuple[int, float]]]):
        """
        Evaluate the specification over the provided dataset.

        Args:
            dataset: Mapping from signal names to a list of (time, value) pairs.

        Returns:
            A mapping from signal names to their robustness values over time.
        """
        return self.spec.evaluate(dataset)
