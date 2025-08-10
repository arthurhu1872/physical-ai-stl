from dataclasses import dataclass

@dataclass
class TemporalSpecConfig:
    """
    Configuration for a temporal STL specification.
    name: human-readable name for the spec
    type: type of specification (e.g., 'upper_bound' or 'response')
    u_max: maximum bound for upper bound specifications
    theta: threshold for response specifications
    tau: time horizon (in discrete steps) for response specifications
    """
    name: str
    type: str
    u_max: float = 1.0
    theta: float = 0.5
    tau: int = 10
