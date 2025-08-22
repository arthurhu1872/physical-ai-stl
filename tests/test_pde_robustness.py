import numpy as np
from physical_ai_stl import pde_example as pe


def test_compute_robustness_simple():
    sig = np.array([0.2, 0.4, 0.6])
    rob = pe.compute_robustness(sig, 0.0, 1.0)
    assert np.isclose(rob, 0.2)


def test_compute_spatiotemporal_robustness():
    mat = np.array([[0.5, 0.6], [0.7, 0.8]])
    rob = pe.compute_spatiotemporal_robustness(mat, 0.0, 1.0)
    assert np.isclose(rob, 0.2)
