import pytest

def test_neuromancer_version() -> None:
    try:
        from physical_ai_stl.frameworks.neuromancer_hello import neuromancer_version
    except Exception:
        pytest.skip("Neuromancer helper missing")
        return
    try:
        v = neuromancer_version()
    except Exception:
        pytest.skip("Neuromancer not installed")
        return
    assert isinstance(v, str)
