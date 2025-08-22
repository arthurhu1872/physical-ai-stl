import pytest

def test_physicsnemo_version() -> None:
    try:
        from physical_ai_stl.frameworks.physicsnemo_hello import physicsnemo_version
    except Exception:
        pytest.skip("PhysicsNeMo helper missing")
        return
    try:
        v = physicsnemo_version()
    except Exception:
        pytest.skip("PhysicsNeMo not installed")
        return
    assert isinstance(v, str)
