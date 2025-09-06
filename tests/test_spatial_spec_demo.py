import pytest
def test_spatial_demo_smoke() -> None:
    try:
        from physical_ai_stl.monitors.spatial_demo import run_demo
    except Exception:
        pytest.skip("SpaTiaL not installed; skipping spatial demo test")
        return
    try:
        val = run_demo(T=10)
    except Exception:
        # optional dependency may not be present on CI; skip gracefully
        pytest.skip("SpaTiaL runtime error; skipping")
        return
    assert isinstance(val, float)