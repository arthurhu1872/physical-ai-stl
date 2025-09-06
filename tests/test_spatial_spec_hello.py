import pytest
def test_spatial_spec_version() -> None:
    try:
        from physical_ai_stl.frameworks.spatial_spec_hello import spatial_spec_version
    except Exception:
        pytest.skip("SpaTiaL helper missing")
        return
    try:
        v = spatial_spec_version()
    except Exception:
        pytest.skip("spatial-spec not installed")
        return
    assert isinstance(v, str)