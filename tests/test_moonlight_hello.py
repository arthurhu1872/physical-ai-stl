from physical_ai_stl.monitors.moonlight_hello import temporal_hello


def test_moonlight_temporal_smoke():
    res = temporal_hello()
    # Ensure res is a 2D array with 2 columns and at least one row
    # res should have attribute ndim and shape if it is a numpy array
    assert hasattr(res, 'ndim'), "Result should have ndim attribute"
    assert hasattr(res, 'shape'), "Result should have shape attribute"
    assert res.ndim == 2, f"Expected 2-dimensional array, got {res.ndim}"
    assert res.shape[1] == 2, f"Expected 2 columns, got {res.shape[1]}"
    assert res.shape[0] > 0, "Expected at least one row"
