from physical_ai_stl.datasets import SyntheticSTLNetDataset


def test_synthetic_stlnet_dataset() -> None:
    ds = SyntheticSTLNetDataset(length=5)
    assert len(ds) == 5
    t, v = ds[0]
    assert isinstance(t, float)
    assert isinstance(v, float)
    assert 0.0 <= t <= 1.0
    assert v == v  # check not NaN
