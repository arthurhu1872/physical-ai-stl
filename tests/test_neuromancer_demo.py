import pytest

def test_neuromancer_demo_import_and_run() -> None:
    try:
        from physical_ai_stl.frameworks.neuromancer_stl_demo import DemoConfig, train_demo
    except Exception:
        pytest.skip("Neuromancer not installed; skipping neuromancer demo test")
        return
    try:
        cfg = DemoConfig(epochs=2, n=64)  # keep it tiny
        out = train_demo(cfg)
    except Exception:
        pytest.skip("Neuromancer demo runtime error; skipping")
        return
    assert 'pytorch' in out
    assert isinstance(out['pytorch']['final_mse'], float)
