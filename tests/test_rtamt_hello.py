from physical_ai_stl.monitors.rtamt_hello import stl_hello_offline


def test_rtamt_smoke() -> None:
    """Smoke test for RTAMT hello example."""
    rob = stl_hello_offline()
    assert isinstance(rob, (int, float))
