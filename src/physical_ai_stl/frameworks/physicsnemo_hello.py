from __future__ import annotations

def available() -> bool:
    try:
        import nvidia.physicsnemo as _  # type: ignore[import-not-found]
    except Exception:
        return False
    return True
