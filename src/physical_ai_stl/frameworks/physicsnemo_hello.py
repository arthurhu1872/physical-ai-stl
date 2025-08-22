from __future__ import annotations


def physicsnemo_version() -> str:
    try:
        import physicsnemo  # type: ignore
    except Exception as e:  # pragma: no cover
        # Keep the message explicit and actionable but lightweight.
        raise ImportError(
            "PhysicsNeMo is not installed. Install with: pip install nvidia-physicsnemo"
        ) from e

    ver = getattr(physicsnemo, "__version__", None)
    return ver if isinstance(ver, str) and ver else "unknown"


__all__ = ["physicsnemo_version"]
