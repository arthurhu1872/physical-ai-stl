from __future__ import annotations

from physical_ai_stl.frameworks.neuromancer_hello import available as neuromancer_ok
from physical_ai_stl.frameworks.physicsnemo_hello import available as physicsnemo_ok


def main() -> int:
    # Only prints; keeps import blocks tidy and used.
    print(f"neuromancer={neuromancer_ok()} physicsnemo={physicsnemo_ok()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
