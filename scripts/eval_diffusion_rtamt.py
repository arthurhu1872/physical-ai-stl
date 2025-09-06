from __future__ import annotations

from physical_ai_stl.pde_example import compute_robustness_scalar


def main() -> int:
    _ = compute_robustness_scalar([0.1, 0.2, 0.3])  # exercise import
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
