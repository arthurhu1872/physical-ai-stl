from __future__ import annotations

import sys


def main() -> int:
    # Minimal environment check for CI.
    print(f"python={sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
