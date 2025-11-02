from __future__ import annotations
import argparse
from . import about

def main() -> int:
    p = argparse.ArgumentParser(description="physical_ai_stl — quick environment/about")
    p.add_argument("--brief", action="store_true", help="Print a one-line summary")
    args = p.parse_args()
    info = about()
    if args.brief:
        # first line only
        print(info.splitlines()[0])
    else:
        print(info)
    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
