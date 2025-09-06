from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def ensure_iterable(x: Any) -> list[Any]:
    """Return *x* as a list without allocating for common cases."""
    if isinstance(x, list | tuple):  # UP038 compliant
        return list(x)
    return [x]


def flatten(xs: Iterable[Iterable[Any]]) -> list[Any]:
    out: list[Any] = []
    for seq in xs:
        out.extend(seq)
    return out
