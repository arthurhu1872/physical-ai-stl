from __future__ import annotations

import csv
import os
from typing import Iterable, Optional

class CSVLogger:
    """Tiny CSV logger that appends one row per epoch."""
    def __init__(self, path: str, header: Optional[Iterable[str]] = None) -> None:
        self.path = path
        self._header = list(header) if header is not None else None
        # Ensure the directory exists before writing
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        if self._header is not None:
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(self._header)

    def append(self, row: Iterable[object]) -> None:
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(list(row))

__all__ = ["CSVLogger"]
