from __future__ import annotations
import csv
from typing import Iterable, Optional

class CSVLogger:
    def __init__(self, path: str, header: Optional[Iterable[str]] = None) -> None:
        self.path = path
        if header is not None:
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(list(header))
    def append(self, row: Iterable[object]) -> None:
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(list(row))
