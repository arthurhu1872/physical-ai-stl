from __future__ import annotations


from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import csv
import os
import time


__all__ = ["CSVLogger"]


# ---- Internal: a tiny cross‑platform lock using a sidecar ".lock" file ----


class _FileLock:

    def __init__(self, target: Path, timeout: float = 5.0, poll: float = 0.01) -> None:
        self._lock_path = Path(str(target) + ".lock")
        self._timeout = float(timeout)
        self._poll = float(poll)
        self._fd: Optional[int] = None

    def acquire(self) -> None:
        deadline = time.time() + self._timeout
        while True:
            try:
                # O_EXCL ensures we fail if the file already exists.
                self._fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(self._fd, str(os.getpid()).encode("ascii"))
                return
            except FileExistsError:
                if time.time() >= deadline:
                    raise TimeoutError(f"Timeout acquiring lock: {self._lock_path}")
                time.sleep(self._poll)

    def release(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            finally:
                self._fd = None
        try:
            os.unlink(self._lock_path)
        except FileNotFoundError:
            pass

    def __enter__(self) -> "_FileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class _NullContext:
    def __enter__(self):  # pragma: no cover - trivial
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover - trivial
        return False


class _Opts:
    __slots__ = ("delimiter","encoding","float_precision","strict_lengths","lock","lock_timeout")
    def __init__(self, delimiter=",", encoding="utf-8", float_precision=None, strict_lengths=True, lock=False, lock_timeout=5.0):
        self.delimiter = delimiter
        self.encoding = encoding
        self.float_precision = float_precision
        self.strict_lengths = strict_lengths
        self.lock = lock
        self.lock_timeout = lock_timeout


class CSVLogger:

    def __init__(
        self,
        path: str | os.PathLike[str],
        header: Optional[Iterable[str]] = None,
        *,
        overwrite: Optional[bool] = None,
        create_dirs: bool = True,
        delimiter: str = ",",
        encoding: str = "utf-8",
        float_precision: Optional[int] = None,
        strict_lengths: bool = True,
        lock: bool = False,
        lock_timeout: float = 5.0,
    ) -> None:
        self.path = Path(path)
        if create_dirs:
            self.path.parent.mkdir(parents=True, exist_ok=True)

        # Store opts compactly
        self._opt = _Opts(
            delimiter=delimiter,
            encoding=encoding,
            float_precision=float_precision,
            strict_lengths=strict_lengths,
            lock=lock,
            lock_timeout=lock_timeout,
        )
        self._header: Optional[list[str]] = list(header) if header is not None else None
        self._lock = _FileLock(self.path, timeout=lock_timeout) if lock else None

        # If header is given, mimic prior behavior (write header, clobber file)
        if self._header is not None:
            if overwrite is None:
                overwrite = True
            if overwrite:
                with self._maybe_locked(), self._open("w") as f:
                    csv.writer(f, delimiter=self._opt.delimiter).writerow(self._header)
            else:
                if self.path.exists():
                    existing = self._peek_first_line()
                    target = self._opt.delimiter.join(self._header)
                    if existing and existing != target:
                        raise ValueError(
                            "Existing CSV header does not match requested header.\n"
                            f"  Existing: {existing}\n  New:      {target}\n"
                            "Pass overwrite=True to replace the file or adjust the header."
                        )
                else:
                    with self._maybe_locked(), self._open("w") as f:
                        csv.writer(f, delimiter=self._opt.delimiter).writerow(self._header)

    # ---- Public API -----------------------------------------------------

    @property
    def header(self) -> Optional[tuple[str, ...]]:
        return tuple(self._header) if self._header is not None else None

    def append(self, row: Iterable[Any] | Mapping[str, Any]) -> None:
        if isinstance(row, Mapping):
            values = self._row_from_mapping(row)
        else:
            values = self._row_from_sequence(row)

        with self._maybe_locked(), self._open("a") as f:
            csv.writer(f, delimiter=self._opt.delimiter).writerow(values)

    def append_many(self, rows: Iterable[Iterable[Any] | Mapping[str, Any]]) -> None:
        with self._maybe_locked(), self._open("a") as f:
            w = csv.writer(f, delimiter=self._opt.delimiter)
            for row in rows:
                if isinstance(row, Mapping):
                    values = self._row_from_mapping(row)
                else:
                    values = self._row_from_sequence(row)
                w.writerow(values)

    def extend_header(self, new_columns: Iterable[str]) -> None:
        if self._header is None:
            self._header = list(new_columns)
            with self._maybe_locked(), self._open("w") as f:
                csv.writer(f, delimiter=self._opt.delimiter).writerow(self._header)
            return

        additions = [c for c in new_columns if c not in self._header]
        if not additions:
            return

        # Read everything (small logs typical for experiments)
        rows: list[list[str]] = []
        if self.path.exists() and self.path.stat().st_size > 0:
            with self._open("r") as f:
                reader = list(csv.reader(f, delimiter=self._opt.delimiter))
            rows = reader[1:] if reader else []

        self._header.extend(additions)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with self._maybe_locked(), tmp.open("w", newline="", encoding=self._opt.encoding) as f:
            w = csv.writer(f, delimiter=self._opt.delimiter)
            w.writerow(self._header)
            pad = [""] * len(additions)
            for r in rows:
                w.writerow(list(r) + pad)
        os.replace(tmp, self.path)

    # ---- Internals ------------------------------------------------------

    def _row_from_sequence(self, row: Iterable[Any]) -> list[Any]:
        vals = list(row)
        if self._opt.float_precision is not None:
            vals = [format(v, f".{self._opt.float_precision}g") if isinstance(v, float) else v for v in vals]

        if self._header is not None:
            if self._opt.strict_lengths and len(vals) != len(self._header):
                raise ValueError(f"Row length {len(vals)} != header length {len(self._header)}")
            if not self._opt.strict_lengths and len(vals) < len(self._header):
                # Right‑pad with empties
                vals = vals + [""] * (len(self._header) - len(vals))
            elif not self._opt.strict_lengths and len(vals) > len(self._header):
                vals = vals[: len(self._header)]
        return vals

    def _row_from_mapping(self, m: Mapping[str, Any]) -> list[Any]:
        # Infer header the first time if necessary
        if self._header is None:
            self._header = list(m.keys())
            with self._maybe_locked(), self._open("w") as f:
                csv.writer(f, delimiter=self._opt.delimiter).writerow(self._header)

        # Validate keys
        extra = [k for k in m.keys() if k not in self._header]
        if extra:
            raise KeyError(
                f"Mapping contains keys not in header: {extra}. "
                "Call extend_header([...]) to add new columns."
            )

        vals = [m.get(k, "") for k in self._header]
        if self._opt.float_precision is not None:
            vals = [format(v, f'.{self._opt.float_precision}g') if isinstance(v, float) else v for v in vals]
        return vals

    def _peek_first_line(self) -> str:
        try:
            with self._open("r") as f:
                return f.readline().rstrip("\r\n")
        except FileNotFoundError:
            return ""

    def _open(self, mode: str):
        return self.path.open(mode, newline="", encoding=self._opt.encoding)

    def _maybe_locked(self):
        return self._lock if self._lock is not None else _NullContext()
