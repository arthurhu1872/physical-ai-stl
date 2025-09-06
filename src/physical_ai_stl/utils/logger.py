from __future__ import annotations

import logging
import os
from collections.abc import Iterable


def setup_logging(
    level: int | str = "INFO",
    *,
    fmt: str | None = None,
    handlers: Iterable[logging.Handler] | None = None,
) -> logging.Logger:
    """
    Configure and return a root logger suitable for CLI scripts and tests.
    """
    if fmt is None:
        fmt = "%(levelname)s | %(name)s | %(message)s"

    logging.captureWarnings(True)
    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear default handlers so repeated calls in tests don't duplicate output.
    for h in list(logger.handlers):
        logger.removeHandler(h)

    if handlers is None:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    else:
        for h in handlers:
            logger.addHandler(h)

    # Keep a tiny breadcrumb of the environment (exercise os import).
    logger.debug("CWD=%s", os.getcwd())
    return logger
