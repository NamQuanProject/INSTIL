"""Logging + progress-bar helpers for Instil.

* :func:`setup_file_logger` configures a logger that writes to both stdout and a
  timestamped file under ``logs/`` (git-ignored) so every run is captured.
* :func:`tqdm_iter` / :func:`tqdm_bar` wrap tqdm with a graceful no-op fallback,
  so the code runs whether or not tqdm is installed.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Optional

_LOGGER_NAME = "instil"


def setup_file_logger(run_name: Optional[str] = None, log_dir: str = "logs",
                      level: int = logging.INFO, name: str = _LOGGER_NAME):
    """Create a logger writing to stdout and ``logs/<run_name>.log``.

    Returns ``(logger, logfile_path)``.  Safe to call multiple times (handlers
    are reset each call).
    """
    os.makedirs(log_dir, exist_ok=True)
    run_name = run_name or time.strftime("run_%Y%m%d_%H%M%S")
    logfile = os.path.join(log_dir, f"{run_name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"logging to {logfile}")
    return logger, logfile


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """Return the Instil logger; falls back to a basic stdout logger if unset."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%H:%M:%S")
    return logger


# --------------------------------------------------------------------------- #
# tqdm with a graceful fallback
# --------------------------------------------------------------------------- #
def _try_tqdm():
    try:
        from tqdm.auto import tqdm
        return tqdm
    except Exception:  # pragma: no cover - tqdm optional
        return None


class _DummyBar:
    """No-op stand-in for a tqdm bar when tqdm is unavailable."""

    def __init__(self, *a, **k):
        pass

    def update(self, n: int = 1):
        pass

    def set_postfix(self, *a, **k):
        pass

    def set_postfix_str(self, *a, **k):
        pass

    def set_description(self, *a, **k):
        pass

    def close(self):
        pass

    def __iter__(self):
        return iter(())


def tqdm_iter(iterable, desc: str = "", total: Optional[int] = None,
              leave: bool = False, disable: bool = False):
    """Wrap an iterable in a tqdm bar (or return it unchanged if tqdm missing)."""
    tqdm = _try_tqdm()
    if tqdm is None or disable:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=leave, dynamic_ncols=True)


def tqdm_bar(total: Optional[int] = None, desc: str = "", leave: bool = False,
             disable: bool = False):
    """Return a manually-updated progress bar (or a no-op bar if tqdm missing)."""
    tqdm = _try_tqdm()
    if tqdm is None or disable:
        return _DummyBar()
    return tqdm(total=total, desc=desc, leave=leave, dynamic_ncols=True)
