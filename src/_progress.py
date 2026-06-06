"""Shared tqdm helper.

Behavior:
- Prefer writing to /dev/tty so progress shows in the terminal/tmux pane while
  bypassing any `2>&1 | tee` log capture in run.sh.
- If /dev/tty is unavailable, fall back to stderr — but only when stderr is
  actually a terminal. When stderr is a pipe/file (e.g., logs being captured
  outside tmux), disable tqdm entirely so the log file stays clean.
- An explicit `disable=` kwarg always wins.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache

from tqdm import tqdm as _tqdm


@lru_cache(maxsize=1)
def _stream_and_disable():
    if os.environ.get("DISABLE_TQDM"):
        return sys.stderr, True
    try:
        return open("/dev/tty", "w", buffering=1), False
    except (OSError, FileNotFoundError):
        if sys.stderr.isatty():
            return sys.stderr, False
        return sys.stderr, True


def tqdm(*args, **kwargs):
    """Drop-in tqdm wrapper that prefers /dev/tty and auto-disables when piped."""
    stream, disable = _stream_and_disable()
    kwargs.setdefault("file", stream)
    kwargs.setdefault("disable", disable)
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", 0.3)
    return _tqdm(*args, **kwargs)
