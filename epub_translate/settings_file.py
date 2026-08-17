"""
The settings file, re-read whenever it changes.

`.env` holds the API key. A key is a secret, it is set once, and reading it
again would never give a different answer — so it stays there and is read at
startup like any other secret.

Everything else is a number someone tunes *while the server is running*: how
fast to translate, what a visitor may spend, how large an upload may be. Those
live here, in a file this module re-reads as soon as its timestamp moves, so
changing one takes effect without a restart — and a restart would abandon any
translation in flight, which is the whole reason to avoid needing one.

Same syntax as `.env`, parsed by the same library, so there is one format to
learn. It is deliberately **not** loaded into `os.environ`: a value there is
read once and cannot be told apart from a real environment variable, which is
exactly the two properties this file exists to avoid.

Whoever asks for a setting gets the value as of that call, so ask late — read
`config.TOKENS_PER_REQUEST` where it is used, not into a module constant, or
the setting is frozen at import and none of this does anything.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

from dotenv import dotenv_values

SETTINGS_FILE = Path(__file__).resolve().parent.parent / "settings.env"

# The file as last parsed, and what it looked like on disk at the time. Rebuilt
# only when that changes, so the common case is one `stat` call.
_values: Dict[str, str] = {}
_stamp: Optional[Tuple[int, int]] = None


def _current() -> Dict[str, str]:
    global _values, _stamp

    try:
        info = SETTINGS_FILE.stat()
        stamp: Optional[Tuple[int, int]] = (info.st_mtime_ns, info.st_size)
    except OSError:
        # Every setting falls back to its default. The file is committed and the
        # image carries a copy, so this is the unusual path rather than the
        # deployed one.
        stamp = None

    if stamp != _stamp:
        parsed: Dict[str, str] = {}
        if stamp is not None:
            try:
                parsed = {k: v for k, v in dotenv_values(SETTINGS_FILE).items() if v is not None}
            except OSError:
                # Compose turns a bind mount of a missing file into a directory.
                # Reading a setting must not be able to take the server down.
                parsed = {}

        # Values before the stamp: another thread reaching the check between the
        # two assignments then re-parses a file it did not have to, which costs
        # nothing. The other order would hand it the old values as current.
        _values = parsed
        _stamp = stamp

    return _values


def text(name: str, default: str) -> str:
    """
    A setting, as of this call.

    The file wins over the environment, rather than the other way around like
    `.env` does. The file is the one being edited — over a bind mount locally,
    by rebuilding the image when deployed — so it has to be the one that takes
    effect. An environment variable is the fallback for a setting the file says
    nothing about, which means setting one for a name the file *does* mention
    has no effect.
    """
    value = _current().get(name, "").strip()
    if value:
        return value
    return os.environ.get(name, default)


def integer(name: str, default: int) -> int:
    """A whole-number setting, falling back on both an unset and an unreadable value."""
    try:
        return int(text(name, str(default)))
    except ValueError:
        return default
