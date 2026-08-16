"""
Server settings.

The API key belongs to whoever runs the server, not to the visitor, so every
limit here exists to bound what a visitor can spend on their behalf.

Everything below is read from `settings.env` at the moment it is asked for, so
`config.DAILY_REQUEST_BUDGET` is the number in that file right now — raising it
takes effect on the next request, with a translation still running. The two
constants are the exceptions, and both describe where the process lives rather
than how it behaves: they cannot change without a restart anyway.
"""

import os
from pathlib import Path

# Importing this loads .env from the project root, so it has to come before
# anything here reads the environment. It is the same file, and the same load,
# that the CLI uses — which is how the two stay in step.
from epub_translate import defaults, settings_file
from epub_translate.defaults import (
    # Whoever runs the server pays for translation, so the key is read once at
    # startup, from the same place the CLI reads it, and never accepted from a
    # request.
    GEMINI_API_KEY,
    PROJECT_ROOT,
)

DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/epub_translate"))
STATIC_DIR = PROJECT_ROOT / "web" / "dist"

# What a visitor is allowed to spend, and what the server will hold, with the
# value each takes when `settings.env` says nothing.
#
# MAX_TRANSLATIONS_AT_ONCE is the one that is not wholly live: the gate that
# refuses an upload reads it per request, but it also sized the thread pool in
# `jobs.py` at startup, so *lowering* it applies at once while raising it only
# queues the extra runs until a restart widens the pool.
_LIMITS = {
    "DAILY_REQUEST_BUDGET": 200,
    "MINUTES_BETWEEN_TRANSLATIONS": 30,
    "MAX_UPLOAD_MB": 25,
    "MAX_TRANSLATIONS_AT_ONCE": 2,
    "DELETE_UPLOADS_AFTER_MINUTES": 60,
}

# How a translation is paced. Defined in `epub_translate.defaults` because the
# CLI paces its runs the same way; named here so `config.X` still reaches them.
# The UI does not offer them: a visitor spending the owner's quota has no
# business tuning it. See that module for what each one governs.
_PACING = ("GEMINI_MODEL", "REQUESTS_PER_MINUTE", "TOKENS_PER_MINUTE", "TOKENS_PER_REQUEST")


def __getattr__(name: str):
    """Read a setting now, rather than having read it at import."""
    if name in _PACING:
        return getattr(defaults, name)
    if name in _LIMITS:
        return settings_file.integer(name, _LIMITS[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
