"""
Server settings, all overridable by environment variable.

The API key belongs to whoever runs the server, not to the visitor, so every
limit here exists to bound what a visitor can spend on their behalf.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Read .env from the project root rather than the working directory, so the
# server picks up the same file the CLI does no matter where it is started from.
# Real environment variables already set win, which is how deployment secrets
# override the local file.
load_dotenv(PROJECT_ROOT / ".env")


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


# Whoever runs the server pays for translation, so this is read once at startup
# and never accepted from a request.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/epub_translate"))

# Spend limits.
DAILY_REQUEST_BUDGET = _int_env("DAILY_REQUEST_BUDGET", 200)
MAX_REQUESTS_PER_BOOK = _int_env("MAX_REQUESTS_PER_BOOK", 25)
MINUTES_BETWEEN_TRANSLATIONS = _int_env("MINUTES_BETWEEN_TRANSLATIONS", 30)

# Resource limits.
MAX_UPLOAD_MB = _int_env("MAX_UPLOAD_MB", 25)
MAX_TRANSLATIONS_AT_ONCE = _int_env("MAX_TRANSLATIONS_AT_ONCE", 2)
DELETE_UPLOADS_AFTER_MINUTES = _int_env("DELETE_UPLOADS_AFTER_MINUTES", 60)

# How a translation is paced. The UI does not offer these — a visitor spending
# the owner's quota has no business tuning it — so they are set here only.
#
# REQUESTS_PER_MINUTE is one dial doing two jobs: it caps the rolling-window rate
# limiter and sizes the worker pool. A pool larger than the per-minute allowance
# would only park threads inside the limiter, so the two are never worth
# setting apart.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
# Floored at 1: MINUTES_BETWEEN_TRANSLATIONS uses 0 to mean "off", so 0 here is
# an easy guess for "no limit". It would instead size the worker pool at zero,
# which raises from the run thread long after the setting was read.
REQUESTS_PER_MINUTE = max(1, _int_env("REQUESTS_PER_MINUTE", 4))
TOKENS_PER_MINUTE = _int_env("TOKENS_PER_MINUTE", 250000)
TOKENS_PER_REQUEST = _int_env("TOKENS_PER_REQUEST", 15000)

STATIC_DIR = PROJECT_ROOT / "web" / "dist"
