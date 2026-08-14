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
MAX_PATCHES_PER_JOB = _int_env("MAX_PATCHES_PER_JOB", 25)
IP_COOLDOWN_MINUTES = _int_env("IP_COOLDOWN_MINUTES", 30)

# Resource limits.
MAX_UPLOAD_MB = _int_env("MAX_UPLOAD_MB", 25)
MAX_ACTIVE_JOBS = _int_env("MAX_ACTIVE_JOBS", 2)
JOB_TTL_MINUTES = _int_env("JOB_TTL_MINUTES", 60)

# Translation defaults, matching the CLI.
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
DEFAULT_MAX_CONCURRENT = _int_env("MAX_CONCURRENT", 5)
DEFAULT_MAX_REQUESTS_PER_MINUTE = _int_env("MAX_REQUESTS_PER_MINUTE", 4)
DEFAULT_MAX_TOKENS_PER_MINUTE = _int_env("MAX_TOKENS_PER_MINUTE", 250000)
DEFAULT_MAX_TOKENS_PER_PATCH = _int_env("MAX_TOKENS_PER_PATCH", 15000)

STATIC_DIR = PROJECT_ROOT / "web" / "dist"
