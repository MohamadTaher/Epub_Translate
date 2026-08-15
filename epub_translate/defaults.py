"""
How a translation run is paced, read from the environment once at import.

The CLI and the server pace their runs the same way, so the numbers live here
rather than being spelled out on both sides. The server layers its own spend
and resource limits on top in `server/config.py`; those are its alone, because
they bound what a *visitor* may spend on the owner's key, and the CLI has no
visitors.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Read .env from the project root rather than the working directory, so the key
# and the pacing are found no matter where the CLI or the server is started
# from. Real environment variables already set win, which is how deployment
# secrets override the local file.
load_dotenv(PROJECT_ROOT / ".env")


def int_env(name: str, default: int) -> int:
    """Read an integer setting, falling back on both an unset and an unparseable value."""
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


# The same model `.env.example` ships, so an unset variable behaves like the
# documented setup rather than quietly picking a costlier one.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

# REQUESTS_PER_MINUTE is one dial doing two jobs: it caps the rolling-window
# rate limiter and sizes the worker pool. A pool larger than the per-minute
# allowance would only park threads inside the limiter, so the two are never
# worth setting apart.
#
# Floored at 1 because MINUTES_BETWEEN_TRANSLATIONS uses 0 to mean "off", which
# makes 0 an easy guess for "no limit" here too. It would instead size the
# worker pool at zero, and that only raises from the run thread, long after the
# setting was read.
REQUESTS_PER_MINUTE = max(1, int_env("REQUESTS_PER_MINUTE", 4))
TOKENS_PER_MINUTE = int_env("TOKENS_PER_MINUTE", 250000)
TOKENS_PER_REQUEST = int_env("TOKENS_PER_REQUEST", 15000)
