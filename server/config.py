"""
Server settings, all overridable by environment variable.

The API key belongs to whoever runs the server, not to the visitor, so every
limit here exists to bound what a visitor can spend on their behalf.
"""

import os
from pathlib import Path

# Importing this loads .env from the project root, so it has to come before
# anything here reads the environment. It is the same file, and the same load,
# that the CLI uses — which is how the two stay in step.
from epub_translate.defaults import (
    GEMINI_MODEL,
    PROJECT_ROOT,
    REQUESTS_PER_MINUTE,
    TOKENS_PER_MINUTE,
    TOKENS_PER_REQUEST,
    int_env as _int_env,
)

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

# How a translation is paced — GEMINI_MODEL, REQUESTS_PER_MINUTE,
# TOKENS_PER_MINUTE and TOKENS_PER_REQUEST, re-exported from
# epub_translate.defaults above so `config.X` still reaches them. The UI does
# not offer them: a visitor spending the owner's quota has no business tuning
# it. See that module for what each one governs.

STATIC_DIR = PROJECT_ROOT / "web" / "dist"
