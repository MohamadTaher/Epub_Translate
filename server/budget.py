"""
Daily spend ceiling.

Every translation is billed to the server's own API key, so requests are counted
against a daily budget and refused once it's gone. Counts are kept in SQLite
rather than memory so they survive a worker restart.
"""

import sqlite3
import threading
from datetime import datetime, timedelta, timezone

from . import config

_lock = threading.Lock()
_connection: sqlite3.Connection = None


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _db() -> sqlite3.Connection:
    global _connection
    if _connection is None:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        _connection = sqlite3.connect(config.DATA_DIR / "budget.db", check_same_thread=False)
        _connection.executescript("""
            CREATE TABLE IF NOT EXISTS daily_usage (
                date TEXT PRIMARY KEY,
                api_requests INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS ip_runs (
                ip TEXT PRIMARY KEY,
                last_run TEXT NOT NULL
            );
        """)
        _connection.commit()
    return _connection


def requests_used_today() -> int:
    with _lock:
        row = _db().execute("SELECT api_requests FROM daily_usage WHERE date = ?", (_today(),)).fetchone()
    return row[0] if row else 0


def remaining_today() -> int:
    return max(0, config.DAILY_REQUEST_BUDGET - requests_used_today())


def record_request():
    """Count one API call. Called per attempt, since retries cost money too."""
    with _lock:
        db = _db()
        db.execute("""
            INSERT INTO daily_usage (date, api_requests) VALUES (?, 1)
            ON CONFLICT(date) DO UPDATE SET api_requests = api_requests + 1
        """, (_today(),))
        db.commit()


def cooldown_remaining(ip: str) -> timedelta:
    """How long until this IP may start another job."""
    if config.IP_COOLDOWN_MINUTES <= 0:
        return timedelta(0)

    with _lock:
        row = _db().execute("SELECT last_run FROM ip_runs WHERE ip = ?", (ip,)).fetchone()

    if not row:
        return timedelta(0)

    elapsed = datetime.now(timezone.utc) - datetime.fromisoformat(row[0])
    return max(timedelta(0), timedelta(minutes=config.IP_COOLDOWN_MINUTES) - elapsed)


def record_run(ip: str):
    with _lock:
        db = _db()
        db.execute("""
            INSERT INTO ip_runs (ip, last_run) VALUES (?, ?)
            ON CONFLICT(ip) DO UPDATE SET last_run = excluded.last_run
        """, (ip, datetime.now(timezone.utc).isoformat()))
        db.commit()
