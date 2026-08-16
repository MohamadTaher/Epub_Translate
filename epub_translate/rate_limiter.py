"""
Pacing a run against the API's per-minute limits.

Two rolling one-minute windows, one counting requests and one counting tokens.
A worker asks for a slot and is held until both windows have room for it.

Three things here are load-bearing:

- **The slot is taken inside the lock that finds room for it.** Finding room and
  then recording the use of it as separate steps lets every worker pass the
  check before any of them has recorded anything, so the limit holds only while
  the pool happens to be no larger than the limit itself. It did happen to be —
  `REQUESTS_PER_MINUTE` sizes both — which is not a thing to rely on.

- **Attempts are counted, not successes.** A request answered with 429 was still
  a request. Recording only successes meant a key with nothing left was asked
  ten times in about two seconds, each one refused, with no pause between them.

- **`back_off` exists for when the server disagrees anyway.** Its limits are the
  real ones and ours are a guess at them, so a 429 is the authoritative answer
  and holds every worker, not just the one that got it.
"""

import threading
import time
from collections import deque
from typing import Callable, Deque, Optional, Tuple

from .console import logger

WINDOW_SECONDS = 60

# Added to every computed wait so a slot has certainly aged out by the time the
# window is looked at again, rather than missing it by milliseconds and sleeping
# another full round.
_WAIT_MARGIN_SECONDS = 1.0


class RateLimiter:
    """
    Holds workers back so a run stays inside the API's per-minute allowances.

    Every method is safe to call from any thread; the windows are only ever read
    or written under `_lock`.
    """

    # No defaults for the two limits: `defaults.py` owns those numbers, and a
    # second spelling of them here would be the copy that goes stale.
    def __init__(self, max_requests_per_minute: int, max_tokens_per_minute: int,
                 on_wait: Optional[Callable[[float, str], None]] = None,
                 on_resume: Optional[Callable[[], None]] = None):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute

        self._requests: Deque[float] = deque()
        self._tokens: Deque[Tuple[float, int]] = deque()
        self._blocked_until = 0.0
        self._lock = threading.Lock()

        # A wait here is the main reason a run appears to stall, so a caller that
        # reports progress needs to hear about it rather than infer it from silence.
        self.on_wait = on_wait
        self.on_resume = on_resume

    def acquire(self, tokens: int, stop_check: Callable[[], bool]) -> bool:
        """
        Wait until there is room for a request of this size, and take it.

        Returns True once the slot is held, False if `stop_check` ended the wait
        first — in which case nothing was reserved and no request should be sent.
        A wait here can last a minute, so a run that is being cancelled has to be
        able to break out of one; `stop_check` is required for that reason.

        A caller that takes a slot and then decides not to send has spent it. The
        window recovers by itself within the minute, and over-counting is the
        harmless direction, so there is nothing to hand back.
        """
        waited = False
        try:
            while True:
                if stop_check():
                    return False

                taken, wait_time, reason = self._try_reserve(tokens)
                if taken:
                    return True

                # Reported through the callback rather than logged here as well.
                # The owner writes to the terminal *and* the event stream, so a
                # direct log would only put a second wording of one wait in front
                # of a reader who is already being told about it.
                waited = True
                self._notify(self.on_wait, wait_time, reason)

                if not self._sleep(wait_time, stop_check):
                    return False
        finally:
            if waited:
                self._notify(self.on_resume)

    def back_off(self, seconds: float, reason: str = "the API's own rate limit"):
        """
        Hold every worker for a while, because the server said so.

        Takes the longest of whatever is already in force and this, so two
        workers refused at once don't shorten each other's wait.
        """
        if seconds <= 0:
            return

        with self._lock:
            self._blocked_until = max(self._blocked_until, time.time() + seconds)

        logger.rate_limit(f"Backing off {seconds:.0f}s: {reason}")

    def _try_reserve(self, tokens: int) -> Tuple[bool, float, str]:
        """
        Take a slot if both windows have room, all under one lock.

        Returns (taken, how long to wait, why) — the wait and the reason are
        meaningless when the slot was taken.
        """
        with self._lock:
            now = time.time()
            self._trim(now)

            if now < self._blocked_until:
                return False, self._blocked_until - now, "back-off after a refused request"

            if len(self._requests) >= self.max_requests_per_minute:
                wait = WINDOW_SECONDS - (now - self._requests[0]) + _WAIT_MARGIN_SECONDS
                return False, wait, "per-minute request limit"

            spent = sum(count for _, count in self._tokens)
            if spent + tokens > self.max_tokens_per_minute:
                if self._tokens:
                    wait = WINDOW_SECONDS - (now - self._tokens[0][0]) + _WAIT_MARGIN_SECONDS
                    return False, wait, "per-minute token limit"

                # An empty window that still has no room means this one request
                # is larger than a whole minute's budget, so no amount of waiting
                # will ever admit it. Send it and let the API judge; blocking the
                # run for ever is the worse answer, and this used to raise
                # IndexError reaching for the front of an empty deque.
                logger.warning(
                    f"A single request is estimated at {tokens:,} tokens, more than the "
                    f"{self.max_tokens_per_minute:,}/minute budget. Sending it anyway — raise "
                    f"TOKENS_PER_MINUTE, or lower TOKENS_PER_REQUEST to send smaller patches."
                )

            self._requests.append(now)
            self._tokens.append((now, tokens))
            return True, 0.0, ""

    def _trim(self, now: float):
        """Drop whatever has aged out of the one-minute windows. Call under the lock."""
        cutoff = now - WINDOW_SECONDS

        while self._requests and self._requests[0] < cutoff:
            self._requests.popleft()

        while self._tokens and self._tokens[0][0] < cutoff:
            self._tokens.popleft()

    @staticmethod
    def _sleep(seconds: float, stop_check: Callable[[], bool]) -> bool:
        """Sleep in short naps, watching for a stop signal. False if stopped."""
        remaining = seconds
        while remaining > 0:
            if stop_check():
                return False
            nap = min(1.0, remaining)
            time.sleep(nap)
            remaining -= nap
        return True

    @staticmethod
    def _notify(callback, *args):
        """A reporting failure must never stop a request from going out."""
        if callback is None:
            return
        try:
            callback(*args)
        except Exception:
            pass
