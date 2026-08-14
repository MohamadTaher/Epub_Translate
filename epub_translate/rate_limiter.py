import threading
import time
from collections import deque
from typing import Callable, Optional

from .logging_utils import logger


class RateLimiter:
    """
    Manages API rate limiting using a rolling window approach.
    Tracks both request count and token usage over time.
    """

    def __init__(self, max_requests_per_minute: int = 4, max_tokens_per_minute: int = 250000,
                 on_wait: Optional[Callable[[float, str], None]] = None,
                 on_resume: Optional[Callable[[], None]] = None):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.request_timestamps_minute = deque()  # Rolling window of request timestamps
        self.token_usage_minute = deque()  # Rolling window of (timestamp, tokens) pairs
        self.lock = threading.Lock()

        # A wait here is the main reason a run appears to stall, so a caller that
        # reports progress needs to hear about it rather than infer it from silence.
        self.on_wait = on_wait
        self.on_resume = on_resume

    def _cleanup_windows(self):
        """Remove entries older than 1 minute from the rolling windows."""
        current_time = time.time()
        cutoff_time = current_time - 60  # 60 seconds ago

        while self.request_timestamps_minute and self.request_timestamps_minute[0] < cutoff_time:
            self.request_timestamps_minute.popleft()

        while self.token_usage_minute and self.token_usage_minute[0][0] < cutoff_time:
            self.token_usage_minute.popleft()

    def can_make_request(self, tokens_to_add: int) -> tuple[bool, float, str]:
        """Check if we can make a request and return wait time and reason if not."""
        with self.lock:
            self._cleanup_windows()

            current_requests_minute = len(self.request_timestamps_minute)
            current_tokens_minute = sum(tokens for _, tokens in self.token_usage_minute)

            if (current_requests_minute < self.max_requests_per_minute and
                    current_tokens_minute + tokens_to_add <= self.max_tokens_per_minute):
                return True, 0.0, ""

            wait_time = 0.0
            reason = ""
            if current_requests_minute >= self.max_requests_per_minute:
                wait_time = max(wait_time, 60 - (time.time() - self.request_timestamps_minute[0]) + 1)
                reason = "per-minute request limit"
            if current_tokens_minute + tokens_to_add > self.max_tokens_per_minute:
                wait_time = max(wait_time, 60 - (time.time() - self.token_usage_minute[0][0]) + 1)
                reason = "per-minute token limit"

            return False, wait_time, reason

    def _notify(self, callback, *args):
        """A reporting failure must never stop a request from going out."""
        if callback is None:
            return
        try:
            callback(*args)
        except Exception:
            pass

    def wait_for_availability(self, tokens_to_add: int, stop_check=None):
        """Wait until we can make a request within rate limits, or until stop_check() returns True."""
        waited = False
        try:
            while True:
                if stop_check is not None and stop_check():
                    return
                can_proceed, wait_time, reason = self.can_make_request(tokens_to_add)
                if can_proceed:
                    return

                logger.rate_limit(f"Rolling window limit reached ({reason}), waiting {wait_time:.1f}s")
                waited = True
                self._notify(self.on_wait, wait_time, reason)

                if stop_check is None:
                    time.sleep(wait_time)
                else:
                    # Sleep in short increments so a stop signal is noticed promptly.
                    remaining = wait_time
                    while remaining > 0:
                        if stop_check():
                            return
                        nap = min(1.0, remaining)
                        time.sleep(nap)
                        remaining -= nap
        finally:
            if waited:
                self._notify(self.on_resume)

    def record_request(self, tokens_used: int):
        """Record a successful request."""
        with self.lock:
            current_time = time.time()
            self.request_timestamps_minute.append(current_time)
            self.token_usage_minute.append((current_time, tokens_used))
