"""
The counters and flags every worker in a run shares.

One object rather than a scattering of attributes, because all of them are
written from several threads at once and the rules about them are easy to get
subtly wrong: attempts and answers are different numbers, and `should_stop`
means two different things depending on who set it.
"""

import threading
from typing import Tuple

# Enough patches failing without one succeeding that the API is likely down
# rather than us unlucky.
GIVE_UP_AFTER_CONSECUTIVE_FAILURES = 6


class RunState:
    """
    How far a run has got, and whether it should carry on.

    Every method here is safe to call from any thread. `should_stop` is a plain
    attribute on purpose: setting it is how `/cancel` and Ctrl+C both work, from
    threads that hold nothing.
    """

    def __init__(self):
        self._lock = threading.Lock()

        self.should_stop = False
        # `should_stop` is also how the run halts itself when everything is
        # failing, so on its own it cannot tell a reader's cancel apart from a
        # collapse. Callers report the outcome from this one, and must ask about
        # it first.
        self.aborted_after_failures = False

        self.total_patches = 0
        self.completed_patches = 0
        self.consecutive_failures = 0

        # Attempts and answers are both worth knowing: the daily budget is
        # charged per attempt, so a run of ten successful patches can still have
        # cost thirty requests. These two carry across runs of the same
        # translator, which is why `begin` leaves them alone.
        self.attempts = 0
        self.answered = 0

    def begin(self, total_patches: int):
        """Start a run of `total_patches`, forgetting any previous one."""
        with self._lock:
            self.total_patches = total_patches
            self.completed_patches = 0
            self.consecutive_failures = 0
            self.aborted_after_failures = False

    def record_attempt(self):
        """A request is about to go out. Counted before it is sent, because the
        budget is charged for asking, not for being answered."""
        with self._lock:
            self.attempts += 1

    def record_success(self) -> Tuple[int, int]:
        """A patch landed. Returns how many are done, out of how many."""
        with self._lock:
            self.answered += 1
            self.completed_patches += 1
            self.consecutive_failures = 0
            return self.completed_patches, self.total_patches

    def record_failure(self) -> bool:
        """
        A patch ran out of attempts.

        Returns True for the one caller that should announce the end of the run.
        The decision is claimed under the lock so that two workers failing at the
        same moment don't both announce it.

        Failures are not literally consecutive with several workers in flight,
        but the question this answers — has anything succeeded lately — is the
        right one either way.
        """
        with self._lock:
            self.consecutive_failures += 1
            tripped = (self.consecutive_failures >= GIVE_UP_AFTER_CONSECUTIVE_FAILURES
                       and not self.should_stop
                       and not self.aborted_after_failures)
            if tripped:
                self.aborted_after_failures = True
                self.should_stop = True

        return tripped
