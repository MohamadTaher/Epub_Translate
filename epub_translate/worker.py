"""
One patch: the chapters going into it, the prompt built around them, and what
becomes of the reply.

This is where a run spends its money, and where the decision to spend it again
is made — a failed attempt is repeated only while repeating it can help, which
`gemini.py` judges.

A group of chapters is a *patch*, in the code and in what a reader is told
alike. "Request" is kept for the API call itself — the per-minute limit and the
daily budget count requests, and a patch that retries costs several of them, so
the two are not interchangeable even though the happy path sends one each.
"""

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup

from .book import Chapter, extract_chapter_title
from .gemini import PatchError, describe_error, read_reply, retry_after, strip_code_fence
from .glossary import Glossary, split_translation
from .prompts import CHAPTER_SEPARATOR, build_translation_prompt, split_chapters
from .rate_limiter import RateLimiter
from .run_state import GIVE_UP_AFTER_CONSECUTIVE_FAILURES, RunState
from .tokens import count_tokens

# Room for the prompt's own wording on top of the chapters and the glossary.
_PROMPT_OVERHEAD_TOKENS = 1000

# The reply is charged too, and is not known until it arrives. Reserving for it
# up front keeps the token window honest between sending and answering.
_REPLY_ALLOWANCE = 1.4

MAX_ATTEMPTS_PER_PATCH = 10


@dataclass
class PatchResult:
    """
    Outcome of translating a single patch (group of chapters).

    A patch that was never finished because the run was stopped carries no
    error: the worker reports its own failures, so a result with `success`
    false and nothing to say is one nobody asked to finish.
    """
    patch_index: int
    success: bool
    attempts: int
    error: Optional[str] = None


class PatchWorker:
    """
    One patch, from prompt to translated chapters.

    A single instance serves the whole run: it is called from every thread in
    the pool at once and keeps nothing per-patch, because the model, the
    glossary and the rate limiter are shared by all of them and the counters
    live in `RunState`. It exists to keep one patch's story in one place, not to
    own anything.
    """

    def __init__(self, model, glossary: Glossary, rate_limiter: RateLimiter, state: RunState,
                 source_lang: str, target_lang: str, emit: Callable[..., None]):
        self.model = model
        self.glossary = glossary
        self.rate_limiter = rate_limiter
        self.state = state
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.emit = emit

    def translate(self, patch: List[Chapter], patch_index: int) -> PatchResult:
        """
        Translate one patch, retrying while retrying can help.

        The request is settled once: a retry asks the same question of the same
        chapters, and carries the glossary the first attempt did rather than a
        newer one.

        The correction is the single thing a retry changes, and it is why a
        retry is worth making at all where the model itself was at fault —
        nothing else about an attempt differs from the one before it, so a reply
        rejected for its shape would be produced again, identically, until the
        attempts ran out. It is replaced only by a later failure that has its
        own correction to offer: a 429 on attempt two says nothing about the
        malformed reply from attempt one, which the model still needs telling
        about on attempt three.
        """
        start_time = time.time()
        glossary_section, combined_html, estimated_tokens = self._build_request(patch)
        correction: Optional[str] = None
        last_error = "no attempt was made"
        attempts_made = 0

        for attempt in range(MAX_ATTEMPTS_PER_PATCH):
            if self.state.should_stop:
                return PatchResult(patch_index=patch_index, success=False, attempts=attempts_made)

            # Takes the slot as well as waiting for it, so this attempt is
            # counted whether or not it comes back with anything.
            if not self.rate_limiter.acquire(estimated_tokens,
                                             stop_check=lambda: self.state.should_stop):
                return PatchResult(patch_index=patch_index, success=False, attempts=attempts_made)

            self.emit('api',
                      f"Sending patch {patch_index} (attempt {attempt + 1}/{MAX_ATTEMPTS_PER_PATCH})",
                      event='patch_start', patch=patch_index, attempt=attempt + 1)
            attempts_made += 1
            self.state.record_attempt()

            try:
                prompt = build_translation_prompt(self.source_lang, self.target_lang,
                                                  glossary_section, combined_html, correction)
                parts, new_terms = self._send(prompt, patch, patch_index)
                self._apply(patch, parts)
                self._learn(new_terms, patch_index)

                done, total = self.state.record_success()
                elapsed = time.time() - start_time
                self.emit('success', f"Patch {patch_index} completed in {elapsed:.1f}s [{done}/{total}]",
                          event='patch_done', patch=patch_index, seconds=round(elapsed, 1),
                          completed=done, total=total)

                return PatchResult(patch_index=patch_index, success=True, attempts=attempts_made)

            except Exception as error:
                last_error = describe_error(error)
                # Kept when this failure has nothing of its own to say, so a
                # transport error between two attempts does not wipe what the
                # model still needs to hear about its last reply.
                correction = getattr(error, 'correction', None) or correction
                self.emit('warning',
                          f"Patch {patch_index}/{self.state.total_patches} attempt "
                          f"{attempt + 1}/{MAX_ATTEMPTS_PER_PATCH} failed: {last_error}",
                          event='attempt_failed', patch=patch_index, attempt=attempt + 1)

                # A refusal is the API's own verdict on our pacing, and it
                # outranks the windows we keep, so it holds every worker rather
                # than only this one. Without it the ten attempts above go out in
                # about two seconds and are all refused for the same reason.
                pause = retry_after(error, attempt)
                if pause:
                    self.rate_limiter.back_off(pause, f"the API refused patch {patch_index}")

                # Only `PatchError` carries a verdict; anything else is assumed
                # worth another go. A safety refusal or a reply too long to fit
                # answers the same way every time, and asking nine more times
                # spends nine more requests to be told so.
                if not getattr(error, 'retriable', True):
                    self.emit('error', f"Patch {patch_index}/{self.state.total_patches} cannot "
                                       f"succeed: {last_error}. Not asking again.")
                    break

        # Reported here rather than by the caller: every other per-patch event is
        # emitted from this method, and the paths above that return without
        # failing have already returned — so this is the one place that knows the
        # patch ran out of attempts rather than being cancelled.
        self.emit('error',
                  f"Patch {patch_index}/{self.state.total_patches} failed after "
                  f"{attempts_made} attempt(s): {last_error}",
                  event='patch_failed', patch=patch_index)

        if self.state.record_failure():
            self.emit('error', f"{GIVE_UP_AFTER_CONSECUTIVE_FAILURES} patches failed without one "
                               f"succeeding. Stopping the run.")

        return PatchResult(patch_index=patch_index, success=False,
                           attempts=attempts_made, error=last_error)

    def _build_request(self, patch: List[Chapter]) -> Tuple[str, str, int]:
        """
        The two halves of one patch's prompt, and what it is expected to cost.

        Handed back unrendered because every attempt renders them again around
        its own correction. Both are settled here all the same — the glossary is
        read once, so the tenth attempt asks with the same terms the first one
        did rather than with whatever other patches have since learned.

        The estimate covers a correction without being recalculated for one:
        `_PROMPT_OVERHEAD_TOKENS` is a thousand tokens of room for the prompt's
        own wording, and the longest correction here runs to a fraction of that.
        """
        combined_html = f"\n{CHAPTER_SEPARATOR}\n".join(ch.source_html for ch in patch)
        combined_text = BeautifulSoup(combined_html, 'html.parser').get_text()

        glossary_section = self.glossary.prompt_section(combined_text)
        prompt_tokens = (sum(ch.source_tokens for ch in patch)
                         + count_tokens(glossary_section)
                         + _PROMPT_OVERHEAD_TOKENS)

        return glossary_section, combined_html, int(prompt_tokens * _REPLY_ALLOWANCE)

    def _send(self, prompt: str, patch: List[Chapter],
              patch_index: int) -> Tuple[List[str], Dict[str, str]]:
        """
        Make one request and take the reply apart, or raise saying why not.

        The chapter count is checked here because nothing downstream can: parts
        are matched to chapters by position, so one missing separator shifts
        every chapter after it — the first is given two chapters' text merged
        together and the last silently keeps its original language, in a patch
        that would otherwise be recorded as a success.

        An empty part is that same failure arriving with the right count, so it
        is turned away in the same breath: `_apply` steps over one rather than
        wiping a chapter, which leaves that chapter in its original language
        while the patch reports success — the one outcome all of this exists to
        prevent.
        """
        reply = strip_code_fence(read_reply(self.model.generate_content(prompt)).strip())
        translated_html, new_terms = split_translation(reply)
        parts = split_chapters(translated_html)

        if len(parts) != len(patch):
            mismatch = f"asked for {len(patch)} chapters, got {len(parts)} back"
            self.emit('warning',
                      f"Patch {patch_index}: {mismatch}. Discarding the response — the chapters "
                      f"can no longer be told apart — and asking again.",
                      event='patch_misaligned', patch=patch_index,
                      expected=len(patch), received=len(parts))
            raise PatchError(mismatch, correction=(
                f"Your last reply divided into {len(parts)} chapter(s), but this request holds "
                f"{len(patch)}. Reproduce all {len(patch) - 1} {CHAPTER_SEPARATOR} markers from "
                f"the input, each on a line of its own, spelled exactly like that — the same "
                f"Latin letters, not translated, not reworded, none added and none dropped. "
                f"Count them in your reply before you send it."))

        empty = [str(position) for position, part in enumerate(parts, 1) if not part.strip()]
        if empty:
            which = f"chapter {empty[0]}" if len(empty) == 1 else f"chapters {', '.join(empty)}"
            blank = f"{which} of {len(patch)} came back empty"
            self.emit('warning',
                      f"Patch {patch_index}: {blank}. Discarding the response — the count is "
                      f"right but that chapter would stay in its original language — and "
                      f"asking again.",
                      event='patch_incomplete', patch=patch_index,
                      expected=len(patch), received=len(parts) - len(empty))
            raise PatchError(blank, correction=(
                f"Your last reply had the right number of chapters, but {which} of {len(patch)} "
                f"was empty. Every chapter between the markers has to carry its own full "
                f"translated HTML — no blank sections, no placeholders, nothing left out, however "
                f"short the chapter looks."))

        return parts, new_terms

    @staticmethod
    def _apply(patch: List[Chapter], parts: List[str]):
        """Fold a reply into the chapters it answers, in place."""
        for chapter, part in zip(patch, parts):
            if part.strip():
                translated = BeautifulSoup(part.strip(), 'html.parser')
                chapter.soup = translated
                chapter.translated_title = extract_chapter_title(translated)

    def _learn(self, new_terms: Dict[str, str], patch_index: int):
        """
        Fold the terms this response reported into the glossary, so the patches
        still to be sent are told what this one decided to call things.

        Terms already settled keep their first translation; the count of those
        the model rendered differently is worth showing, because a run full of
        them means the glossary in the prompt is being ignored.
        """
        if not new_terms:
            return

        learned = self.glossary.merge(new_terms)
        if not learned.added and not learned.conflicts:
            return

        settled = f", {learned.conflicts} kept as first translated" if learned.conflicts else ""
        self.emit('system',
                  f"Patch {patch_index}: {len(learned.added)} new glossary term(s){settled} "
                  f"({len(self.glossary)} in total)",
                  event='glossary_learned', patch=patch_index,
                  added=len(learned.added), total=len(self.glossary))
