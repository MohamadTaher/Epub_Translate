"""
Running a translation: planning it, pacing it, and saving as it goes.

`prepare` works out what a run would cost without spending anything, and `run`
carries out a plan. They are separate because the server shows a reader the plan
and waits for them to agree to it; the CLI joins them back together in
`translate_book`.

One worker per request, `REQUESTS_PER_MINUTE` of them at once, each translating
one batch of chapters and mutating those chapters in place. The book is re-saved
after every batch that lands, so a run that dies halfway leaves a readable book
behind.

The parts that are their own subject live next door: `plan.py` works out what a
run would involve, `worker.py` carries out one batch of it, `run_state.py` holds
the counters they share, and `gemini.py` judges whether a failed attempt is
worth repeating.
"""

import signal
import threading
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Set

import google.generativeai as genai

from . import defaults, language
from .book import Chapter, EpubWriter, SourceBook
from .glossary import Glossary
from .console import confirm, logger
from .plan import TranslationPlan, build_plan
from .rate_limiter import RateLimiter
from .run_state import RunState
from .worker import PatchResult, PatchWorker

# `TranslationPlan` is re-exported because a caller that runs a plan gets both
# from here; `plan.py` remains where it is defined.
__all__ = ["EPUBTranslator", "TranslationPlan"]


class EPUBTranslator:
    """
    Translates an EPUB with Gemini: concurrent requests, rate limiting,
    auto-save after every batch, and a glossary that grows as it goes.

    One instance runs one book. `should_stop` may be set from any thread — that
    is what `/cancel` and Ctrl+C both do — and is honoured between attempts and
    inside rate-limit waits, so in-flight requests finish and are saved.
    """

    # The pacing defaults come from .env by way of `defaults`, the same values the
    # CLI and the server pass in explicitly. Spelling them again here would mean a
    # caller that omits one gets something nobody configured — which is how this
    # went on using its own hardcoded model long after .env had chosen another,
    # and a model the key has no quota for answers 429 to everything.
    def __init__(self, api_key: str, source_language: str = "auto", target_language: str = "English",
                 glossary_file_path: str = None,
                 requests_per_minute: int = defaults.REQUESTS_PER_MINUTE,
                 tokens_per_minute: int = defaults.TOKENS_PER_MINUTE,
                 model_name: str = defaults.GEMINI_MODEL,
                 on_event: Optional[Callable[[Dict], None]] = None):
        if not api_key:
            raise ValueError("API key is required")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.on_event = on_event
        # "auto" until `prepare` detects what the book is actually written in.
        self.source_lang = source_language
        self.target_lang = target_language

        # One number sizes the worker pool as well as the rate limiter: the
        # limiter is the real governor, and workers beyond what it lets through
        # in a minute would spend their lives waiting inside it.
        self.requests_per_minute = requests_per_minute
        self.rate_limiter = RateLimiter(
            max_requests_per_minute=requests_per_minute,
            max_tokens_per_minute=tokens_per_minute,
            on_wait=self._on_rate_limit_wait,
            on_resume=self._on_rate_limit_resume,
        )

        self.state = RunState()
        self.glossary = Glossary(glossary_file_path)

        # Set once translation starts; shared by the workers for auto-saving.
        # One lock for the file, held while a whole book is written, and kept
        # apart from the counters so a save never blocks a worker counting.
        self.writer = None
        self.output_path = None
        self._save_lock = threading.Lock()

        self._emit('system', f"Rate limits: {requests_per_minute} req/min (and {requests_per_minute} "
                             f"at a time), {tokens_per_minute:,} tokens/min")
        if glossary_file_path:
            self._emit('system', f"Glossary loaded: {len(self.glossary)} terms")

    # `/cancel` and Ctrl+C both set this, and `server/jobs.py` reads the abort
    # flag beside it to tell a reader's cancel from a run that gave up.
    @property
    def should_stop(self) -> bool:
        return self.state.should_stop

    @should_stop.setter
    def should_stop(self, value: bool):
        self.state.should_stop = value

    @property
    def aborted_after_failures(self) -> bool:
        return self.state.aborted_after_failures

    # ------------------------------------------------------------------ reporting

    def _emit(self, level: str, message: str, **data):
        """
        Log a message and, if a listener is attached, hand it the same message
        as structured data. Called from worker threads, so listeners must be
        thread-safe.
        """
        logger.log(level, message)

        if self.on_event:
            try:
                self.on_event({'level': level, 'message': message, **data})
            except Exception:
                # A listener that throws is the listener's problem; a translation
                # in progress must not die because nobody is reading the feed.
                pass

    def _on_rate_limit_wait(self, seconds: float, reason: str):
        """
        A rate-limit wait is the longest stretch of a run where nothing visibly
        happens, so it is reported rather than left to look like a stall.
        """
        self._emit('rate_limit', f"Waiting {seconds:.0f}s for the {reason} to clear",
                   event='rate_limit_wait', seconds=round(seconds, 1), reason=reason)

    def _on_rate_limit_resume(self):
        self._emit('rate_limit', "Rate limit cleared, continuing", event='rate_limit_done')

    def _request_stop(self, signum, frame):
        if self.should_stop:
            # Second Ctrl+C: let the default handler raise KeyboardInterrupt.
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            return
        self._emit('interrupt', "Stop signal received. Finishing in-flight patches, then saving and exiting...",
                   event='stopping')
        self.should_stop = True

    # ------------------------------------------------------------------- planning

    def prepare(self, source: SourceBook,
                max_tokens_per_patch: int = defaults.TOKENS_PER_REQUEST,
                only_chapter_ids: Optional[Set[str]] = None) -> TranslationPlan:
        """
        Work out what translating this book would involve, without calling the
        API. Safe to call more than once on the same book.
        """
        plan = build_plan(source, self.source_lang, max_tokens_per_patch, only_chapter_ids)

        # Kept so a caller asking the translator what language it settled on
        # gets an answer; `run` reads the plan's copy rather than this one.
        self.source_lang = plan.source_language

        if not plan.chapters:
            self._emit('error', "No chapters found in the EPUB.")

        self._report_plan(plan)
        return plan

    def _report_plan(self, plan: TranslationPlan):
        """
        What reading the book turned up.

        Said here rather than from inside the reader: parsing an archive should
        not also have to narrate itself, and this is the layer that already
        knows a run is being planned.
        """
        if not plan.chapters:
            return

        already_translated = sum(1 for chapter in plan.chapters if chapter.already_translated)
        logger.system(f"Extracted {len(plan.chapters)} total chapters")

        if not language.is_detectable(plan.source_language):
            logger.system(f"Source language '{plan.source_language}' uses the Latin alphabet, so "
                          f"already-translated chapters can't be detected. All chapters will be "
                          f"translated.")
        elif already_translated:
            logger.system(f"Found {already_translated} already translated chapters "
                          f"(will be preserved)")

    def translate_book(self, epub_path: str, output_path: str,
                       max_tokens_per_patch: int = defaults.TOKENS_PER_REQUEST):
        """
        Plan a book, ask for confirmation, and translate it. The CLI's entry point.

        The book is opened once here and handed on, so planning, translating and
        saving all work from one reading of the archive.
        """
        source = SourceBook(epub_path)
        plan = self.prepare(source, max_tokens_per_patch)

        if not plan.chapters:
            return

        if not plan.patches:
            logger.system("All chapters are already translated. Creating final EPUB...")
            EpubWriter(source, self.target_lang).save(plan.chapters, output_path)
            logger.success(f"Successfully created translated EPUB: {output_path}")
            return

        logger.system(f"Total tokens to translate: {plan.total_tokens:,}")
        logger.system(f"Max tokens per patch: {max_tokens_per_patch:,}")
        logger.system(f"Patches created: {len(plan.patches)}")

        if not confirm("Proceed with translation?"):
            logger.system("Translation cancelled.")
            return

        self.run(plan, source, output_path)

    # -------------------------------------------------------------------- running

    def run(self, plan: TranslationPlan, source: SourceBook, output_path: str):
        """
        Execute a prepared plan: translate every patch concurrently, saving the
        book after each one. Safe to call off the main thread.

        `source` must be the book the plan was prepared from, and is taken over
        by the writer from here on.
        """
        self.writer = EpubWriter(source, self.target_lang)
        self.output_path = output_path
        self.state.begin(len(plan.patches))

        self._emit('system', "🚀 Starting concurrent translation with auto-save enabled",
                   event='started', total=len(plan.patches))
        self._emit('system', "Progress will be saved after each successful patch")

        # The resolved language is read off the plan, which is where detection
        # put it. Reading it from `self` instead made this depend on `prepare`
        # having been called on this same instance beforehand — a rule nothing
        # enforced, whose price for breaking it is asking the model to translate
        # "from auto".
        worker = PatchWorker(self.model, self.glossary, self.rate_limiter, self.state,
                             plan.source_language, self.target_lang, self._emit)

        with self._sigint_handler():
            results = self._translate_all(worker, plan.patches, plan.chapters)

        logger.info(f"Final glossary size: {len(self.glossary)} terms")
        logger.info(f"API requests: {self.state.attempts} sent, {self.state.answered} answered")

        self._report_outcome(results, len(plan.patches), output_path)

    def _translate_all(self, worker: PatchWorker, patches: List[List[Chapter]],
                       chapters: List[Chapter]) -> List[PatchResult]:
        """
        Run every patch through the pool, saving the book as each one lands.

        Patches still queued when a stop is asked for are cancelled outright.
        Those never produce a result, which is the point — they were never sent,
        so there is nothing to report about them.
        """
        results: List[PatchResult] = []
        announced_stop = False

        with ThreadPoolExecutor(max_workers=self.requests_per_minute) as executor:
            futures = [
                executor.submit(worker.translate, patch, i + 1)
                for i, patch in enumerate(patches)
            ]

            self._emit('concurrency', f"Starting batch of {min(self.requests_per_minute, len(patches))} "
                                      f"patches. {len(patches)} total in queue")

            for future in as_completed(futures):
                # Once, not once per remaining future: this used to repeat the
                # same interrupt line for every patch left in the queue.
                if self.should_stop and not announced_stop:
                    announced_stop = True
                    self._emit('interrupt', "Stop signal received. Cancelling remaining patches...",
                               event='stopping')
                    for pending in futures:
                        pending.cancel()

                try:
                    result = future.result()
                except CancelledError:
                    # Asked for, so not worth a word. Reporting it as an error is
                    # what made cancelling a run print a blank-message failure.
                    continue
                except Exception as e:
                    self._emit('error', f"Unexpected error in patch processing: {e}")
                    continue

                results.append(result)

                # Workers translate the chapters in place, so `chapters` already
                # reflects this patch and is ready to be saved. A patch that
                # failed has already said so from inside the worker.
                if result.success:
                    self._save_progress(chapters)

        return results

    @contextmanager
    def _sigint_handler(self):
        """
        Ctrl+C stops the run gracefully, for as long as this is held.

        Only the main thread may install a signal handler, so a server running
        this in a worker relies on setting `should_stop` directly instead.
        """
        on_main_thread = threading.current_thread() is threading.main_thread()
        previous = None

        if on_main_thread:
            logger.system("Press Ctrl+C to stop gracefully "
                          "(in-flight patches finish, progress is saved)")
            previous = signal.signal(signal.SIGINT, self._request_stop)

        try:
            yield
        finally:
            if on_main_thread:
                # `signal.signal` returns None when the previous handler was not
                # set from Python, and handing None back raises. SIG_DFL and
                # SIG_IGN survive the `or` — they are 0 and 1, and 0 is what we
                # would substitute anyway.
                signal.signal(signal.SIGINT, previous or signal.SIG_DFL)

    def _report_outcome(self, results: List[PatchResult], total: int, output_path: str):
        """The one 'finished' event, whatever shape the run ended in."""
        successful = sum(1 for r in results if r.success)

        if self.aborted_after_failures:
            level = 'error'
            message = (f"Gave up after {successful}/{total} patches: too many failed in a row. "
                       f"What was translated is saved to: {output_path}")
        elif self.should_stop:
            level = 'info'
            message = (f"Stopped after {successful}/{total} patches. "
                       f"What was translated is saved to: {output_path}")
        elif successful == 0:
            level = 'info'
            message = "No patches were successfully translated."
        elif successful == total:
            level = 'success'
            message = f"Successfully created translated EPUB: {output_path}"
        else:
            level = 'info'
            message = (f"Translation incomplete: {successful}/{total} patches completed. "
                       f"Partial translation saved to: {output_path}")

        self._emit(level, message, event='finished', successful=successful, total=total)

    def _save_progress(self, chapters: List[Chapter]):
        """Save the current translation progress. Serialized: workers share one writer."""
        done, total = self.state.completed_patches, self.state.total_patches
        try:
            with self._save_lock:
                logger.progress(f"Auto-saving progress: {done}/{total} patches completed")
                self.writer.save(chapters, self.output_path)
                self._emit('success', f"Progress saved to: {self.output_path}",
                           event='saved', completed=done, total=total)
        except Exception as e:
            self._emit('error', f"Auto-save failed: {e}")
