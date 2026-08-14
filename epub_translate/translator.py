import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

import google.generativeai as genai
from bs4 import BeautifulSoup

from . import epub_io
from .glossary import GlossaryManager
from .logging_utils import confirm, logger
from .prompts import build_translation_prompt
from .rate_limiter import RateLimiter
from .tokens import count_tokens

CHAPTER_SEPARATOR = "<!-- CHAPTER_SEPARATOR -->"


@dataclass
class PatchResult:
    """Outcome of translating a single patch (group of chapters)."""
    patch_index: int
    success: bool
    attempts: int


@dataclass
class TranslationPlan:
    """
    What a translation run would do, worked out before any API call is made, so
    the cost can be reviewed and confirmed first.
    """
    chapters: List[Dict]
    patches: List[List[Dict]] = field(default_factory=list)
    total_tokens: int = 0
    source_language: str = "auto"


class EPUBTranslator:
    """
    Main translator class that handles EPUB book translation using Google's Gemini API.
    Features concurrent translation, rate limiting, auto-save, and glossary management.
    """

    def __init__(self, api_key: str, source_language: str = "auto", target_language: str = "English",
                 max_concurrent: int = 5, glossary_file_path: str = None, max_requests_per_minute: int = 4,
                 max_tokens_per_minute: int = 250000, model_name: str = "gemini-2.5-pro",
                 on_event: Optional[Callable[[Dict], None]] = None):
        if not api_key:
            raise ValueError("API key is required")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.on_event = on_event
        self.source_lang = source_language
        self.target_lang = target_language
        self.max_concurrent = max_concurrent
        self.rate_limiter = RateLimiter(
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            on_wait=self._on_rate_limit_wait,
            on_resume=self._on_rate_limit_resume,
        )
        self.should_stop = False
        self.total_api_requests = 0
        self.completed_patches_count = 0
        self.consecutive_failures = 0
        self.lock = threading.Lock()

        # Set once translation starts; shared by the workers for auto-saving progress.
        self.writer = None
        self.output_path = None
        self.total_patches = 0

        self.glossary_manager = GlossaryManager(glossary_file_path)

        self._emit('system', f"Concurrency limit: {max_concurrent}")
        self._emit('system', f"Rate limits: {max_requests_per_minute} req/min, {max_tokens_per_minute:,} tokens/min")
        if glossary_file_path:
            self._emit('system', f"Glossary loaded: {self.glossary_manager.get_glossary_size()} terms")

    def _emit(self, level: str, message: str, **data):
        """
        Log a message and, if a listener is attached, hand it the same message
        as structured data. Called from worker threads, so listeners must be
        thread-safe.
        """
        getattr(logger, level)(message)

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

    def prepare(self, epub_path: str, max_tokens_per_patch: int = 15000,
                only_chapter_ids: Optional[Set[str]] = None) -> TranslationPlan:
        """
        Work out what translating this book would involve, without calling the API.

        By default, chapters detected as already translated are left out. Pass
        `only_chapter_ids` to translate exactly the chapters named instead.
        """
        chapters, resolved_language = epub_io.extract_chapters(epub_path, self.source_lang)

        if not chapters:
            self._emit('error', "No chapters found in the EPUB.")
            return TranslationPlan(chapters=[], source_language=resolved_language)

        # Every chapter is measured, not just the ones about to be translated, so
        # a caller can re-cost a different selection without reading the book again.
        epub_io.count_chapter_tokens(chapters)

        if only_chapter_ids is None:
            patches = epub_io.create_patches(chapters, max_tokens_per_patch)
        else:
            selected = [ch for ch in chapters if ch['id'] in only_chapter_ids]
            patches = epub_io.create_patches(selected, max_tokens_per_patch, include_translated=True)

        total_tokens = sum(ch['source_tokens'] for patch in patches for ch in patch)

        return TranslationPlan(
            chapters=chapters,
            patches=patches,
            total_tokens=total_tokens,
            source_language=resolved_language,
        )

    def translate_book(self, epub_path: str, output_path: str, max_tokens_per_patch: int = 15000):
        """
        Main entry point for translating an EPUB book.
        Extracts chapters, groups them into patches, and manages concurrent translation.
        """
        plan = self.prepare(epub_path, max_tokens_per_patch)

        if not plan.chapters:
            return

        if not plan.patches:
            logger.system("All chapters are already translated. Creating final EPUB...")
            epub_io.EpubWriter(epub_path, self.target_lang).save(plan.chapters, output_path)
            logger.success(f"Successfully created translated EPUB: {output_path}")
            return

        logger.system(f"Total tokens to translate: {plan.total_tokens:,}")
        logger.system(f"Max tokens per patch: {max_tokens_per_patch:,}")
        logger.system(f"Patches created: {len(plan.patches)}")

        if not confirm("Proceed with translation?"):
            logger.system("Translation cancelled.")
            return

        self.run(plan, epub_path, output_path)

    def run(self, plan: TranslationPlan, epub_path: str, output_path: str):
        """
        Execute a prepared plan: translate every patch concurrently, saving the
        book after each one. Safe to call off the main thread.
        """
        chapters = plan.chapters
        patches = plan.patches

        self.writer = epub_io.EpubWriter(epub_path, self.target_lang)
        self.output_path = output_path
        self.total_patches = len(patches)

        self._emit('system', "🚀 Starting concurrent translation with auto-save enabled",
                   event='started', total=len(patches))
        self._emit('system', "Progress will be saved after each successful patch")

        self.completed_patches_count = 0
        self.consecutive_failures = 0
        results = []

        # signal.signal only works on the main thread, so a server running this
        # in a worker relies on setting `should_stop` directly instead.
        on_main_thread = threading.current_thread() is threading.main_thread()
        previous_handler = None
        if on_main_thread:
            logger.system("Press Ctrl+C to stop gracefully (in-flight patches finish, progress is saved)")
            previous_handler = signal.signal(signal.SIGINT, self._request_stop)

        try:
            with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
                futures = [
                    executor.submit(self._translate_patch_worker, patch, i + 1)
                    for i, patch in enumerate(patches)
                ]

                self._emit('concurrency', f"Starting batch of {min(self.max_concurrent, len(patches))} patches. {len(patches)} total in queue")

                for future in as_completed(futures):
                    if self.should_stop:
                        self._emit('interrupt', "Stop signal received. Cancelling remaining patches...",
                                   event='stopping')
                        for f in futures:
                            if not f.done():
                                f.cancel()

                    try:
                        result = future.result()
                    except Exception as e:
                        self._emit('error', f"Unexpected error in patch processing: {e}")
                        continue

                    results.append(result)

                    # Workers translate the chapter dicts in place, so `chapters`
                    # already reflects this patch and is ready to be saved.
                    if result.success:
                        self._save_progress(chapters)
                    else:
                        self._emit('error', f"Patch {result.patch_index}/{len(patches)} failed after {result.attempts} attempts",
                                   event='patch_failed', patch=result.patch_index)
        finally:
            if on_main_thread:
                signal.signal(signal.SIGINT, previous_handler)

        successful_results = [r for r in results if r.success]

        logger.info(f"Final glossary size: {self.glossary_manager.get_glossary_size()} terms")
        logger.info(f"Total API requests made: {self.total_api_requests}")

        if not successful_results:
            self._emit('info', "No patches were successfully translated.",
                       event='finished', successful=0, total=len(patches))
        elif len(successful_results) == len(patches):
            logger.info("All patches completed successfully. Final save...")
            self._save_progress(chapters)
            self._emit('success', f"Successfully created translated EPUB: {output_path}",
                       event='finished', successful=len(successful_results), total=len(patches))
        else:
            self._emit('info', f"Translation incomplete: {len(successful_results)}/{len(patches)} patches completed. "
                               f"Partial translation saved to: {output_path}",
                       event='finished', successful=len(successful_results), total=len(patches))

    def _save_progress(self, chapters: List[Dict]):
        """Save the current translation progress. Serialized: workers share one writer."""
        try:
            with self.lock:
                logger.progress(f"Auto-saving progress: {self.completed_patches_count}/{self.total_patches} patches completed")
                self.writer.save(chapters, self.output_path)
                self._emit('success', f"Progress saved to: {self.output_path}",
                           event='saved', completed=self.completed_patches_count, total=self.total_patches)
        except Exception as e:
            self._emit('error', f"Auto-save failed: {e}")

    def _translate_patch_worker(self, patch: List[Dict], patch_index: int, max_retries: int = 10) -> PatchResult:
        """
        Worker function to translate a single patch with glossary support.
        Handles retries, rate limiting, and glossary term lookup.
        """
        start_time = time.time()

        combined_html = f"\n{CHAPTER_SEPARATOR}\n".join(ch['source_html'] for ch in patch)
        combined_text = BeautifulSoup(combined_html, 'html.parser').get_text()

        try:
            relevant_terms = self.glossary_manager.extract_relevant_terms(combined_text)
            glossary_section = self.glossary_manager.create_glossary_prompt_section(relevant_terms)
        except RuntimeError as e:
            self._emit('warning', f"Glossary iteration error for patch {patch_index}, using empty glossary: {e}")
            glossary_section = ""

        prompt_tokens = sum(ch['source_tokens'] for ch in patch) + count_tokens(glossary_section) + 1000
        estimated_total_tokens = int(prompt_tokens * 1.4)

        prompt = build_translation_prompt(self.source_lang, self.target_lang, glossary_section, combined_html)

        last_error = None

        for attempt in range(max_retries):
            if self.should_stop:
                return PatchResult(patch_index, success=False, attempts=attempt)

            try:
                self.rate_limiter.wait_for_availability(estimated_total_tokens, stop_check=lambda: self.should_stop)
                if self.should_stop:
                    return PatchResult(patch_index, success=False, attempts=attempt)

                self._emit('api', f"Sending patch {patch_index} (attempt {attempt + 1}/{max_retries})",
                           event='patch_start', patch=patch_index, attempt=attempt + 1)

                response = self.model.generate_content(prompt)
                if not response or not response.text:
                    raise Exception("Empty response from API")

                response_text = self._strip_code_fence(response.text.strip())

                translated_parts = response_text.split(CHAPTER_SEPARATOR)
                if len(translated_parts) != len(patch):
                    self._emit(
                        'warning',
                        f"Patch {patch_index}: expected {len(patch)} chapter separator(s) in response, "
                        f"got {len(translated_parts)}. Some chapters in this patch may not be updated."
                    )

                for chapter, translated_part in zip(patch, translated_parts):
                    if translated_part.strip():
                        translated_soup = BeautifulSoup(translated_part.strip(), 'html.parser')
                        chapter['soup'] = translated_soup
                        chapter['translated_title'] = epub_io.extract_chapter_title(translated_soup)

                self.rate_limiter.record_request(prompt_tokens + count_tokens(response_text))

                with self.lock:
                    self.total_api_requests += 1
                    self.completed_patches_count += 1
                    self.consecutive_failures = 0

                elapsed = time.time() - start_time
                self._emit('success', f"Patch {patch_index} completed in {elapsed:.1f}s "
                                      f"[{self.completed_patches_count}/{self.total_patches}]",
                           event='patch_done', patch=patch_index, seconds=round(elapsed, 1),
                           completed=self.completed_patches_count, total=self.total_patches)

                return PatchResult(patch_index, success=True, attempts=attempt + 1)

            except Exception as e:
                last_error = str(e).split(' ')[0]
                self._emit('warning', f"Patch {patch_index}/{self.total_patches} attempt {attempt + 1}/{max_retries} failed: {last_error}",
                           event='attempt_failed', patch=patch_index, attempt=attempt + 1)

        with self.lock:
            self.consecutive_failures += 1
            if self.consecutive_failures >= 6:
                self._emit('error', "6 consecutive patch failures detected. API may be down. Stopping translation.")
                self.should_stop = True

        self._emit('error', f"Patch {patch_index}/{self.total_patches} failed after {max_retries} attempts")

        return PatchResult(patch_index, success=False, attempts=max_retries)

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """Remove a wrapping markdown code fence, which the model sometimes adds."""
        for marker in ["```html", "```"]:
            if text.startswith(marker):
                text = text[len(marker):]
            if text.endswith(marker):
                text = text[:-len(marker)]
        return text.strip()
