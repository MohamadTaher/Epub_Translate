"""
Translation jobs.

A translation takes minutes to tens of minutes, far longer than a request can be
held open, so each one runs on a background thread and reports progress through a
queue the browser subscribes to.

Jobs live in memory only: restarting the server loses whatever was in flight.
"""

import json
import queue
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from epub_translate import epub_io
from epub_translate.translator import EPUBTranslator, TranslationPlan

from . import budget, config

# Extensions for the cover formats an EPUB may carry, so the file written to disk
# keeps a name the browser can infer a type from.
_COVER_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/svg+xml": ".svg",
}

# Sent to a subscriber when the job has no more events coming.
STREAM_END = object()


@dataclass
class Job:
    id: str
    work_dir: Path
    client_ip: str
    options: Dict
    original_filename: str = "book.epub"
    book_title: Optional[str] = None
    book_author: Optional[str] = None
    cover_name: Optional[str] = None   # file name within work_dir, when the book had cover art
    status: str = "preparing"          # preparing|ready|running|done|failed|cancelled
    error: Optional[str] = None
    plan: Optional[TranslationPlan] = None
    translator: Optional[EPUBTranslator] = None
    completed: int = 0
    total: int = 0
    log: List[Dict] = field(default_factory=list)
    subscribers: List[queue.Queue] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def input_path(self) -> Path:
        return self.work_dir / "input.epub"

    @property
    def output_path(self) -> Path:
        return self.work_dir / "translated.epub"

    @property
    def glossary_path(self) -> Path:
        return self.work_dir / "glossary.json"

    @property
    def cover_path(self) -> Optional[Path]:
        return self.work_dir / self.cover_name if self.cover_name else None

    @property
    def download_name(self) -> str:
        """What the browser should call the finished book."""
        stem = Path(self.original_filename).stem or "book"
        return f"{stem} ({self.options.get('target_lang', 'translated')}).epub"

    def publish(self, event: Dict):
        """Record an event and fan it out to anyone watching."""
        with self.lock:
            if event.get('event') == 'patch_done':
                self.completed = event.get('completed', self.completed)
                self.total = event.get('total', self.total)
            elif event.get('event') == 'started':
                self.total = event.get('total', self.total)

            self.log.append(event)
            subscribers = list(self.subscribers)

        for subscriber in subscribers:
            subscriber.put(event)

    def subscribe(self) -> queue.Queue:
        """
        Watch this job's events. The returned queue is primed with everything
        that has already happened, so a browser that connects late — or
        reconnects — still sees the whole run.
        """
        subscriber = queue.Queue()
        with self.lock:
            for event in self.log:
                subscriber.put(event)
            if self.status in ("done", "failed", "cancelled"):
                subscriber.put(STREAM_END)
            else:
                self.subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue):
        with self.lock:
            if subscriber in self.subscribers:
                self.subscribers.remove(subscriber)

    def snapshot(self) -> Dict:
        with self.lock:
            return {
                'id': self.id,
                'status': self.status,
                'error': self.error,
                'completed': self.completed,
                'total': self.total,
                'stats': self.stats(),
            }

    def stats(self) -> Optional[Dict]:
        """
        The plan, as one list of chapters in book order.

        Kept in book order, with the skipped ones left in place, so a reader can
        recognise their book in the list. `patch` is the number the progress
        events use, which is what lets a chapter be shown as translated.
        """
        if not self.plan:
            return None

        patch_of = {
            id(chapter): index + 1
            for index, patch in enumerate(self.plan.patches)
            for chapter in patch
        }

        return {
            'book': {
                'title': self.book_title,
                'author': self.book_author,
                'has_cover': bool(self.cover_name),
                'filename': self.original_filename,
            },
            'chapter_count': len(self.plan.chapters),
            'patch_count': len(self.plan.patches),
            'total_tokens': self.plan.total_tokens,
            'source_language': self.plan.source_language,
            'target_language': self.options.get('target_lang', ''),
            'chapters': [
                {
                    'id': chapter['id'],
                    'title': chapter['title'],
                    'file_name': chapter['file_name'],
                    'tokens': chapter.get('source_tokens', 0),
                    'patch': patch_of.get(id(chapter)),
                    'skip_reason': (
                        None if id(chapter) in patch_of
                        else chapter.get('skip_reason') or "Not selected"
                    ),
                }
                for chapter in self.plan.chapters
            ],
        }


class JobStore:
    """Holds every live job and the thread pool that runs them."""

    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=config.MAX_TRANSLATIONS_AT_ONCE)

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for job in self._jobs.values() if job.status in ("preparing", "ready", "running"))

    def create(self, epub_bytes: bytes, options: Dict, client_ip: str,
               filename: str = "book.epub") -> Job:
        """Store the upload and work out a plan for it, without calling the API."""
        work_dir = Path(tempfile.mkdtemp(dir=_ensure_scratch_dir()))
        job = Job(id=uuid.uuid4().hex, work_dir=work_dir, client_ip=client_ip,
                  options=options, original_filename=filename)
        job.input_path.write_bytes(epub_bytes)
        job.glossary_path.write_text("{}", encoding="utf-8")

        with self._lock:
            self._jobs[job.id] = job

        try:
            _load_book_info(job)
            translator = self._build_translator(job)
            job.plan = translator.prepare(
                str(job.input_path),
                max_tokens_per_patch=config.TOKENS_PER_REQUEST,
            )
            job.total = len(job.plan.patches)
            job.status = "ready"
        except Exception as e:
            job.status = "failed"
            job.error = f"Could not read this EPUB: {e}"

        return job

    def preview(self, job: Job, only_chapter_ids: Optional[Set[str]]) -> Dict:
        """
        Re-cost a chapter selection without committing to it.

        Packs the token counts already cached on the plan, so this costs nothing
        and reads nothing from disk — a visitor ticking chapters gets the real
        request count back, not an estimate, and `start` will group them the same
        way because both go through `pack_by_tokens`.
        """
        if only_chapter_ids is None:
            # Same default as `prepare`: leave out what already looks translated.
            chapters = [ch for ch in job.plan.chapters if not ch.get('already_translated', False)]
        else:
            chapters = [ch for ch in job.plan.chapters if ch['id'] in only_chapter_ids]

        patches = epub_io.pack_by_tokens(
            [(chapter, chapter.get('source_tokens', 0)) for chapter in chapters],
            config.TOKENS_PER_REQUEST,
        )

        preview_plan = TranslationPlan(
            chapters=job.plan.chapters,
            patches=patches,
            total_tokens=sum(ch.get('source_tokens', 0) for patch in patches for ch in patch),
            source_language=job.plan.source_language,
        )

        # stats() reads whichever plan is on the job, so swap the preview in just
        # long enough to render it. The real plan is never replaced.
        with job.lock:
            real_plan, job.plan = job.plan, preview_plan
            try:
                return job.stats()
            finally:
                job.plan = real_plan

    def replan(self, job: Job, only_chapter_ids: Optional[Set[str]] = None) -> int:
        """
        Re-plan against the visitor's chapter selection and return how many API
        requests it would take, so the caller can check that against the budget
        before any money is spent.

        Planning again rather than reusing the plan from upload keeps the chapter
        objects unmutated, and is cheap next to the translation itself.
        """
        translator = self._build_translator(job)

        def listener(event: Dict):
            _count_against_budget(event)
            job.publish(event)

        translator.on_event = listener

        job.plan = translator.prepare(
            str(job.input_path),
            max_tokens_per_patch=config.TOKENS_PER_REQUEST,
            only_chapter_ids=only_chapter_ids,
        )
        job.translator = translator
        job.total = len(job.plan.patches)
        return len(job.plan.patches)

    def launch(self, job: Job):
        """Begin translating the plan set by `replan`."""
        job.status = "running"
        budget.record_run(job.client_ip)
        self._executor.submit(self._run, job)

    def _run(self, job: Job):
        try:
            job.translator.run(job.plan, str(job.input_path), str(job.output_path))
            if job.translator.should_stop:
                job.status = "cancelled"
            elif job.total and job.completed == 0:
                # Every patch exhausted its retries; there is no book to download.
                job.status = "failed"
                job.error = "No chapters could be translated. The API rejected every request."
            else:
                job.status = "done"
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.publish({'level': 'error', 'message': f"Translation failed: {e}", 'event': 'finished'})
        finally:
            with job.lock:
                subscribers = list(job.subscribers)
                job.subscribers.clear()
            for subscriber in subscribers:
                subscriber.put(STREAM_END)

    def _build_translator(self, job: Job) -> EPUBTranslator:
        return EPUBTranslator(
            api_key=config.GEMINI_API_KEY,
            source_language=job.options['source_lang'],
            target_language=job.options['target_lang'],
            glossary_file_path=str(job.glossary_path),
            requests_per_minute=config.REQUESTS_PER_MINUTE,
            tokens_per_minute=config.TOKENS_PER_MINUTE,
            model_name=config.GEMINI_MODEL,
        )

    def reap_expired(self):
        """Delete finished jobs past their expiry, along with their uploads."""
        cutoff = time.time() - config.DELETE_UPLOADS_AFTER_MINUTES * 60
        with self._lock:
            expired = [
                job for job in self._jobs.values()
                if job.created_at < cutoff and job.status in ("done", "failed", "cancelled", "ready")
            ]
            for job in expired:
                del self._jobs[job.id]

        for job in expired:
            shutil.rmtree(job.work_dir, ignore_errors=True)


def _load_book_info(job: Job):
    """
    Record the book's title, author and cover on the job.

    Cosmetic, so a book that carries none of it still translates: anything that
    goes wrong here leaves the fields empty rather than failing the upload.
    """
    try:
        info = epub_io.read_book_info(str(job.input_path))
    except Exception:
        return

    job.book_title = info.get('title')
    job.book_author = info.get('author')

    cover_bytes = info.get('cover_bytes')
    if not cover_bytes:
        return

    extension = _COVER_EXTENSIONS.get(info.get('cover_media_type') or "", ".jpg")
    try:
        (job.work_dir / f"cover{extension}").write_bytes(cover_bytes)
        job.cover_name = f"cover{extension}"
    except OSError:
        pass


def _count_against_budget(event: Dict):
    """Charge the daily budget for each API call as it is made."""
    if event.get('event') == 'patch_start':
        budget.record_request()


def _ensure_scratch_dir() -> Path:
    scratch = config.DATA_DIR / "jobs"
    scratch.mkdir(parents=True, exist_ok=True)
    return scratch


def read_glossary(job: Job) -> Dict[str, str]:
    try:
        return json.loads(job.glossary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


store = JobStore()
