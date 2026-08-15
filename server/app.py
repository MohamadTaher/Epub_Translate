"""
HTTP interface to the translator.

The React app runs in the visitor's browser and the translator runs here, so
everything they exchange goes over these endpoints. Keeping translation on this
side is also what keeps the API key off the visitor's machine.
"""

import json
import queue
import threading
import time
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from epub_translate import epub_io

from . import budget, config, jobs

app = FastAPI(title="EPUB Translate")


@app.on_event("startup")
def start_reaper():
    """Delete expired jobs and their uploads in the background, forever."""
    def reap_forever():
        while True:
            time.sleep(60)
            try:
                jobs.store.reap_expired()
            except Exception:
                pass    # a failed sweep must not end the loop

    threading.Thread(target=reap_forever, daemon=True).start()


class StartRequest(BaseModel):
    chapter_ids: Optional[List[str]] = None


class PreviewRequest(BaseModel):
    chapter_ids: Optional[List[str]] = None


class GlossaryRequest(BaseModel):
    terms: Dict[str, str]


def _client_ip(request: Request) -> str:
    # Most hosts sit behind a proxy, so the socket address is the proxy's,
    # not the visitor's.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _require_job(job_id: str) -> jobs.Job:
    job = jobs.store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="This job has expired or never existed.")
    return job


@app.get("/api/status")
def get_status(request: Request):
    """
    What the UI needs to know before offering an upload.

    Includes the two reasons a translation would be refused after the fact — the
    per-visitor cooldown and a server already at capacity — so they can be said
    up front rather than after a book has been uploaded and reviewed.
    """
    return {
        'configured': bool(config.GEMINI_API_KEY),
        'model': config.GEMINI_MODEL,
        'remaining_requests': budget.remaining_today(),
        'daily_budget': config.DAILY_REQUEST_BUDGET,
        'max_requests_per_book': config.MAX_REQUESTS_PER_BOOK,
        'max_upload_mb': config.MAX_UPLOAD_MB,
        'cooldown_seconds': int(budget.cooldown_remaining(_client_ip(request)).total_seconds()),
        'busy': jobs.store.active_count() >= config.MAX_TRANSLATIONS_AT_ONCE,
        # Not a setting the UI offers, only the number it needs to estimate how
        # long a run will take.
        'requests_per_minute': config.REQUESTS_PER_MINUTE,
        'languages': sorted(epub_io.LANGUAGE_CODES),
        'detectable_languages': sorted(epub_io.SCRIPT_RANGES),
    }


@app.post("/api/jobs")
async def create_job(
    request: Request,
    file: UploadFile = File(...),
    source_lang: str = Form("auto"),
    target_lang: str = Form("English"),
):
    """Accept an EPUB and report what translating it would involve."""
    if not config.GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="This server has no API key configured.")

    if not file.filename or not file.filename.lower().endswith(".epub"):
        raise HTTPException(status_code=400, detail="Please upload a .epub file.")

    if jobs.store.active_count() >= config.MAX_TRANSLATIONS_AT_ONCE:
        raise HTTPException(status_code=429, detail="The server is busy with other translations. Try again shortly.")

    # Read in chunks so an oversized upload is rejected before it is all in memory.
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    chunks, total = [], 0
    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail=f"That file is over the {config.MAX_UPLOAD_MB} MB limit.")
        chunks.append(chunk)

    job = jobs.store.create(
        epub_bytes=b"".join(chunks),
        options={
            'source_lang': source_lang,
            'target_lang': target_lang,
        },
        client_ip=_client_ip(request),
        filename=file.filename,
    )

    if job.status == "failed":
        raise HTTPException(status_code=400, detail=job.error)

    return job.snapshot()


@app.post("/api/jobs/{job_id}/start")
def start_job(job_id: str, body: StartRequest):
    """Confirm the plan and begin spending. Every limit is enforced here."""
    job = _require_job(job_id)

    if job.status != "ready":
        raise HTTPException(status_code=409, detail=f"This job is {job.status}, not ready to start.")

    cooldown = budget.cooldown_remaining(job.client_ip)
    if cooldown.total_seconds() > 0:
        raise HTTPException(
            status_code=429,
            detail=f"Please wait {int(cooldown.total_seconds() // 60) + 1} more minute(s) before starting another translation.",
        )

    selected = set(body.chapter_ids) if body.chapter_ids is not None else None
    planned_patches = jobs.store.replan(job, only_chapter_ids=selected)

    if planned_patches == 0:
        raise HTTPException(status_code=400, detail="No chapters selected to translate.")

    if planned_patches > config.MAX_REQUESTS_PER_BOOK:
        raise HTTPException(
            status_code=413,
            detail=f"This book needs {planned_patches} requests, over the {config.MAX_REQUESTS_PER_BOOK} "
                   f"per-translation limit. Select fewer chapters.",
        )

    if planned_patches > budget.remaining_today():
        raise HTTPException(
            status_code=429,
            detail=f"Only {budget.remaining_today()} requests are left in today's budget, and this needs "
                   f"{planned_patches}. Please try again tomorrow.",
        )

    jobs.store.launch(job)
    return job.snapshot()


@app.post("/api/jobs/{job_id}/preview")
def preview_job(job_id: str, body: PreviewRequest):
    """
    Cost a chapter selection without committing to it.

    Nothing is spent and nothing is decided here, so a visitor can see what
    ticking a chapter does to the request count before agreeing to any of it.
    """
    job = _require_job(job_id)
    if not job.plan:
        raise HTTPException(status_code=409, detail="This job has no plan to preview.")

    selected = set(body.chapter_ids) if body.chapter_ids is not None else None
    return jobs.store.preview(job, only_chapter_ids=selected)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    """Snapshot, for reconnecting after the event stream drops."""
    return _require_job(job_id).snapshot()


@app.get("/api/jobs/{job_id}/cover")
def get_cover(job_id: str):
    """The book's own cover art, when it has any."""
    job = _require_job(job_id)
    cover_path = job.cover_path

    if not cover_path or not cover_path.exists():
        raise HTTPException(status_code=404, detail="This book has no cover.")

    return FileResponse(cover_path, headers={"Cache-Control": "private, max-age=3600"})


@app.get("/api/jobs/{job_id}/events")
def stream_events(job_id: str):
    """
    Server-sent events for the life of the job.

    This is a sync generator on purpose: FastAPI runs it in a worker thread, so
    blocking on the queue is fine and the thread-based translator needs no
    async wrapper.
    """
    job = _require_job(job_id)

    def event_stream():
        subscriber = job.subscribe()
        try:
            while True:
                try:
                    event = subscriber.get(timeout=15)
                except queue.Empty:
                    yield ": keep-alive\n\n"      # stop proxies closing an idle stream
                    continue

                if event is jobs.STREAM_END:
                    yield f"event: end\ndata: {json.dumps(job.snapshot())}\n\n"
                    return

                yield f"data: {json.dumps(event)}\n\n"
        finally:
            job.unsubscribe(subscriber)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    """Stop after in-flight patches finish; whatever is done stays saved."""
    job = _require_job(job_id)
    if job.translator:
        job.translator.should_stop = True
    return job.snapshot()


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str):
    """
    Download the translated EPUB.

    Valid mid-run too: the book is rewritten in full after every patch, so this
    returns a readable EPUB containing everything finished so far.
    """
    job = _require_job(job_id)
    if not job.output_path.exists():
        raise HTTPException(status_code=404, detail="Nothing has been translated yet.")

    return FileResponse(
        job.output_path,
        media_type="application/epub+zip",
        filename=job.download_name,
    )


@app.get("/api/jobs/{job_id}/glossary")
def get_glossary(job_id: str):
    return {'terms': jobs.read_glossary(_require_job(job_id))}


@app.put("/api/jobs/{job_id}/glossary")
def put_glossary(job_id: str, body: GlossaryRequest):
    """Save the glossary. Takes effect on patches not yet sent."""
    job = _require_job(job_id)
    job.glossary_path.write_text(json.dumps(body.terms, ensure_ascii=False, indent=2), encoding="utf-8")

    if job.translator:
        job.translator.glossary_manager.replace_terms(body.terms)

    return {'terms': body.terms}


@app.exception_handler(404)
async def spa_fallback(request: Request, exc):
    """Client-side routes fall through to the app shell; API 404s stay 404s."""
    if request.url.path.startswith("/api/") or not config.STATIC_DIR.exists():
        return JSONResponse({'detail': getattr(exc, 'detail', 'Not found')}, status_code=404)
    return FileResponse(config.STATIC_DIR / "index.html")


if config.STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=config.STATIC_DIR, html=True), name="static")
