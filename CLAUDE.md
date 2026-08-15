# EPUB Translate

Gemini-powered EPUB translator. A CLI (`translate_epub.py`) and a FastAPI backend
(`server/`) share the translation core in `epub_translate/`.

`docker compose up --build` builds both and serves the app on
[localhost:7860](http://localhost:7860).

## The frontend

`web/` is React 19 + Vite + TypeScript with **plain CSS** — CSS Modules, which Vite
supports with no dependency. There is deliberately no Tailwind, no router, no state
library and no SSE library; the whole dependency list is React, Vite, TypeScript and
two self-hosted fonts.

The design is editorial rather than technical: warm paper, one clay accent, and a
split between two typefaces that carries meaning — **Literata for anything that comes
from the book** (its title, chapter names, headings) and **Inter for the interface**
around it. Every colour lives in `web/src/styles/tokens.css` with its measured
contrast ratio in a comment; keep new colours going through those tokens. Only a light
theme exists, but the tokens are structured so a dark one is a second block rather
than a rewrite.

Two things worth knowing before changing it:

- **`EventSource` must be closed in the `end` handler.** The server closes the stream
  when a job finishes, EventSource reconnects on close, and the job is terminal by
  then — so it would replay the whole log and send `end` again, forever. See
  `web/src/useJobStream.ts`.
- **The word is "request", not "patch".** The server calls a batch of chapters a
  patch; everywhere a reader can see, it is a request, because that is the unit the
  limits and the daily budget are counted in. `ActivityLog.tsx` rewrites the server's
  wording rather than leaking it.

Build output must land in `web/dist` — `Dockerfile:26` copies it and
`server/config.py:57` points `STATIC_DIR` there. `server/app.py` mounts it only
`if config.STATIC_DIR.exists()` and that is evaluated at import, so a build has to
precede the server starting.

## This machine runs everything through Docker

**Node, npm and Python are not on the host PATH** — only Docker is. Never run `npm`,
`npx`, `tsc`, `pip` or `uvicorn` directly; they will fail with "not found". Anything
needing a toolchain has to go through a container.

For a Node toolchain without a full image build, a throwaway container over a bind
mount works and takes seconds:

```
docker run --rm -v "${PWD}\web:/w" -w /w node:20-alpine sh -c "npm install && npm run build"
```

Node 20 is the ceiling (`Dockerfile:2`), which Vite 7 needs 20.19+ to satisfy — the
image currently ships 20.20, so don't pin tooling that wants Node 22.

Changes to `.env` need only a restart (`docker compose restart`); it is mounted into
the container rather than baked into the image, but settings are read once at startup.

Everything about how a run is paced lives there and nowhere else — the UI offers a
visitor only the two languages. `REQUESTS_PER_MINUTE` is deliberately one number
doing two jobs, capping the rate limiter *and* sizing the worker pool: requests take
tens of seconds, so a larger pool would only park threads inside the limiter.
`/api/status` reports it because the ETA needs it, not because it is editable.

To iterate on the UI with hot reload, run Vite in a container and point its proxy at
the API container:

```
docker run --rm -p 5173:5173 -v "${PWD}\web:/w" -w /w \
  -e API_TARGET=http://host.docker.internal:7860 \
  node:20-alpine sh -c "npm run dev -- --host"
```

## Deployment target is Cloud Run

The container listens on `$PORT` and falls back to 7860, so compose and Cloud Run
both work untouched. `CMD` is shell form on purpose — it needs the variable
expanded at runtime — with `exec` in front so uvicorn stays PID 1 and still gets
the SIGTERM Cloud Run sends 10 seconds before it kills an instance. Docker lints
shell-form `CMD` as a signal hazard; the `exec` is what answers that, so don't
"fix" the warning by switching to JSON form.

Three deploy flags in `README.md` are load-bearing, and all three exist because
this app does its work *after* answering the request:

- `--no-cpu-throttling`, or the background translation thread loses the CPU as
  soon as `/start` returns.
- `--max-instances 1`, because jobs live in one process's memory and session
  affinity is best-effort. Anything that assumes a second instance could serve the
  same job is wrong unless job state moves out of memory first.
- `--timeout 3600`, the ceiling on the SSE stream, which is a single long request.
  60 minutes is Cloud Run's maximum, not a choice.

`DATA_DIR` is overridden to `/tmp/epub_translate` there. The Cloud Run filesystem
is in memory and counts against the instance's memory limit, so uploads are
charged twice over — once as files, once as RAM.

## API surface

The frontend talks to these endpoints. The API key never leaves the server, so
translation cannot be driven directly from a browser against Google.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/status` | Config, model, remaining budget, limits, cooldown, languages |
| `POST` | `/api/jobs` | Upload an EPUB (multipart), returns an analysed plan |
| `POST` | `/api/jobs/{id}/preview` | Re-cost a chapter selection; spends and commits nothing |
| `POST` | `/api/jobs/{id}/start` | Confirm the plan and begin; enforces every limit |
| `GET` | `/api/jobs/{id}` | Snapshot, for reconnecting after a dropped stream |
| `GET` | `/api/jobs/{id}/events` | Server-sent events for the life of the run |
| `GET` | `/api/jobs/{id}/cover` | The book's own cover art, 404 when it has none |
| `POST` | `/api/jobs/{id}/cancel` | Stop after in-flight batches finish; returns a snapshot |
| `GET` | `/api/jobs/{id}/download` | Translated EPUB; valid mid-run too |
| `GET`/`PUT` | `/api/jobs/{id}/glossary` | Read and replace glossary terms |

`stats` is one `chapters` array in **book order**, with skipped chapters left in
place. Each carries `patch` — the 1-based request number the progress events use,
`null` when it is not being translated — plus its own `tokens` and `skip_reason`.
That `patch` number is the only thing tying a `patch_done` event back to chapters.

`/preview` exists so the request count shown to a reader is the server's real answer
rather than a guess: it packs the token counts already cached on the plan through the
same `pack_by_tokens` that `/start` uses, so the two always agree. Don't reimplement
the packing rule anywhere else.

Progress arrives only over the SSE stream — a run outlives any single request. The
book is re-saved after every batch, so `/download` returns a valid partial EPUB at any
point during a run. Every connection replays the job's whole event log before going
live, so reconnecting loses nothing but duplicates everything.

A run at the default 4 requests/minute spends most of its time waiting on the rate
limiter. Those waits emit `rate_limit_wait` / `rate_limit_done` events; without them
the UI has no way to tell waiting apart from hung.

## Layout

```
translate_epub.py    CLI entry point
epub_translate/      translation core, shared by CLI and server
  cli.py             argument parsing
  translator.py      orchestration, retries, rate limiting
  epub_io.py         EPUB read/write, chapter extraction, language detection
  glossary.py        glossary loading and term matching
  prompts.py         prompt templates
  rate_limiter.py    API rate limiting
  tokens.py          token counting (cl100k_base, approximate for Gemini)
server/
  app.py             HTTP endpoints and static file serving
  jobs.py            background runs, progress fan-out, selection preview
  budget.py          daily spend ceiling
  config.py          settings, all environment-overridable
web/                 React + Vite frontend, builds to web/dist
  src/
    App.tsx          phase machine: upload -> plan -> run
    api.ts           fetch wrappers; ApiError carries the HTTP status
    useJobStream.ts  EventSource lifecycle and per-request progress
    format.ts        number and duration wording
    styles/          design tokens and element defaults
    components/      one folder-flat component + CSS Module each
```

## Gemini models

`gemini-2.5-pro` has a **zero** free-tier quota — a key without billing gets a 429 on
every request, which surfaces as a run that retries ten times and then fails. Set
`GEMINI_MODEL=gemini-2.5-flash` in `.env` to work on a free-tier key.

Failed attempts are not recorded by the rate limiter (`record_request` is only called
on success), so a quota-exhausted key is retried ten times in quick succession rather
than being backed off. Worth knowing when a run fails fast.
