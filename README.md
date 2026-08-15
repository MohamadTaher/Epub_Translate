# EPUB Translate

Translates EPUB books using Google's Gemini API. Preserves HTML structure, batches chapters into token-limited patches, and uses a glossary to keep term translations consistent across the book.

Runs two ways: a CLI, and a web app you can deploy.

## Setup

Copy `.env.example` to `.env` and put your key in it:

```
GEMINI_API_KEY=your-key-here
```

Get a key at <https://aistudio.google.com/apikey>. The same `.env` is used by both the CLI and the server. It is gitignored and never copied into the Docker image.

Pick the model in the same file:

```
GEMINI_MODEL=gemini-2.5-flash
```

`gemini-2.5-pro` is the default and gives the best translations, but **it has no free
tier** — on a key without billing enabled every request comes back `429`, and a run
retries ten times before failing. If you are on a free key, set `gemini-2.5-flash`,
which is also far cheaper and faster for testing.

The CLI's `--model` flag overrides this for a single run, and the server reports the
model in use from `GET /api/status`.

If a real environment variable of the same name is set, it wins over `.env` — that is how deployment secrets override the local file.

**Changing `.env` needs a restart, not a rebuild** — the file is mounted into the container, but settings are read once at startup:

```
docker compose restart
```

## Web app

```
docker compose up --build
```

Then open <http://localhost:7860>. Compose mounts your `.env` into the container and
keeps the spend counter in a named volume.

Drop in an EPUB and you get its cover, title and author, then the plan: every chapter
in book order, grouped by the request that will translate it, with chapters that are
already in the target language marked and left out. Ticking chapters re-costs the run
on the server, so the request count and the grouping are always the real ones rather
than an estimate. Nothing is spent until you press start.

During a run the book is re-saved after every request, so the download works part-way
through and gives you a valid EPUB of whatever is finished. Reloading the page picks
the run back up — the job id is in the URL.

The API key stays on the server. Translation is never driven from the browser.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/status` | Model, remaining budget, limits, cooldown, known languages |
| `POST` | `/api/jobs` | Upload an EPUB, get back an analysed plan |
| `POST` | `/api/jobs/{id}/preview` | Re-cost a chapter selection; spends nothing |
| `POST` | `/api/jobs/{id}/start` | Confirm the plan and begin translating |
| `GET` | `/api/jobs/{id}/events` | Server-sent progress events for the whole run |
| `GET` | `/api/jobs/{id}/cover` | The book's cover art, when it has any |
| `POST` | `/api/jobs/{id}/cancel` | Stop after in-flight batches finish |
| `GET` | `/api/jobs/{id}/download` | Translated EPUB, valid mid-run too |
| `GET`/`PUT` | `/api/jobs/{id}/glossary` | Read and replace glossary terms |

A run outlives any single request, so progress arrives only on the event stream.

Because the key stays server-side, this needs a real host and can't run on static
hosting like GitHub Pages.

### Deploying

Any host that can run the Docker image will do. Set `GEMINI_API_KEY` in the host's own configuration rather than shipping a `.env`, which is gitignored and never copied into the image.

#### Cloud Run

```
gcloud run deploy epub-translate \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --no-cpu-throttling \
  --max-instances 1 \
  --timeout 3600 \
  --memory 1Gi \
  --set-env-vars GEMINI_API_KEY=your-key-here,GEMINI_MODEL=gemini-2.5-flash,DATA_DIR=/tmp/epub_translate
```

The first run offers to enable the Cloud Run, Cloud Build and Artifact Registry APIs; the upload respects `.gitignore`, so `node_modules` and `web/dist` stay out of it.

Cloud Run assumes a service's work happens inside a request, and most of this one's happens after the request has been answered. Four of those flags are what bridge that:

| Flag | Why it is there |
|---|---|
| `--no-cpu-throttling` | Translation runs on a background thread after `/start` has returned. By default Cloud Run takes the CPU away the moment a request finishes, which would leave the run frozen mid-book. |
| `--max-instances 1` | Jobs live in one container's memory, and Cloud Run's session affinity is best-effort. A second instance would answer with jobs it has never heard of. |
| `--timeout 3600` | The progress stream is one long-lived request. The five-minute default would sever it repeatedly; 60 minutes is the most Cloud Run allows, and a run capped at `MAX_REQUESTS_PER_BOOK` finishes well inside that. |
| `DATA_DIR=/tmp/epub_translate` | `/tmp` is writable on every Cloud Run instance. The image points this at `/data`, which is where compose mounts a volume instead. |

No port flag is needed: Cloud Run sets `PORT` and the container listens on it, falling back to 7860 everywhere else.

Redeploying is `gcloud run deploy epub-translate --source .` by itself — settings already on the service are kept.

Because the key belongs to whoever deploys it, every visitor spends the owner's money. These limits bound that, and all are settable as environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `DAILY_REQUEST_BUDGET` | `200` | API requests allowed per day across all visitors |
| `MAX_REQUESTS_PER_BOOK` | `25` | Caps the size of a single translation |
| `MINUTES_BETWEEN_TRANSLATIONS` | `30` | Minimum wait between runs from one address; `0` disables it |
| `MAX_UPLOAD_MB` | `25` | Largest accepted EPUB |
| `MAX_TRANSLATIONS_AT_ONCE` | `2` | Translations running at once |
| `DELETE_UPLOADS_AFTER_MINUTES` | `60` | How long an uploaded book is kept before deletion |

How a run is paced is set here too, rather than in the UI — a visitor spending
the owner's quota has no business tuning it:

| Variable | Default | Purpose |
|---|---|---|
| `REQUESTS_PER_MINUTE` | `4` | Request rate, which is also how many run at a time |
| `TOKENS_PER_REQUEST` | `15000` | How much of the book goes into each request |
| `TOKENS_PER_MINUTE` | `250000` | The other half of the Gemini quota |

`REQUESTS_PER_MINUTE` is one number doing both jobs on purpose: requests take
tens of seconds, so a worker pool larger than the per-minute allowance would
only park threads inside the rate limiter.

### Local development

```
pip install -r requirements.txt
uvicorn server.app:app --reload --port 7860
```

That serves the API, plus the last build of the frontend if `web/dist` exists — the
check happens at import, so build before starting.

For the UI with hot reload, run Vite alongside it; its dev server proxies `/api`
through to the one above:

```
cd web && npm install && npm run dev
```

## CLI

```
pip install -r requirements.txt
python translate_epub.py book.epub
```

Reads the key from the same `.env`. To use an environment variable instead:

```
setx GEMINI_API_KEY "your-key-here"     # Windows (new shells)
export GEMINI_API_KEY="your-key-here"   # macOS/Linux
```

Common options:

| Flag | Purpose | Default |
|---|---|---|
| `-o, --output` | Output path | `<input> (Translated).epub` |
| `--source-lang` | Source language | `auto` |
| `--target-lang` | Target language | `English` |
| `--glossary` | Glossary JSON file for consistent terms | none |
| `--model` | Gemini model name | `gemini-2.5-pro` |
| `--requests-per-minute` | Request rate, which is also how many patches run at a time | `4` |
| `--tokens-per-minute` | API token rate limit | `250000` |
| `--tokens-per-request` | Max tokens per batch of chapters | `15000` |

Run `python translate_epub.py --help` for the full list.

## How it works

1. Extracts chapters from the EPUB, skipping any already translated (see below).
2. Groups untranslated chapters into token-limited patches.
3. Translates patches concurrently via Gemini, injecting relevant glossary terms into each prompt.
4. Auto-saves the EPUB after every successful patch. `Ctrl+C` stops gracefully — in-flight patches finish and progress is saved before exit.

### Detecting already-translated chapters

A chapter counts as already translated when almost none of its text is still written in the source language's script. This lets you re-run a partial translation without paying to redo finished chapters.

It only works for languages written in a non-Latin script — Chinese, Japanese, Korean, Russian, Arabic, Hebrew, Greek, Thai, Hindi. Spanish and English share an alphabet, so for Latin-script sources nothing is auto-skipped and every chapter is translated. Over-translating costs tokens; wrongly skipping would silently leave the book half done.

## Glossary

A JSON file mapping original terms to their translation, used to keep names/places consistent:

```json
{"道玄": "Dao Xuan"}
```

Pass it with `--glossary terms.json`, or edit it over the API at `/api/jobs/{id}/glossary` — edits mid-run apply to batches not yet sent. Relevant entries are pulled into each translation prompt automatically.

## Known limitations

- Jobs live in memory. Restarting the server loses whatever was in flight.
- The daily budget counter lives on disk, so it resets whenever the container is rebuilt or restarted.
- Token counts use OpenAI's `cl100k_base` tokenizer, so Gemini token totals and time estimates are approximate.
- The first request after the container has been idle is slow.

## Project layout

```
translate_epub.py    CLI entry point (python translate_epub.py ...)
epub_translate/
  cli.py             argument parsing
  translator.py      translation orchestration, retries, rate limiting
  epub_io.py         EPUB reading/writing, chapter extraction, patching
  glossary.py        glossary loading and term matching
  prompts.py         prompt templates
  rate_limiter.py    API rate limiting
  logging_utils.py   colored console logging
  tokens.py          token counting
server/
  app.py             HTTP endpoints and static file serving
  jobs.py            background translation jobs and progress fan-out
  budget.py          daily spend ceiling
  config.py          settings, all environment-overridable
web/                 React + Vite frontend, builds to web/dist
  src/App.tsx        upload -> plan -> run
  src/useJobStream.ts  progress event stream
  src/styles/        design tokens
  src/components/
```
