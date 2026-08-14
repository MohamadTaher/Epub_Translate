---
title: EPUB Translate
emoji: 📖
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

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

### Deploying to Hugging Face Spaces

Create a **Docker** Space and push this repo. Set `GEMINI_API_KEY` under *Settings → Secrets* (a secret, not a variable, so it stays out of build logs) — deployed, the key comes from there rather than from `.env`, which is never pushed. The frontmatter at the top of this file tells Spaces which port to serve.

Because the key belongs to whoever deploys it, every visitor spends the owner's money. These limits bound that, and all are settable as environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `DAILY_REQUEST_BUDGET` | `200` | API requests allowed per day across all visitors |
| `MAX_PATCHES_PER_JOB` | `25` | Caps the size of a single translation |
| `IP_COOLDOWN_MINUTES` | `30` | Minimum wait between runs from one address |
| `MAX_UPLOAD_MB` | `25` | Largest accepted EPUB |
| `MAX_ACTIVE_JOBS` | `2` | Translations running at once |
| `JOB_TTL_MINUTES` | `60` | How long an uploaded book is kept before deletion |

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
| `--max-concurrent` | Patches translated in parallel | `5` |
| `--max-requests-per-minute` | API request rate limit | `4` |
| `--max-tokens-per-minute` | API token rate limit | `250000` |
| `--max-tokens-per-patch` | Max tokens per batch of chapters | `15000` |

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

- Jobs live in memory. Restarting the server, or a Space going to sleep, loses whatever was in flight.
- The daily budget counter lives on disk, so it resets if the Space is rebuilt (free Spaces have no persistent storage).
- Token counts use OpenAI's `cl100k_base` tokenizer, so Gemini token totals and time estimates are approximate.
- The first request after a Space wakes from sleep is slow.

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
