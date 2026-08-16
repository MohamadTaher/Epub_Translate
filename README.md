# EPUB Translate

Translates EPUB books using Google's Gemini API. Preserves HTML structure, groups chapters into token-limited patches, and uses a glossary to keep term translations consistent across the book.

Runs two ways: a CLI, and a web app you can deploy.

## Setup

There are two files, split by how often they change.

**`.env` — your API key, and nothing else.** Copy `.env.example` to `.env`:

```
GEMINI_API_KEY=your-key-here
```

Get a key at <https://aistudio.google.com/apikey>. A key is a secret and it never
changes while the process runs, so it is read once at startup. Both the CLI and the
server read this same file. It is gitignored and never copied into the Docker image.

**`settings.env` — everything else, live.** Copy `settings.env.example` to
`settings.env`. The model, the pace, the spend ceilings and the upload limit all live
here, and **the file is re-read whenever it changes** — no restart, and no rebuild:

```
GEMINI_MODEL=gemini-3.5-flash-lite
REQUESTS_PER_MINUTE=4
DAILY_REQUEST_BUDGET=200
```

`gemini-3.1-pro` gives the best translations, but **it has no free tier** — on a key
without billing enabled every request comes back `429`, and a run retries ten times
before failing. `gemini-3.5-flash-lite` is the default for that reason: it works on a
free key, and is far cheaper and faster for testing.

The CLI's `--model` flag overrides this for a single run, and the server reports the
model in use from `GET /api/status`.

Each setting is read at the moment it is used, which is what "live" amounts to in
practice:

| Change | When it applies |
|---|---|
| `DAILY_REQUEST_BUDGET`, `MINUTES_BETWEEN_TRANSLATIONS`, `MAX_UPLOAD_MB`, `DELETE_UPLOADS_AFTER_MINUTES` | the next request |
| `TOKENS_PER_REQUEST` | the next book analysed — it decides how chapters are grouped |
| `GEMINI_MODEL`, `REQUESTS_PER_MINUTE`, `TOKENS_PER_MINUTE` | the next run started; one already going keeps the pace it began with |
| `MAX_TRANSLATIONS_AT_ONCE` | at once when lowered. Raising it accepts more uploads, but the thread pool was sized at startup, so the extra runs queue until a restart |

`settings.env` wins over an environment variable of the same name, which is the
reverse of how `.env` works. An environment variable is how a deployment with no file
to edit gets configured — Cloud Run, where nothing is mounted — and it cannot be
changed without a restart anyway; the file is the one being edited, so it takes
precedence. `settings.env` is gitignored and, like `.env`, never enters the image.

**Neither file ever needs a rebuild.** `.env` needs a restart of the Python service —
the dev server does not read it:

```
docker compose restart epub-translate
```

## Web app

```
docker compose up --build
```

Then open <http://localhost:7860>. Compose mounts your `.env` into the container and
keeps the spend counter in a named volume.

Drop in an EPUB and you get its cover, title and author, then the plan: every chapter
in book order, grouped by the patch that will translate it, with chapters that are
already in the target language marked and left out. Ticking chapters re-costs the run
on the server, so the patch count and the grouping are always the real ones rather
than an estimate. Nothing is spent until you press start.

During a run the book is re-saved after every patch, so the download works part-way
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
| `POST` | `/api/jobs/{id}/cancel` | Stop after in-flight patches finish |
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
  --set-env-vars GEMINI_API_KEY=your-key-here,GEMINI_MODEL=gemini-3.5-flash-lite,DATA_DIR=/tmp/epub_translate
```

The first run offers to enable the Cloud Run, Cloud Build and Artifact Registry APIs; the upload respects `.gitignore`, so `node_modules` and `web/dist` stay out of it.

Cloud Run assumes a service's work happens inside a request, and most of this one's happens after the request has been answered. Four of those flags are what bridge that:

| Flag | Why it is there |
|---|---|
| `--no-cpu-throttling` | Translation runs on a background thread after `/start` has returned. By default Cloud Run takes the CPU away the moment a request finishes, which would leave the run frozen mid-book. |
| `--max-instances 1` | Jobs live in one container's memory, and Cloud Run's session affinity is best-effort. A second instance would answer with jobs it has never heard of. |
| `--timeout 3600` | The progress stream is one long-lived request. The five-minute default would sever it repeatedly; 60 minutes is the most Cloud Run allows. |
| `DATA_DIR=/tmp/epub_translate` | `/tmp` is writable on every Cloud Run instance. The image points this at `/data`, which is where compose mounts a volume instead. |

No port flag is needed: Cloud Run sets `PORT` and the container listens on it, falling back to 7860 everywhere else.

Redeploying is `gcloud run deploy epub-translate --source .` by itself — settings already on the service are kept.

Because the key belongs to whoever deploys it, every visitor spends the owner's money. These limits bound that. They are the contents of `settings.env`, which is not part of the image — a deployment has no file to edit, so it sets them as environment variables instead:

| Variable | Default | Purpose |
|---|---|---|
| `DAILY_REQUEST_BUDGET` | `200` | API requests allowed per day across all visitors |
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

Changing any of them on a deployed service is a `gcloud run services update
--update-env-vars` and a new revision, because there is no `settings.env` up there to
re-read. Locally, the file is the faster path.

### Local development

`docker-compose.override.yml` is merged in automatically and turns `docker compose up`
into a development loop, still on <http://localhost:7860>. It mounts the working tree
over `/app` and runs uvicorn with `--reload`, so the container serves the code in this
directory rather than the copy baked into the image when it was built.

Nothing needs rebuilding to see a change. Python edits restart the server on their own,
and that port belongs to the Vite dev server, which builds the browser app itself and
forwards `/api` to uvicorn over the compose network — so a frontend edit updates the
open page, and a refresh always works. The image only has to be rebuilt when
`requirements.txt` or `package.json` change.

That mount is the one thing a deployed container will not have, and neither is the dev
server — deployed, uvicorn serves the built `web/dist` on that same port. To run what
would actually ship, leave the override out:

```
docker compose -f docker-compose.yml up --build
```

Without Docker at all:

```
pip install -r requirements.txt
uvicorn server.app:app --reload --port 7860
```

That serves the API, plus the last build of the frontend if `web/dist` exists — the
check happens at import, so build before starting.

For the UI with hot reload, run Vite alongside it; its dev server proxies `/api`
through to the one above, and is then the address to open:

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
| `--model` | Gemini model name | `GEMINI_MODEL`, else `gemini-3.5-flash-lite` |
| `--requests-per-minute` | Request rate, which is also how many requests run at a time | `REQUESTS_PER_MINUTE`, else `4` |
| `--tokens-per-minute` | API token rate limit | `TOKENS_PER_MINUTE`, else `250000` |
| `--tokens-per-request` | Max tokens per request | `TOKENS_PER_REQUEST`, else `15000` |

The last four default to the same `.env` settings the server is paced by, so a
book runs at one speed however it is started; passing the flag overrides it.
Run `python translate_epub.py --help` for the full list, which prints the
defaults your `.env` currently resolves to.

## How it works

1. Extracts chapters from the EPUB, skipping any already translated (see below).
2. Groups untranslated chapters into token-limited patches.
3. Translates patches concurrently via Gemini, injecting relevant glossary terms into each prompt and merging in the new terms each response reports.
4. Auto-saves the EPUB after every successful patch. `Ctrl+C` stops gracefully — in-flight patches finish and progress is saved before exit.

### Detecting already-translated chapters

A chapter counts as already translated when almost none of its text is still written in the source language's script. This lets you re-run a partial translation without paying to redo finished chapters.

It only works for languages written in a non-Latin script — Chinese, Japanese, Korean, Russian, Arabic, Hebrew, Greek, Thai, Hindi. Spanish and English share an alphabet, so for Latin-script sources nothing is auto-skipped and every chapter is translated. Over-translating costs tokens; wrongly skipping would silently leave the book half done.

## Glossary

A JSON file mapping original terms to their translation, used to keep names/places consistent:

```json
{"道玄": "Dao Xuan"}
```

Pass it with `--glossary terms.json`, or edit it over the API at `/api/jobs/{id}/glossary` — edits mid-run apply to patches not yet sent. Only the entries a patch actually mentions are pulled into its prompt, so a long glossary doesn't cost tokens on every request.

**It fills itself in as the book is translated.** Each patch asks the model to
report whichever names and terms in those chapters it was not already given, and
they are merged into the glossary and travel back out with the later patches that mention them — which is
what keeps a character called the same thing in chapter 40 as in chapter 1, when
the model only ever sees one patch at a time. The first translation of a term
wins: once settled, a term keeps its translation however the model renders it
later. Terms you supplied yourself are settled from the start, so learning never
overrules you — except that a term you left with a blank translation counts as
one for us to fill in.

Learned terms are written back to the glossary file after each patch, so
`--glossary terms.json` accumulates across runs, and `GET /api/jobs/{id}/glossary`
returns what the run has learned so far. `PUT` replaces the *whole* glossary, so a
client that saves a list fetched before the run would drop what it learned; the web
app re-reads the server's copy inside every save and carries those terms over.

In the browser, the glossary panel appears once a book has been analysed and stays
through the run. **Import a file** to bring in a glossary from elsewhere — the one
the CLI writes, or one saved from an earlier book in the same series. Importing
merges rather than replaces, and what this book has already decided wins, so it can
never rename a character mid-way. **Export** hands the list back as the same kind of
file, which is how a series carries its names from one book to the next.

## Known limitations

- Jobs live in memory. Restarting the server loses whatever was in flight.
- The daily budget counter lives on disk, so it resets whenever the container is rebuilt or restarted.
- Token counts use OpenAI's `cl100k_base` tokenizer, so Gemini token totals and time estimates are approximate.
- The per-minute limits are enforced in this process only. Two runs started from two containers against one key will each keep to the limit and together exceed it.
- The first request after the container has been idle is slow.

## Project layout

```
translate_epub.py    CLI entry point (python translate_epub.py ...)
epub_translate/
  cli.py             argument parsing
  defaults.py        pacing settings and the API key, shared with the server
  translator.py      wiring a run together, pacing it, and saving as it goes
  plan.py            what a run would involve, before anything is spent
  worker.py          one patch, from prompt to translated chapters
  run_state.py       the counters every worker shares
  gemini.py          sending a request, and reading what comes back
  packing.py         grouping chapters into patches, by token budget
  book/              reading an EPUB, its chapters, and writing the translation
  glossary/          the glossary, and the terms each response teaches it
  prompts.py         the prompt sent for one patch
  rate_limiter.py    API rate limiting
  console.py         colored terminal output, and the one prompt the CLI asks
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
