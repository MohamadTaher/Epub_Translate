# EPUB Translate

Gemini-powered EPUB translator. A CLI (`translate_epub.py`) and a FastAPI backend
(`server/`) share the translation core in `epub_translate/`.

`docker compose up` serves the app on [localhost:7860](http://localhost:7860).

## Toolchains run in Docker

**Node, npm and Python are not on the host PATH** — only Docker is. `npm`, `npx`,
`tsc`, `pip` and `uvicorn` fail with "not found"; run them in a container instead.

```
# build / test / lint the frontend
docker run --rm -v "${PWD}\web:/w" -w /w node:20-alpine sh -c "npm install && npm run build"

# Vite dev server with hot reload, proxying to the API container
docker run --rm -p 5173:5173 -v "${PWD}\web:/w" -w /w \
  -e API_TARGET=http://host.docker.internal:7860 \
  node:20-alpine sh -c "npm run dev -- --host"
```

`npm run test` (vitest), `npm run lint` and `npm run typecheck` work the same way.
Node 20 is the ceiling (`Dockerfile`), so don't pin tooling that wants Node 22.

`docker-compose.override.yml` is merged in automatically: it mounts the working tree
over `/app` and runs uvicorn with `--reload`. Python edits restart the server on their
own; a frontend change needs the build above plus a browser refresh. Rebuild the image
only when `requirements.txt` or `package.json` change. `.env` changes need
`docker compose restart` — it is mounted, but read once at startup.

That bind mount is what Cloud Run will not have, so a Dockerfile that stopped copying
something still looks fine locally. Check with
`docker compose -f docker-compose.yml up --build`, which leaves the override out.

## Pacing settings

`epub_translate/defaults.py` is the only module that reads the four pacing settings
(model, requests/min, tokens/min, tokens/request) and loads `.env`. `server/config.py`
re-exports them; `cli.py` uses them as argparse defaults. Don't re-spell the numbers
anywhere else.

`REQUESTS_PER_MINUTE` does two jobs: it caps the rate limiter *and* sizes the worker
pool. Requests take tens of seconds, so a larger pool would only park threads inside
the limiter.

`gemini-3.1-pro` has a **zero** free-tier quota — a key without billing 429s on every
request. `gemini-3.5-flash-lite` is the default in `defaults.py` and what
`.env.example` ships, so a free-tier key works untouched; don't "upgrade" it casually.

A refused request backs off *every* worker (`RateLimiter.back_off`, driven by
`gemini.retry_after`), using Google's stated retry delay when the error body has
one, otherwise doubling from five seconds to a minute.

## Not every failure is worth retrying

`gemini.read_reply` decides that, and it is the only thing standing between a
refusal and ten identical requests. `response.text` is a property that *raises* when
the model returned no usable part, so a safety block cannot be caught with a truth
test — the reason has to be read off `candidates[0].finish_reason`.

`SAFETY`, `PROHIBITED_CONTENT`, `BLOCKLIST`, `SPII`, `RECITATION`, `LANGUAGE` and a
blocked prompt raise `PatchError(retriable=False)`: the same prompt gets the same
answer, so the patch fails at once. `MAX_TOKENS` is fatal too, *including when text
came back* — a truncated reply is the dangerous kind of wrong, plausible and cut off
mid-chapter, and the fix is a smaller `TOKENS_PER_REQUEST`, not another attempt.
Everything else retries.

Both flags live on `RunState` (`run_state.py`), which every worker shares and the
translator exposes as properties, so `translator.should_stop = True` still works from
any thread — that is all `/cancel` does.

`should_stop` is set both by a reader cancelling and by the run giving up after six
failures in a row, so it cannot tell those apart on its own. `aborted_after_failures`
is what separates them, and `server/jobs.py` must check it *first* — otherwise a
collapsed run reports itself as "cancelled".

## Frontend

`web/` is React 19 + Vite + TypeScript with **plain CSS** — CSS Modules, no Tailwind,
no router, no state library, no SSE library. Dependencies are React, Vite, TypeScript
and two self-hosted fonts.

The design is editorial: warm paper, one clay accent, and two typefaces that carry
meaning — **Literata for anything from the book** (title, chapter names, headings),
**Inter for the interface**. Colours live in `web/src/styles/tokens.css` with their
measured contrast ratios; route new ones through those tokens. Light theme only, but
the tokens are structured so a dark one is a second block rather than a rewrite.

- **`EventSource` must be closed in the `end` handler.** The server closes the stream
  when a job finishes, EventSource reconnects on close, and the job is terminal by
  then — it would replay the whole log and send `end` again, forever. See
  `web/src/useJobStream.ts`.
- **The word is "request", not "patch".** The server calls a batch of chapters a
  patch; the reader sees "request", because that is the unit the limits and the daily
  budget are counted in. `ActivityLog.tsx` rewrites the server's wording.
- Build output must land in `web/dist` — the Dockerfile copies it and `STATIC_DIR`
  points there. `server/app.py` mounts it only `if config.STATIC_DIR.exists()`,
  evaluated at import, so a build has to precede the server starting.

## API surface

The API key never leaves the server, so translation cannot be driven from a browser
against Google.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/status` | Config, model, remaining budget, limits, cooldown, languages |
| `POST` | `/api/jobs` | Upload an EPUB (multipart), returns an analysed plan |
| `POST` | `/api/jobs/{id}/preview` | Re-cost a chapter selection; commits nothing |
| `POST` | `/api/jobs/{id}/start` | Confirm the plan and begin; enforces every limit |
| `GET` | `/api/jobs/{id}` | Snapshot, for reconnecting after a dropped stream |
| `GET` | `/api/jobs/{id}/events` | Server-sent events for the life of the run |
| `GET` | `/api/jobs/{id}/cover` | The book's cover art, 404 when it has none |
| `POST` | `/api/jobs/{id}/cancel` | Stop after in-flight batches finish; returns a snapshot |
| `GET` | `/api/jobs/{id}/download` | Translated EPUB; valid mid-run too |
| `GET`/`PUT` | `/api/jobs/{id}/glossary` | Read and replace glossary terms |

`stats` is one `chapters` array in **book order**, skipped chapters left in place.
Each carries `patch` — the 1-based request number the progress events use, `null` when
not being translated — plus its own `tokens` and `skip_reason`. That `patch` number is
the only thing tying a `patch_done` event back to chapters.

Chapters are `Chapter` objects held **by identity**: `Job.stats` keys on
`id(chapter)`, which is why the dataclass is `eq=False`. A job parses its upload once
into a `SourceBook` it keeps — planning, re-planning and the writer all work from that
parse, and `EpubWriter` replaces its documents with translations in place. Extract
chapters before making a writer, and note that `SourceBook.chapters()` returns fresh
objects every call, because a run mutates them.

`/preview` packs the token counts already cached on the plan through the same
`pack_by_tokens` that `/start` uses, so the two always agree. Don't reimplement the
packing rule anywhere else.

Progress arrives only over the SSE stream — a run outlives any single request. The
book is re-saved after every batch, so `/download` returns a valid partial EPUB at any
point. Every connection replays the job's whole event log before going live, so
reconnecting loses nothing but duplicates everything. Rate-limiter waits emit
`rate_limit_wait` / `rate_limit_done`; without them the UI can't tell waiting from
hung.

## The glossary teaches itself

Every response carries a second payload after the translated HTML: `<!-- GLOSSARY -->`
followed by a JSON object of the terms that batch met. Those merge into the glossary
and travel back out with later requests whose chapters mention them — the model only
ever sees one batch, so this is the only thing keeping a character's name steady from
chapter 1 to chapter 40.

- **The marker and its parser are one contract**, so they live in one file
  (`glossary/protocol.py`). Changing the marker in one place alone loses every learned
  term without failing anything. `CHAPTER_SEPARATOR` is in `prompts.py` for the same
  reason: the instruction to preserve it and the string being preserved must move
  together, or a batch silently collapses into its first chapter.
- **`split_translation` runs before the `CHAPTER_SEPARATOR` split.** The other order
  lands the glossary JSON inside the last chapter's HTML.
- **First translation wins.** A settled term keeps its translation however the model
  renders it later; only a blank one gets filled in. `PUT /glossary` replaces
  everything, learned terms included, because that list is the reader's decision — so
  `web/src/useGlossary.ts` re-reads the server's copy in `persist` and carries over
  terms it has never seen. Deleting still works: a term the reader removed was in the
  list they were shown.

## Replies are checked against the chapter count

`prompts.py` states how many chapters are in the batch and how many
`CHAPTER_SEPARATOR` markers must come back; the worker discards a reply whose part
count doesn't match and retries the attempt.

Parts are matched to chapters **by position**, so one missing separator shifts
everything after it — the first chapter gets two chapters merged, the last ones keep
their original language, and the patch records as a success. Losing a request to ask
again is cheaper than a book that is quietly wrong and reports itself finished.

The cost: a batch the model *consistently* miscounts burns all ten retries, and the
daily budget is charged per attempt (`budget.record_request`, called on `patch_start`).
Nothing checks the budget mid-run, so that ceiling is enforced only at job start.

## Layout

```
translate_epub.py    CLI entry point
epub_translate/      translation core, shared by CLI and server
  cli.py             argument parsing
  defaults.py        pacing settings, read from .env once
  translator.py      wiring a run together, pacing it, and saving as it goes
  plan.py            what a run would involve, worked out before anything is spent
  worker.py          one batch of chapters: the prompt, the request, the reply
  run_state.py       the counters every worker shares, and the rule for giving up
  gemini.py          sending a request, and judging whether a failure is worth retrying
  packing.py         grouping chapters into the batches that become requests
  language.py        language codes, and script detection on plain text
  book/
    chapter.py       the Chapter dataclass, mutated in place as it is translated
    reader.py        SourceBook: one parse of the archive, chapters and metadata
    writer.py        EpubWriter, working on the SourceBook's parsed book
  glossary/
    terms.py         the Glossary itself: what is known, and how it grows
    protocol.py      what the model is asked for, and the parser for its reply
    matching.py      which terms are worth sending with a given batch
    storage.py       the file on disk, and the one definition of its format
  prompts.py         the prompt, and CHAPTER_SEPARATOR
  rate_limiter.py    per-minute windows; a slot is reserved, not checked for
  tokens.py          token counting (cl100k_base, approximate for Gemini)
  console.py         colored terminal output, and the CLI's one question
server/
  app.py             HTTP endpoints and static file serving
  jobs.py            background runs, progress fan-out, selection preview
  budget.py          daily spend ceiling
  config.py          settings, all environment-overridable
web/                 React + Vite frontend, builds to web/dist
  src/
    App.tsx          phase machine: upload -> plan -> run
    api.ts           fetch wrappers; ApiError carries the HTTP status
    use*.ts          hooks: job stream, server status, resume, preview, glossary
    format.ts        number and duration wording
    styles/          design tokens and element defaults
    components/      one flat component + CSS Module each
```

## Deployment target is Cloud Run

The container listens on `$PORT`, falling back to 7860, so compose and Cloud Run both
work untouched. `CMD` is shell form on purpose — the variable has to expand at runtime
— with `exec` in front so uvicorn stays PID 1 and gets the SIGTERM Cloud Run sends
before killing an instance. Docker lints shell-form `CMD` as a signal hazard; the
`exec` is the answer, so don't "fix" it to JSON form.

Three deploy flags in `README.md` are load-bearing, all because this app does its work
*after* answering the request:

- `--no-cpu-throttling`, or the background translation thread loses the CPU as soon as
  `/start` returns.
- `--max-instances 1`, because jobs live in one process's memory and session affinity
  is best-effort. Nothing may assume a second instance can serve the same job unless
  job state moves out of memory first.
- `--timeout 3600`, the ceiling on the SSE stream, which is a single long request.
  60 minutes is Cloud Run's maximum, not a choice.

`DATA_DIR` is overridden to `/tmp/epub_translate` there — the Cloud Run filesystem is
in memory, so uploads are charged twice, once as files and once as RAM.
