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
```

`npm run test` (vitest), `npm run lint` and `npm run typecheck` work the same way.
Node 20 is the ceiling (`Dockerfile`), so don't pin tooling that wants Node 22.

## One port, and nothing to rebuild

`docker-compose.override.yml` is merged in automatically, and the app is on
<http://localhost:7860> whether or not it is. It mounts the working tree over `/app`
and runs uvicorn with `--reload`, so Python edits restart the server on their own.

In development that published port is the **`web` service**, a Vite dev server which
forwards `/api` to `epub-translate:7860` over the compose network — so a frontend edit
needs no `npm run build`: the open page updates itself, and a refresh always works.
Deployed, the same port is uvicorn serving the `web/dist` the Dockerfile built. Two
things hold that together and both look removable:

- `ports: !reset []` on `epub-translate` in the override. Compose *appends* `ports`
  when merging, so without it both services try to publish 7860 and neither starts.
- Vite is published `7860:7860`, the same number inside and out, because the page
  opens a hot-reload socket back to the port it was served from.

`VITE_POLL=1` makes that watcher poll (`web/vite.config.ts`). A bind-mounted Windows
directory delivers no file events into a Linux container, so without it nothing is
ever seen to change and the page silently serves stale modules.

Rebuild the image only when `requirements.txt` or `package.json` change. `.env`
changes need `docker compose restart` — it is mounted, but read once at startup.

That bind mount is what Cloud Run will not have, so a Dockerfile that stopped copying
something still looks fine locally. Check with
`docker compose -f docker-compose.yml up --build`, which leaves the override out.

## Two settings files, and only one of them is a constant

`.env` holds `GEMINI_API_KEY` and nothing else. `settings.env` holds everything a
running server might want retuned — model, pace, spend ceilings, upload limits — and
`epub_translate/settings_file.py` re-reads it whenever its mtime or size moves.

**Settings are not constants any more.** `defaults.py` and `server/config.py` both
serve theirs from a module-level `__getattr__` (PEP 562), so `config.TOKENS_PER_REQUEST`
still reads like an attribute at every call site but is evaluated *then*. Two things
follow, and both fail silently:

- **Read late.** `x = config.MAX_UPLOAD_MB` at module scope freezes the value at
  import and quietly opts that setting out of being live. Read it where it is used.
- **Never assign one of those names** in either module. A real attribute shadows
  `__getattr__`, so the setting keeps working — frozen — with nothing to show for it.

`defaults.py` still owns the four pacing settings and their defaults, and is still the
only module that calls `load_dotenv`; `settings_file.py` knows about the file, not
about which settings exist. `server/config.py` names its own limits and forwards the
pacing ones. `cli.py` uses them as argparse defaults, which is a read at import — fine
there, because the process is over in one run.

Don't reach for `os.environ["GEMINI_API_KEY"]` directly: it is only populated once
`defaults` has been imported, so that works by import order rather than by design.

Precedence is `settings.env` → environment → built-in default, which is the reverse of
`.env`. The file has to win or editing it would do nothing whenever a stale copy of the
same key was left in `.env`. The Dockerfile copies `settings.env` in, so that holds
when deployed too: an environment variable set on a Cloud Run service, for a name the
file mentions, is silently ignored. Retuning a deployed server is an edit to the file
and a push, never a console change.

`MAX_TRANSLATIONS_AT_ONCE` is the one setting that is only half live: the admission
gate in `app.py` reads it per request, but it also sized the `ThreadPoolExecutor` in
`jobs.py` at import, so raising it queues the extra runs until a restart.

`REQUESTS_PER_MINUTE` does two jobs: it caps the rate limiter *and* sizes the worker
pool. Requests take tens of seconds, so a larger pool would only park threads inside
the limiter.

`gemini-3.1-pro` has a **zero** free-tier quota — a key without billing 429s on every
request. `gemini-3.5-flash-lite` is the default in `defaults.py` and what
`settings.env.example` ships, so a free-tier key works untouched; don't "upgrade" it
casually.

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

## A retry says what the last attempt got wrong

`PatchError.retriable` decides whether to ask again; `PatchError.correction` decides
what to say when asking. A correction is rendered into the next attempt's prompt as
`# YOUR LAST ATTEMPT WAS REJECTED`, last of the instructions and directly above the
input — the failures worth retrying at all are otherwise a closed loop, since the same
prompt earns the same reply until the ten attempts are gone.

**Only failures the model itself caused carry one.** A 429, a timeout or a dropped
connection never reached the model, and accusing it of a reply it never sent puts a
falsehood in the prompt and pays for it on every retry. Transport errors aren't
`PatchError` at all, so `getattr(error, 'correction', None)` leaves them silent — but
don't add corrections to `retry_after`'s cases by hand either.

A correction is replaced only by a later failure carrying its own, never cleared: a
429 on attempt two says nothing about the malformed reply from attempt one, which the
model still has to be told about on attempt three.

This is the one thing a retry changes. `worker._build_request` still settles the
chapters and the glossary once, so the tenth attempt asks with the terms the first one
had rather than whatever other patches have since learned — and the token estimate is
not recomputed, because `_PROMPT_OVERHEAD_TOKENS` already reserves a thousand tokens
and the longest correction is a fraction of that.

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
- **A group of chapters is a "patch", on both sides and on screen.** Identifiers,
  event names (`patch_start`, `patch_done`, …), `stats.chapters[].patch`, the log
  lines, `Patch 3 of 6` in the chapter list — all the same word. `ActivityLog.tsx`
  shows the server's message as it arrives; the four regexes that used to rewrite
  "patch" into "request" are gone, and nothing should replace them.
- **"Request" means the API call**, and only that: `4 requests/min`, `12 requests
  left in today's budget`, "the API rejected every request". It is not a synonym
  for patch — the budget is charged per *attempt*, so a patch that retries three
  times spends three requests. It is also what `request` means throughout
  `server/` (FastAPI's `Request`), which is why it can't name the unit of work.
  "Batch" is a third word for the same thing; don't.
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
| `POST` | `/api/jobs/{id}/cancel` | Stop after in-flight patches finish; returns a snapshot |
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

`EpubWriter` adds an NCX and a nav document **only when the book has neither**
(`_ensure_navigation`). Adding them unconditionally is the obvious thing and it
produces a duplicate manifest id and a duplicate zip member on every save, which
`zipfile` warns about and strict readers reject. Most EPUB3 books arrive with both.

Two more things ebooklib does that the writer has to undo, both of which look like
setup nobody needs:

- **`set_title` and `set_language` add rather than replace.** Without the
  `_forget_metadata` calls in front of them the package document declares the
  original title and language *first*, and a reader shows the first — so a finished
  translation presents itself as the book it was translated from.
- **ebooklib does not write the document it was given; it rebuilds one**, and that
  rebuild's `<head>` holds `item.title` and `item.links` and nothing else. Reading a
  book fills in neither, so `_keep_document_heads` copies each head across before the
  first chapter is replaced. Drop it and every chapter loses its `<title>` and its
  `<link>` to the book's stylesheet, which stays in the archive with nothing pointing
  at it — the translation renders unstyled, and nothing fails.

The save itself goes to a temporary file and is **renamed into place**
(`_write_in_one_move`), because `/download` serves that same path while the run
writing it is still going.

`/preview` packs the token counts already cached on the plan through the same
`pack_by_tokens` that `/start` uses, so the two always agree. Don't reimplement the
packing rule anywhere else. It renders through `Job.stats(plan)`, passing its own
plan — a preview must never assign `job.plan`, even briefly: it used to swap the
plan in and restore it, and a preview landing while `/start` was re-planning put the
old plan back over the one the run then executed.

Progress arrives only over the SSE stream — a run outlives any single request. The
book is re-saved after every patch, so `/download` returns a valid partial EPUB at any
point — which is why it reads the file itself rather than answering with a
`FileResponse`. That measures the file and then opens it, and a save landing between
the two sends a length the file it opens no longer has, so the download dies part-way
through. Every connection replays the job's whole event log before going live, so
reconnecting loses nothing but duplicates everything. Rate-limiter waits emit
`rate_limit_wait` / `rate_limit_done`; without them the UI can't tell waiting from
hung.

## The glossary teaches itself

Every response carries a second payload after the translated HTML: `<!-- GLOSSARY -->`
followed by a JSON object of the terms that patch met. Those merge into the glossary
and travel back out with later requests whose chapters mention them — the model only
ever sees one patch, so this is the only thing keeping a character's name steady from
chapter 1 to chapter 40.

- **The marker and its parser are one contract**, so they live in one file
  (`glossary/protocol.py`). Changing the marker in one place alone loses every learned
  term without failing anything. `CHAPTER_SEPARATOR`, the instruction to preserve it
  and `split_chapters` are in `prompts.py` for the same reason: the string asked for
  and what counts as it coming back must move together, or a patch silently collapses
  into its first chapter.
- **`split_translation` runs before the `CHAPTER_SEPARATOR` split.** The other order
  lands the glossary JSON inside the last chapter's HTML.
- **First translation wins.** A settled term keeps its translation however the model
  renders it later; only a blank one gets filled in. `PUT /glossary` replaces
  everything, learned terms included, because that list is the reader's decision — so
  `web/src/useGlossary.ts` re-reads the server's copy in `persist` and carries over
  terms it has never seen. Deleting still works: a term the reader removed was in the
  list they were shown.

## Replies are checked against the chapter count

`prompts.py` states how many chapters are in the patch and how many
`CHAPTER_SEPARATOR` markers must come back; the worker discards a reply whose part
count doesn't match (`patch_misaligned`) and retries the attempt.

Parts are matched to chapters **by position**, so one missing separator shifts
everything after it — the first chapter gets two chapters merged, the last ones keep
their original language, and the patch records as a success. Losing a request to ask
again is cheaper than a book that is quietly wrong and reports itself finished.

**An empty part is rejected the same way** (`patch_incomplete`). It is the same
failure arriving with the right count: `_apply` steps over a blank rather than wiping
a chapter, so that chapter stays in its original language inside a patch that reports
success — which is the one outcome all of this exists to prevent.

The split itself is deliberately loose about spelling — `prompts.py:split_chapters`
matches case-insensitively and takes the underscore as a space or a hyphen. The count
check is the real guard, and a model that tidies `<!-- CHAPTER_SEPARATOR -->` into
`<!--CHAPTER SEPARATOR-->` meant it as the marker; rejecting that spends a patch's
whole ten attempts on punctuation. The prompt still asks for it exactly.

**A marker that comes back translated is past saving**, because nothing is left to
match — and it costs the whole patch, not one attempt, since the same prompt earns the
same answer ten times. So the prompt forbids it twice: `# THE HTML` exempts comments
from translation as a class, and `_reply_shape` names the marker a literal to copy in
Latin letters. That wording is load-bearing — the surrounding rules read as an
instruction to translate it, since an HTML comment is neither a tag nor an attribute
("reproduce every tag exactly" misses it) and its contents *are* text.

The cost: a patch the model *consistently* miscounts burns all ten retries, and the
daily budget is charged per attempt (`budget.record_request`, called on `patch_start`).
Nothing checks the budget mid-run, so that ceiling is enforced only at job start.

## Layout

```
translate_epub.py    CLI entry point
epub_translate/      translation core, shared by CLI and server
  cli.py             argument parsing
  defaults.py        the pacing settings, and the API key read from .env once
  settings_file.py   settings.env, re-read whenever it changes
  translator.py      wiring a run together, pacing it, and saving as it goes
  plan.py            what a run would involve, worked out before anything is spent
  worker.py          one patch: its prompt, its reply, and what becomes of it
  run_state.py       the counters every worker shares, and the rule for giving up
  gemini.py          sending a request, and judging whether a failure is worth retrying
  packing.py         grouping chapters into patches, by token budget
  language.py        language codes, and script detection on plain text
  book/
    chapter.py       the Chapter dataclass, mutated in place as it is translated
    reader.py        SourceBook: one parse of the archive, chapters and metadata
    writer.py        EpubWriter, working on the SourceBook's parsed book
  glossary/
    terms.py         the Glossary itself: what is known, and how it grows
    protocol.py      what the model is asked for, and the parser for its reply
    matching.py      which terms are worth sending with a given patch
    storage.py       the file on disk, and the one definition of its format
  prompts.py         the prompt, CHAPTER_SEPARATOR, and the split that reads it back
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

Deploys are continuous from GitHub: a merge into `main` builds and rolls out a new
revision, so `main` is the live demo. The build must stay on the **Dockerfile**, not
buildpacks — buildpacks read `requirements.txt`, build the Python app alone and skip
`web/` entirely, which yields a green build serving an API with no frontend.

Three service settings listed in `README.md` are load-bearing, all because this app
does its work *after* answering the request:

- **CPU always allocated**, or the background translation thread loses the CPU as soon
  as `/start` returns.
- **Maximum instances of 1**, because jobs live in one process's memory and session
  affinity is best-effort. Nothing may assume a second instance can serve the same job
  unless job state moves out of memory first.
- **A 3600-second request timeout**, the ceiling on the SSE stream, which is a single
  long request. 60 minutes is Cloud Run's maximum, not a choice.

They live on the service, not in the repo, so a new revision inherits them and nothing
in a commit can set them.

`DATA_DIR` is overridden to `/tmp/epub_translate` there — the Cloud Run filesystem is
in memory, so uploads are charged twice, once as files and once as RAM.
