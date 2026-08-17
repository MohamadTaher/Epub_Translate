# Testing

Three suites that use the app the way a reader does: upload a book, look at what
it would cost, translate it, watch it happen, and open what comes out.

```
docker compose exec -T epub-translate python /app/testing/run_all.py
docker compose exec -T epub-translate python /app/testing/run_all.py --only api
```

Run it inside the container — Python and the dependencies are there, and
`localhost:7860` inside it is uvicorn itself rather than the Vite dev server.

| Suite | Costs | What it covers |
|---|---|---|
| `api` | nothing | status, upload, plan, preview, glossary, and every refusal |
| `writer` | nothing | what a book loses on its way back out, with nothing translated |
| `translation` | ~8 requests | four real runs against Gemini, checked in the downloaded EPUB |

## The files

- `fixtures.py` — the books, built rather than committed. `zh_small` is the
  workhorse: three short Chinese chapters, a cover page with no text in it, and
  names that recur so the glossary has something to learn. `zh_mixed` is half
  already in English, `zh_plain` has no navigation and no metadata at all,
  `styled` is laid out like a book from a shop (stylesheet, image, chapters in
  their own directory), `zh_long` is long enough to cancel, and `padded` is only
  large. Run it on its own to rebuild them: `python testing/fixtures.py`.
- `harness.py` — the API client, the `settings.env` override, the EPUB
  inspections, and the tally that prints the report.
- `test_api.py`, `test_writer.py`, `test_translation.py` — the suites.
- `run_all.py` — runs them, writes `results/<timestamp>-<suite>.json`, and says
  how many requests the run spent.
- `results/` — the reports, and every EPUB a run downloaded, kept for looking at.

## How the suites work

**Settings are changed, not worked around.** A test that needs a smaller token
budget or a lower ceiling edits `settings.env` inside a `with settings(...)`
block and the file is restored afterwards, byte for byte. The server re-reads it
whenever its timestamp or size moves, so an override is live on the next
request — which is itself one of the things being tested.

**The downloaded book is the evidence.** A run that reports success and hands
back a book still in Chinese is the failure worth catching, so nothing is
believed on the strength of the server's own account of it: the EPUB is opened
as a plain zip, its text is counted character by character against the Unicode
ranges restated in `harness.py`, and its package document is read with a regex.
None of that goes through `epub_translate`, which would otherwise be agreeing
with itself.

**One override wraps the whole run.** Every upload leaves a job that counts as
active until it expires, so `MAX_TRANSLATIONS_AT_ONCE` is raised for the length
of the run — otherwise the capacity gate starts refusing uploads part-way
through a suite that makes a dozen of them.

## Two things to know before running it

Editing a `.py` file **on the host** restarts uvicorn, even one outside the
`--reload-dir` list, and jobs live in memory: an edit that lands mid-run makes
the next request answer *"This job has expired or never existed"*. Let an edit
settle before starting a suite.

The cooldown is per visitor address, and `run_all.py` sends
`X-Forwarded-For: 198.51.100.7` for all of it. Pass `--ip` to arrive as someone
else — after a suite has run, that address has a real cooldown record.

## What these caught

All 169 checks pass. Ten of them didn't, the first time they were run, and the
three defects behind them are what the suites are now guarding:

- **The book kept its old language and title.** ebooklib's `set_title` and
  `set_language` *append*, so the package document declared `zh` and then `en`,
  original title first — and a reader shows the first of each. Fixed by
  `_forget_metadata` in `book/writer.py`; watched by `test_writer.py`, which
  asserts the counts as well as the values.
- **Every chapter lost its `<head>`.** ebooklib rebuilds the head from
  `item.title` and `item.links`, neither of which is filled in when a book is
  read, so the stylesheet link and the `<title>` were dropped from every
  chapter — translated or not. The CSS stayed in the archive with nothing
  pointing at it and the book rendered unstyled. Fixed by
  `_keep_document_heads`. The `styled` fixture exists for this one, and the
  round-trip that finds it costs nothing: no chapter has to be translated for a
  book to come back damaged.
- **A download taken mid-run could be a broken file.** The book is rewritten
  after every patch while `/download` serves that same path. Two races, one
  behind the other: the save wrote in place (fixed with a rename), and
  `FileResponse` measures a file and then opens it (fixed by reading the bytes
  in the endpoint). `test_translation.py` samples the download twelve times
  around every `patch_done` — it caught 1 to 3 of 36 per run, and now catches
  none.
