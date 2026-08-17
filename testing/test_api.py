"""
Everything that can be checked without spending a request.

Uploading, planning, re-costing a selection, the glossary editor, and every way
the server can refuse — all of it happens before the API key is ever used, so
this suite is free and can be run as often as you like. The half that costs
money is in `test_translation.py`.

Each test uploads its own book. Uploads are free, and a job that has been
prepared but not started keeps its plan for an hour, so sharing one between
tests would only make a failure harder to read.
"""

from harness import Api, Report, patches_by_rule, settings


def test_status(api: Api, report: Report, books):
    report.section("Server status")

    response = api.status()
    if not report.check("GET /api/status answers 200", response.status_code == 200,
                        f"got {response.status_code}"):
        return

    status = response.json()
    report.check("an API key is configured", status['configured'] is True)
    report.check("a model is named", bool(status['model']), status['model'])
    report.check("today's budget has room left", status['remaining_requests'] > 0,
                 f"{status['remaining_requests']} of {status['daily_budget']} left")
    report.check("the pace is a positive number of requests/min",
                 status['requests_per_minute'] >= 1, str(status['requests_per_minute']))
    report.check("every language the translator knows is offered",
                 len(status['languages']) == 15, f"{len(status['languages'])} languages")
    report.check("the languages a script check can detect are listed apart",
                 set(status['detectable_languages']) < set(status['languages']),
                 f"{len(status['detectable_languages'])} detectable")
    report.check("the upload limit is published", status['max_upload_mb'] > 0,
                 f"{status['max_upload_mb']} MB")


def test_settings_are_live(api: Api, report: Report, books):
    report.section("settings.env takes effect without a restart")

    before = api.status().json()

    with settings(MAX_UPLOAD_MB=3, DAILY_REQUEST_BUDGET=12345):
        during = api.status().json()
        report.check("an edited upload limit is live on the next request",
                     during['max_upload_mb'] == 3, f"got {during['max_upload_mb']}")
        report.check("an edited daily budget is live on the next request",
                     during['daily_budget'] == 12345, f"got {during['daily_budget']}")

    after = api.status().json()
    report.check("putting the file back puts the settings back",
                 after['max_upload_mb'] == before['max_upload_mb']
                 and after['daily_budget'] == before['daily_budget'],
                 f"{after['max_upload_mb']} MB, budget {after['daily_budget']}")


def test_upload_and_plan(api: Api, report: Report, books):
    report.section("Uploading a book, and what the plan says about it")

    with settings(TOKENS_PER_REQUEST=1500):
        response = api.upload(books['zh_small'])
        if not report.check("POST /api/jobs answers 200", response.status_code == 200,
                            response.text[:200]):
            return

        job = response.json()
        report.check("the job is ready to start", job['status'] == "ready", job['status'])
        report.check("nothing has been translated yet", job['completed'] == 0)

        stats = job['stats']
        book = stats['book']
        report.check("the book's own title is read out of it",
                     book['title'] == "青石镇的铁匠", str(book['title']))
        report.check("so is its author", book['author'] == "佚名", str(book['author']))
        report.check("a cover is noticed", book['has_cover'] is True)
        report.check("the uploaded filename is kept", book['filename'] == "zh_small.epub",
                     str(book['filename']))

        report.check("the three chapters are found, and the text-free cover page is not",
                     stats['chapter_count'] == 3, f"{stats['chapter_count']} chapters")
        report.check("the source language is detected from the script",
                     stats['source_language'] == "chinese", stats['source_language'])
        report.check("the target language is the one asked for",
                     stats['target_language'] == "English", stats['target_language'])

        chapters = stats['chapters']
        report.check("every chapter is measured", all(c['tokens'] > 0 for c in chapters),
                     str([c['tokens'] for c in chapters]))
        report.check("every chapter is in a patch",
                     all(c['patch'] is not None and c['skip_reason'] is None for c in chapters))
        report.check("patches are numbered from 1 without a gap",
                     sorted({c['patch'] for c in chapters}) == list(range(1, stats['patch_count'] + 1)),
                     f"{stats['patch_count']} patches")
        report.check("the token total is the sum of the chapters in the plan",
                     stats['total_tokens'] == sum(c['tokens'] for c in chapters))

        expected = patches_by_rule([c['tokens'] for c in chapters], 1500)
        report.check("the chapters are grouped the way the packing rule says",
                     stats['patch_count'] == len(expected),
                     f"{stats['patch_count']} patches, the rule says {len(expected)}")

    # The same book, planned again with more room per request: fewer, larger
    # patches, and no restart in between.
    with settings(TOKENS_PER_REQUEST=6000):
        roomier = api.upload(books['zh_small']).json()['stats']
        report.check("a larger token budget means fewer requests for the same book",
                     roomier['patch_count'] < stats['patch_count'],
                     f"{roomier['patch_count']} at 6000 tokens vs {stats['patch_count']} at 1500")


def test_cover(api: Api, report: Report, books):
    report.section("Cover art")

    with_cover = api.upload(books['zh_small']).json()
    response = api.cover(with_cover['id'])
    report.check("a book's cover is served back", response.status_code == 200,
                 f"got {response.status_code}")
    report.check("it is served as an image",
                 response.headers.get("content-type", "").startswith("image/"),
                 response.headers.get("content-type", ""))
    report.check("the image has content", len(response.content) > 0,
                 f"{len(response.content)} bytes")

    without = api.upload(books['zh_mixed']).json()
    report.check("a book with no cover says so with a 404",
                 api.cover(without['id']).status_code == 404)


def test_already_translated_chapters(api: Api, report: Report, books):
    report.section("Chapters that are already in the target language")

    stats = api.upload(books['zh_mixed']).json()['stats']
    report.check("all four chapters are listed", stats['chapter_count'] == 4,
                 f"{stats['chapter_count']}")
    report.check("the book is still recognised as Chinese",
                 stats['source_language'] == "chinese", stats['source_language'])

    planned = [c for c in stats['chapters'] if c['patch'] is not None]
    skipped = [c for c in stats['chapters'] if c['patch'] is None]

    report.check("the two Chinese chapters are planned", len(planned) == 2,
                 str([c['title'] for c in planned]))
    report.check("the two English ones are left out", len(skipped) == 2,
                 str([c['title'] for c in skipped]))
    report.check("and each says why",
                 all(c['skip_reason'] == "No Chinese characters detected" for c in skipped),
                 str([c['skip_reason'] for c in skipped]))
    report.check("skipped chapters keep their place in the book",
                 [c['title'] for c in stats['chapters']][1] == skipped[0]['title'],
                 "the second chapter of the book is the second in the list")


def test_preview(api: Api, report: Report, books):
    report.section("Re-costing a selection without committing to it")

    with settings(TOKENS_PER_REQUEST=1500):
        job = api.upload(books['zh_small']).json()
        job_id, planned = job['id'], job['stats']['patch_count']
        chapters = job['stats']['chapters']

        response = api.preview(job_id, [chapters[1]['id']])
        if not report.check("POST /preview answers 200", response.status_code == 200,
                            response.text[:200]):
            return

        preview = response.json()
        report.check("one chapter costs one request", preview['patch_count'] == 1,
                     f"{preview['patch_count']}")
        report.check("the chapter picked is the one in the patch",
                     [c['patch'] for c in preview['chapters']] == [None, 1, None],
                     str([c['patch'] for c in preview['chapters']]))
        report.check("the rest say they were not selected",
                     preview['chapters'][0]['skip_reason'] == "Not selected",
                     str(preview['chapters'][0]['skip_reason']))
        report.check("the whole book is still listed, not only the selection",
                     preview['chapter_count'] == 3, f"{preview['chapter_count']}")
        report.check("only the selection is costed",
                     preview['total_tokens'] == chapters[1]['tokens'],
                     f"{preview['total_tokens']} vs {chapters[1]['tokens']}")

        report.check("previewing changes nothing about the job",
                     api.job(job_id).json()['stats']['patch_count'] == planned,
                     f"still {planned} patches")

        empty = api.preview(job_id, []).json()
        report.check("selecting nothing costs nothing", empty['patch_count'] == 0)

        default = api.preview(job_id, None).json()
        report.check("no selection at all means the plan as uploaded",
                     default['patch_count'] == planned)

        mixed = api.upload(books['zh_mixed']).json()
        translated = [c for c in mixed['stats']['chapters'] if c['patch'] is None][0]
        reselected = api.preview(mixed['id'], [translated['id']]).json()
        report.check("a chapter the server skipped can still be chosen deliberately",
                     reselected['patch_count'] == 1, f"{reselected['patch_count']}")


def test_glossary_editor(api: Api, report: Report, books):
    report.section("The glossary editor")

    job_id = api.upload(books['zh_small']).json()['id']
    other_id = api.upload(books['zh_small']).json()['id']

    report.check("a new job starts with an empty glossary",
                 api.get_glossary(job_id).json() == {'terms': {}},
                 str(api.get_glossary(job_id).json()))

    terms = {"李明": "Li Ming", "寒霜剑": "Frost Sword", "青石镇": "Greenstone Town"}
    response = api.put_glossary(job_id, terms)
    report.check("PUT answers with what it stored", response.status_code == 200
                 and response.json()['terms'] == terms, response.text[:200])
    report.check("and reading it back gives the same terms, unicode intact",
                 api.get_glossary(job_id).json()['terms'] == terms)

    report.check("another job's glossary is untouched",
                 api.get_glossary(other_id).json()['terms'] == {})

    api.put_glossary(job_id, {"李明": "Li Ming"})
    report.check("PUT replaces the list rather than merging into it, so deleting works",
                 api.get_glossary(job_id).json()['terms'] == {"李明": "Li Ming"},
                 str(api.get_glossary(job_id).json()['terms']))

    api.put_glossary(job_id, {"  王秀兰  ": "  Wang Xiulan  ", "": "dropped"})
    stored = api.get_glossary(job_id).json()['terms']
    report.check("stray whitespace is trimmed and a blank term is dropped",
                 stored == {"王秀兰": "Wang Xiulan"}, str(stored))

    report.check("a translation that is not text is refused",
                 api.put_glossary(job_id, {"李明": 5}).status_code == 422)


def test_book_without_metadata(api: Api, report: Report, books):
    report.section("A book with no title, no author, no cover and no navigation")

    job = api.upload(books['zh_plain']).json()
    stats = job['stats']

    report.check("it still uploads and plans", job['status'] == "ready", job['status'])
    report.check("its one chapter is found", stats['chapter_count'] == 1,
                 f"{stats['chapter_count']}")
    report.check("no title is reported rather than a made-up one",
                 not stats['book']['title'], str(stats['book']['title']))
    report.check("no author either", stats['book']['author'] is None,
                 str(stats['book']['author']))
    report.check("and no cover", stats['book']['has_cover'] is False)
    report.check("asking for the cover anyway is a 404",
                 api.cover(job['id']).status_code == 404)


def test_refusals(api: Api, report: Report, books):
    report.section("Every way an upload or a start is refused")

    wrong_type = api.upload(books['not_an_epub'], filename="notes.txt")
    report.check("a file that is not an .epub is refused", wrong_type.status_code == 400,
                 f"got {wrong_type.status_code}")
    report.check("and the reason says what to upload",
                 "epub" in wrong_type.json()['detail'].lower(), wrong_type.json()['detail'])

    corrupt = api.upload(books['corrupt'])
    report.check("an unreadable archive is refused", corrupt.status_code == 400,
                 f"got {corrupt.status_code}")
    report.check("and the reason names the book, not the traceback",
                 corrupt.json()['detail'].startswith("Could not read this EPUB"),
                 corrupt.json()['detail'][:120])

    with settings(MAX_UPLOAD_MB=1):
        oversized = api.upload(books['padded'])
        report.check("a book over the upload limit is refused", oversized.status_code == 413,
                     f"got {oversized.status_code}")
        report.check("and the reason quotes the limit in force",
                     "1 MB" in oversized.json()['detail'], oversized.json()['detail'])

    report.check("the same book uploads once the limit is back",
                 api.upload(books['padded']).status_code == 200)

    missing = "0" * 32
    report.check("an unknown job is a 404 everywhere it can be asked for",
                 [api.job(missing).status_code, api.start(missing).status_code,
                  api.cancel(missing).status_code, api.download(missing).status_code,
                  api.cover(missing).status_code, api.get_glossary(missing).status_code]
                 == [404] * 6)

    job_id = api.upload(books['zh_small']).json()['id']

    report.check("downloading before anything is translated is a 404",
                 api.download(job_id).status_code == 404)
    report.check("and it says nothing has been translated yet",
                 "Nothing has been translated" in api.download(job_id).json()['detail'],
                 api.download(job_id).json()['detail'])

    empty_start = api.start(job_id, [])
    report.check("starting with no chapters selected is refused",
                 empty_start.status_code == 400, f"got {empty_start.status_code}")
    report.check("and says so plainly",
                 "No chapters selected" in empty_start.json()['detail'],
                 empty_start.json()['detail'])

    unknown_chapters = api.start(job_id, ["no-such-chapter"])
    report.check("so is starting with chapter ids the book doesn't have",
                 unknown_chapters.status_code == 400, f"got {unknown_chapters.status_code}")

    cancelled = api.cancel(job_id)
    report.check("cancelling a job that never started is harmless",
                 cancelled.status_code == 200 and cancelled.json()['status'] == "ready",
                 f"{cancelled.status_code}, status {cancelled.json().get('status')}")


def test_budget_ceiling(api: Api, report: Report, books):
    report.section("The daily spend ceiling")

    status = api.status().json()
    used_today = status['daily_budget'] - status['remaining_requests']

    with settings(TOKENS_PER_REQUEST=1500, DAILY_REQUEST_BUDGET=used_today + 1):
        job = api.upload(books['zh_small']).json()
        needed = job['stats']['patch_count']

        lowered = api.status().json()
        report.check("the remaining budget follows the ceiling down",
                     lowered['remaining_requests'] == 1,
                     f"{lowered['remaining_requests']} left of {lowered['daily_budget']}")

        response = api.start(job['id'], None)
        report.check("a run that would overspend the day is refused",
                     response.status_code == 429, f"got {response.status_code}")
        report.check("and the refusal quotes both numbers",
                     "1 requests are left" in response.json()['detail']
                     and str(needed) in response.json()['detail'],
                     response.json()['detail'])
        report.check("the job is left ready rather than half-started",
                     api.job(job['id']).json()['status'] == "ready")

    after = api.status().json()
    report.check("nothing was spent by being refused",
                 after['remaining_requests'] == status['remaining_requests'],
                 f"{after['remaining_requests']} left, was {status['remaining_requests']}")


def test_capacity_gate(api: Api, report: Report, books):
    report.section("The limit on translations at once")

    api.upload(books['zh_small'])       # one job in hand, so the server is at capacity

    with settings(MAX_TRANSLATIONS_AT_ONCE=1):
        status = api.status().json()
        report.check("the server reports itself busy", status['busy'] is True)

        refused = api.upload(books['zh_small'])
        report.check("a further upload is refused while it is",
                     refused.status_code == 429, f"got {refused.status_code}")
        report.check("and says the server is busy",
                     "busy" in refused.json()['detail'].lower(), refused.json()['detail'])

    report.check("raising the limit lets uploads through again",
                 api.upload(books['zh_small']).status_code == 200)
    report.note("a job that is uploaded and never started still counts as active",
                "abandoned uploads occupy the capacity gate until DELETE_UPLOADS_AFTER_MINUTES")


TESTS = [
    test_status,
    test_settings_are_live,
    test_upload_and_plan,
    test_cover,
    test_already_translated_chapters,
    test_preview,
    test_glossary_editor,
    test_book_without_metadata,
    test_refusals,
    test_budget_ceiling,
    test_capacity_gate,
]
