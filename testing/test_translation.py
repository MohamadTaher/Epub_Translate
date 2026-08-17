"""
The half that calls Gemini, and therefore costs something.

Four runs, kept as small as they can be while still proving what they are here
for: a whole book end to end, a single chapter chosen out of one, a run stopped
in the middle, and a book with no navigation translated into a right-to-left
language. Roughly eight requests in total, plus whatever retries the model
provokes.

Everything is checked against the downloaded EPUB rather than against the
server's own account of what it did — a run that reports success and hands back
a book still in Chinese is the failure worth catching.
"""

import time
from pathlib import Path

import requests

from harness import (Api, Report, RESULTS_DIR, chapter_texts, download_filename,
                     duplicate_members, epub_members, is_epub, opf_metadata,
                     script_ratio, settings)

# The book the first run produces, kept for the test that re-uploads a finished
# translation. Sequential by nature: that test has nothing to say on its own.
ARTIFACTS = {}


def _save(response, name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    path.write_bytes(response.content)
    return path


def _sample_download(api: Api, job_id: str) -> dict:
    """
    One download, and whether a reader would have got a book out of it.

    A transfer that dies part-way counts as a failed download and not as an
    error in the test: the server sends the length it saw when it opened the
    file, so a file that shrinks underneath it ends the response early rather
    than refusing it.
    """
    try:
        response = api.download(job_id)
    except requests.RequestException as error:
        return {'served': True, 'readable': False, 'content': b"",
                'why': f"{type(error).__name__}"}

    if response.status_code != 200:
        return {'served': False, 'readable': False, 'content': b"",
                'why': f"HTTP {response.status_code}"}

    readable = is_epub(response.content)
    return {'served': True, 'readable': readable, 'content': response.content,
            'why': "" if readable else f"{len(response.content)} bytes, not an archive"}


def _by_file_name(texts, file_name: str) -> str:
    """The text of one chapter, whatever directory the writer stored it under."""
    for name, text in texts.items():
        if Path(name).name == Path(file_name).name:
            return text
    return ""


def test_full_run(api: Api, report: Report, books):
    report.section("A whole book, uploaded and translated")

    with settings(TOKENS_PER_REQUEST=1500, REQUESTS_PER_MINUTE=3):
        job = api.upload(books['zh_small']).json()
        job_id = job['id']
        planned = job['stats']['patch_count']
        chapters = job['stats']['chapters']
        report.note("plan", f"{len(chapters)} chapters in {planned} patches, "
                            f"{job['stats']['total_tokens']} tokens")

        # A name the reader has already decided on. It must survive the run
        # untouched, however the model would have rendered it.
        api.put_glossary(job_id, {"李明": "Bright Li"})

        response = api.start(job_id, None)
        if not report.check("POST /start answers 200", response.status_code == 200,
                            response.text[:200]):
            return
        report.check("the job reports itself running", response.json()['status'] == "running",
                     response.json()['status'])

        # Sampled from inside the stream, so "mid-run" means the moment after a
        # patch landed rather than a guessed sleep — and sampled hard, because
        # the book is rewritten right after that event and a download taken
        # while it is being written is the thing worth catching.
        midrun = {'downloads': [], 'snapshot': None}

        def watch(event):
            if event.get('event') != "patch_done":
                return
            if midrun['snapshot'] is None:
                midrun['snapshot'] = api.job(job_id).json()
            for _ in range(12):
                midrun['downloads'].append(_sample_download(api, job_id))

        events, final = api.stream(job_id, watch, timeout=900)

    if not report.check("the stream ended with a final snapshot", final is not None):
        return

    # ------------------------------------------------------------- the events
    kinds = [event.get('event') for event in events]
    starts = [e for e in events if e.get('event') == "patch_start"]
    dones = [e for e in events if e.get('event') == "patch_done"]

    report.check("the run announced its size up front",
                 any(e.get('event') == "started" and e.get('total') == planned for e in events),
                 f"total {planned}")
    report.check("every patch was sent", len({e['patch'] for e in starts}) == planned,
                 f"{len(starts)} attempts over {len({e['patch'] for e in starts})} patches")
    report.check("every patch came back", {e['patch'] for e in dones} == set(range(1, planned + 1)),
                 str(sorted(e['patch'] for e in dones)))
    report.check("the book was saved after patches landed", kinds.count("saved") >= 1,
                 f"{kinds.count('saved')} saves")
    report.check("the run finished, reporting every patch successful",
                 any(e.get('event') == "finished" and e.get('successful') == planned
                     for e in events))
    report.check("the final snapshot says done", final['status'] == "done", final['status'])
    report.check("and counts every patch complete",
                 final['completed'] == final['total'] == planned,
                 f"{final['completed']}/{final['total']}")

    report.check("a patch is called a patch everywhere, never a batch",
                 not any("batch" in (e.get('message') or "").lower() for e in events))
    report.check("the patch numbers in the events are the ones the chapter list carries",
                 {e['patch'] for e in dones} <= {c['patch'] for c in final['stats']['chapters']})

    # ------------------------------------------------------------- mid-run
    served = [sample for sample in midrun['downloads'] if sample['served']]
    broken = [sample for sample in served if not sample['readable']]

    report.check("the book could be downloaded before the run ended",
                 len(served) > 0, f"{len(served)} of {len(midrun['downloads'])} were served")
    report.check("every mid-run download was a readable archive", not broken,
                 f"{len(broken)} of {len(served)} came back unusable: "
                 + ", ".join(sorted({sample['why'] for sample in broken})))
    if broken:
        # Kept as evidence: the book is rewritten in place after every patch, so
        # a download taken during that write gets however much of the zip had
        # been flushed by then — or a transfer that stops short of the length
        # the response promised.
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "truncated-mid-run.epub").write_bytes(broken[0]['content'])

    report.soft("the snapshot taken mid-run said running",
                (midrun.get('snapshot') or {}).get('status') == "running",
                str((midrun.get('snapshot') or {}).get('status')))

    # ------------------------------------------------------------- the book
    downloaded = api.download(job_id)
    report.check("the finished book downloads", downloaded.status_code == 200,
                 f"got {downloaded.status_code}")
    report.check("as an EPUB",
                 downloaded.headers.get("content-type") == "application/epub+zip",
                 downloaded.headers.get("content-type", ""))
    report.check("named after the book and the language it is now in",
                 download_filename(downloaded) == "zh_small (English).epub",
                 download_filename(downloaded))

    path = _save(downloaded, "zh_small-english.epub")
    ARTIFACTS['translated'] = path

    report.check("the archive has no duplicated members", duplicate_members(path) == [],
                 str(duplicate_members(path)))

    # By position, not by presence: an EPUB may declare several titles and
    # several languages, and a reader shows the first one it finds.
    metadata = opf_metadata(path)
    report.check("the language the book declares first is the one it is now in",
                 metadata['languages'][:1] == ["en"], str(metadata['languages']))
    report.check("and the title it declares first says it has been translated",
                 metadata['titles'][:1] == ["青石镇的铁匠 (Translated)"],
                 str(metadata['titles']))

    texts = chapter_texts(path)
    report.check("every chapter is still there", len(texts) == len(chapters),
                 f"{len(texts)} of {len(chapters)}")

    still_chinese = {name: round(script_ratio(text, "chinese"), 3)
                     for name, text in texts.items()
                     if script_ratio(text, "chinese") >= 0.05}
    report.check("no chapter is still in Chinese", not still_chinese, str(still_chinese))
    report.check("and every chapter reads as English",
                 all(script_ratio(text, "latin") > 0.8 for text in texts.values()),
                 str({n: round(script_ratio(t, "latin"), 2) for n, t in texts.items()}))
    report.check("chapters have real length, not a one-line summary",
                 all(len(text) > 200 for text in texts.values()),
                 str({n: len(t) for n, t in texts.items()}))

    # ------------------------------------------------------------- the glossary
    glossary = api.get_glossary(job_id).json()['terms']
    report.check("the reader's own term survived the run untouched",
                 glossary.get("李明") == "Bright Li", str(glossary.get("李明")))
    report.check("the run learned terms of its own", len(glossary) > 1,
                 f"{len(glossary)} terms: {list(glossary)[:8]}")
    report.check("every learned term is a Chinese term with an English rendering",
                 all(script_ratio(term, "chinese") > 0.5 for term in glossary),
                 str([t for t in glossary if script_ratio(t, "chinese") <= 0.5][:5]))

    whole_book = " ".join(texts.values())
    report.soft("the model used the name the reader chose",
                "Bright Li" in whole_book,
                "the glossary asked for 'Bright Li' for 李明")

    # ------------------------------------------------------------- reconnecting
    replayed, replayed_final = api.stream(job_id, timeout=60)
    report.check("reconnecting to a finished job replays the whole log",
                 len(replayed) == len(events), f"{len(replayed)} events, first time {len(events)}")
    report.check("and ends immediately rather than waiting",
                 replayed_final is not None and replayed_final['status'] == "done")


def test_selected_chapters_only(api: Api, report: Report, books):
    report.section("Translating one chapter out of a book")

    with settings(TOKENS_PER_REQUEST=6000):
        job = api.upload(books['zh_small']).json()
        job_id = job['id']
        chapters = job['stats']['chapters']
        chosen, untouched = chapters[1], [chapters[0], chapters[2]]

        response = api.start(job_id, [chosen['id']])
        if not report.check("a one-chapter run starts", response.status_code == 200,
                            response.text[:200]):
            return
        report.check("and costs one request",
                     response.json()['stats']['patch_count'] == 1,
                     f"{response.json()['stats']['patch_count']} patches")

        events, final = api.stream(job_id, timeout=600)

    report.check("it finishes", final and final['status'] == "done",
                 str(final and final['status']))

    path = _save(api.download(job_id), "zh_small-one-chapter.epub")
    texts = chapter_texts(path)

    report.check("the chosen chapter came back in English",
                 script_ratio(_by_file_name(texts, chosen['file_name']), "chinese") < 0.05,
                 f"{script_ratio(_by_file_name(texts, chosen['file_name']), 'chinese'):.2f} Chinese")
    report.check("the chapters nobody asked for are untouched",
                 all(script_ratio(_by_file_name(texts, c['file_name']), "chinese") > 0.5
                     for c in untouched),
                 str([round(script_ratio(_by_file_name(texts, c['file_name']), "chinese"), 2)
                      for c in untouched]))
    report.check("the book still holds every chapter", len(texts) == 3, f"{len(texts)}")
    report.check("and is a valid archive", duplicate_members(path) == [])


def test_cancelling_a_run(api: Api, report: Report, books):
    report.section("Stopping a run halfway")

    with settings(TOKENS_PER_REQUEST=1500, REQUESTS_PER_MINUTE=2):
        job = api.upload(books['zh_long']).json()
        job_id = job['id']
        planned = job['stats']['patch_count']
        report.check("the long book plans several patches", planned >= 4, f"{planned} patches")

        if api.start(job_id, None).status_code != 200:
            report.check("the run starts", False)
            return

        cancelled_at = {}

        def watch(event):
            if event.get('event') == "patch_done" and 'response' not in cancelled_at:
                cancelled_at['response'] = api.cancel(job_id)
                cancelled_at['at'] = time.time()

        events, final = api.stream(job_id, watch, timeout=900)

    report.check("cancelling answered with a snapshot",
                 cancelled_at.get('response') is not None
                 and cancelled_at['response'].status_code == 200)
    report.check("the run said it was stopping",
                 any(e.get('event') == "stopping" for e in events))
    report.check("and ended as cancelled rather than failed",
                 final and final['status'] == "cancelled", str(final and final['status']))
    report.check("some patches were done, and not all of them",
                 final and 1 <= final['completed'] < planned,
                 f"{final and final['completed']} of {planned}")
    report.check("stopping took seconds, not the rest of the run",
                 time.time() - cancelled_at.get('at', time.time()) < 180,
                 f"{time.time() - cancelled_at.get('at', time.time()):.0f}s after the cancel")

    path = _save(api.download(job_id), "zh_long-cancelled.epub")
    texts = chapter_texts(path)
    english = [t for t in texts.values() if script_ratio(t, "chinese") < 0.05]

    report.check("what was translated before the stop is in the book", len(english) >= 1,
                 f"{len(english)} of {len(texts)} chapters translated")
    report.check("the rest are left as they were", len(english) < len(texts),
                 f"{len(texts) - len(english)} still Chinese")
    report.check("and the partial book is a valid archive", duplicate_members(path) == [])


def test_cooldown_between_runs(api: Api, report: Report, books):
    report.section("The wait a visitor is asked for between translations")

    with settings(MINUTES_BETWEEN_TRANSLATIONS=60):
        status = api.status().json()
        report.check("the visitor who just ran a translation is told to wait",
                     status['cooldown_seconds'] > 0, f"{status['cooldown_seconds']}s")

        job = api.upload(books['zh_small']).json()
        response = api.start(job['id'], None)
        report.check("and a second run is refused", response.status_code == 429,
                     f"got {response.status_code}")
        report.check("with the wait spelled out in minutes",
                     "minute" in response.json()['detail'], response.json()['detail'])

        stranger = Api(ip="203.0.113.44")
        report.check("someone who has not run one is not made to wait",
                     stranger.status().json()['cooldown_seconds'] == 0)

    report.check("turning the cooldown off lifts it at once",
                 api.status().json()['cooldown_seconds'] == 0)


def test_finished_book_replans_to_nothing(api: Api, report: Report, books):
    report.section("Re-uploading a book that has already been translated")

    path = ARTIFACTS.get('translated')
    if not path:
        report.note("skipped", "the full run produced no book to re-upload")
        return

    job = api.upload(path, source_lang="chinese").json()
    stats = job['stats']

    report.check("it uploads and is read", job['status'] == "ready", job['status'])
    report.check("every chapter is recognised as already translated",
                 all(c['patch'] is None for c in stats['chapters']),
                 str([c['patch'] for c in stats['chapters']]))
    report.check("so the plan asks for nothing", stats['patch_count'] == 0,
                 f"{stats['patch_count']} patches")

    response = api.start(job['id'], None)
    report.check("and starting it is refused rather than costing a request",
                 response.status_code == 400, f"got {response.status_code}")


def test_right_to_left_target_and_missing_navigation(api: Api, report: Report, books):
    report.section("A book with no navigation, translated into Arabic")

    with settings(TOKENS_PER_REQUEST=6000):
        job = api.upload(books['zh_plain'], target_lang="Arabic").json()
        job_id = job['id']

        if api.start(job_id, None).status_code != 200:
            report.check("the run starts", False)
            return

        events, final = api.stream(job_id, timeout=600)

    report.check("it finishes", final and final['status'] == "done",
                 str(final and final['status']))

    downloaded = api.download(job_id)
    report.check("the download is named for the language asked for",
                 download_filename(downloaded) == "zh_plain (Arabic).epub",
                 download_filename(downloaded))

    path = _save(downloaded, "zh_plain-arabic.epub")
    texts = chapter_texts(path)
    metadata = opf_metadata(path)

    report.check("the language the book declares first is Arabic's code",
                 metadata['languages'][:1] == ["ar"], str(metadata['languages']))
    report.check("a book that arrived without a title is given one",
                 metadata['titles'][:1] == ["Translated Book"], str(metadata['titles']))
    report.check("the chapter came back in Arabic script",
                 all(script_ratio(text, "arabic") > 0.8 for text in texts.values()),
                 str({n: round(script_ratio(t, "arabic"), 2) for n, t in texts.items()}))

    members = [Path(name).name.lower() for name in epub_members(path)]
    report.check("a book that arrived without an NCX is given exactly one",
                 sum(1 for name in members if name.endswith(".ncx")) == 1,
                 str([n for n in members if n.endswith(".ncx")]))
    report.check("and exactly one nav document",
                 sum(1 for name in members if name == "nav.xhtml") == 1,
                 str([n for n in members if "nav" in n]))
    report.check("with nothing stored twice", duplicate_members(path) == [],
                 str(duplicate_members(path)))


TESTS = [
    test_full_run,
    test_selected_chapters_only,
    test_cancelling_a_run,
    test_cooldown_between_runs,
    test_finished_book_replans_to_nothing,
    test_right_to_left_target_and_missing_navigation,
]
