"""
What every test in this folder is built out of.

Four things live here: a client that calls the API the way the browser does, a
way to retune `settings.env` for the length of a `with` block and put it back, a
handful of EPUB inspections, and the tally that becomes the report at the end.

The inspections deliberately avoid `epub_translate`. Checking the writer's
output with the reader that produced it would agree with itself no matter what
went wrong, so a downloaded book is opened here as a plain zip and its text is
counted character by character.
"""

import io
import json
import os
import re
import sys
import time
import traceback
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = ROOT / "settings.env"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Inside the app container this is uvicorn itself. From anywhere else, point it
# at the published port instead.
BASE_URL = os.environ.get("EPUB_TEST_BASE_URL", "http://localhost:7860")

# Unicode ranges, restated here rather than imported, so a book that comes back
# untranslated cannot be declared translated by the same table that misjudged it.
SCRIPTS = {
    'chinese': [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
    'arabic': [(0x0600, 0x06FF)],
    'latin': [(0x0041, 0x005A), (0x0061, 0x007A)],
}

# The packing rule as `packing.py` documents it, restated so the request count a
# reader is shown can be checked against the rule rather than against the code
# that produced it.
PATCH_OVERHEAD_TOKENS = 500
CHAPTER_SEPARATOR_TOKENS = 50


# --------------------------------------------------------------------- the API

class Api:
    """
    The endpoints, called the way the browser calls them.

    `ip` is sent as `X-Forwarded-For`, which is what `_client_ip` reads: that is
    how a test pretends to be a second visitor, since the cooldown is per
    address and every test otherwise arrives from the same one.
    """

    def __init__(self, base_url: str = BASE_URL, ip: str = "198.51.100.7",
                 headers: Optional[Dict[str, str]] = None):
        self.base_url = base_url.rstrip("/")
        self.ip = ip
        # Sent with every call. The one use is a `Host` header: the Vite dev
        # server answers 403 to a request that arrives under the container's own
        # hostname, so reaching it over the compose network means saying
        # localhost the way a browser would.
        self.headers = headers or {}

    def _call(self, method: str, path: str, **kwargs) -> requests.Response:
        headers = {'X-Forwarded-For': self.ip, **self.headers, **kwargs.pop('headers', {})}
        kwargs.setdefault('timeout', 60)
        return requests.request(method, f"{self.base_url}{path}", headers=headers, **kwargs)

    def status(self) -> requests.Response:
        return self._call("GET", "/api/status")

    def upload(self, path, source_lang: str = "auto", target_lang: str = "English",
               filename: Optional[str] = None) -> requests.Response:
        path = Path(path)
        with open(path, "rb") as handle:
            return self._call(
                "POST", "/api/jobs",
                files={'file': (filename or path.name, handle.read(), "application/epub+zip")},
                data={'source_lang': source_lang, 'target_lang': target_lang},
            )

    def job(self, job_id: str) -> requests.Response:
        return self._call("GET", f"/api/jobs/{job_id}")

    def preview(self, job_id: str, chapter_ids: Optional[List[str]] = None) -> requests.Response:
        return self._call("POST", f"/api/jobs/{job_id}/preview", json={'chapter_ids': chapter_ids})

    def start(self, job_id: str, chapter_ids: Optional[List[str]] = None) -> requests.Response:
        return self._call("POST", f"/api/jobs/{job_id}/start", json={'chapter_ids': chapter_ids})

    def cancel(self, job_id: str) -> requests.Response:
        return self._call("POST", f"/api/jobs/{job_id}/cancel")

    def cover(self, job_id: str) -> requests.Response:
        return self._call("GET", f"/api/jobs/{job_id}/cover")

    def download(self, job_id: str) -> requests.Response:
        return self._call("GET", f"/api/jobs/{job_id}/download")

    def get_glossary(self, job_id: str) -> requests.Response:
        return self._call("GET", f"/api/jobs/{job_id}/glossary")

    def put_glossary(self, job_id: str, terms: Dict) -> requests.Response:
        return self._call("PUT", f"/api/jobs/{job_id}/glossary", json={'terms': terms})

    def stream(self, job_id: str, on_event: Optional[Callable[[Dict], None]] = None,
               timeout: int = 900) -> Tuple[List[Dict], Optional[Dict]]:
        """
        Follow a job to its end, returning every event and the final snapshot.

        `on_event` runs on this thread, between two reads of the socket, which is
        what lets a test cancel a run or download the book mid-flight at a known
        point rather than after a guessed sleep.

        The stream is deliberately not reopened when it closes: the server ends
        it when the job is terminal, and a reconnect would replay the whole log
        and end again, forever. Tests that want the replay ask for it by calling
        this a second time.
        """
        events: List[Dict] = []
        final: Optional[Dict] = None
        deadline = time.time() + timeout
        name = None

        with self._call("GET", f"/api/jobs/{job_id}/events", stream=True,
                        timeout=(10, 60)) as response:
            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
                if time.time() > deadline:
                    raise TimeoutError(f"job {job_id} was still running after {timeout}s")
                if not line or line.startswith(":"):        # blank, or a keep-alive
                    continue
                if line.startswith("event:"):
                    name = line[len("event:"):].strip()
                    continue
                if not line.startswith("data:"):
                    continue

                payload = json.loads(line[len("data:"):].strip())
                if name == "end":
                    final = payload
                    break

                events.append(payload)
                if on_event:
                    on_event(payload)

        return events, final


def wait_for(condition: Callable[[], bool], timeout: float = 30, interval: float = 0.5) -> bool:
    """Poll until something becomes true, or give up and say so."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return condition()


# ---------------------------------------------------------------- the settings

_write_counter = 0


@contextmanager
def settings(**overrides):
    """
    Retune `settings.env` for the length of a `with` block, then put it back.

    The server re-reads the file whenever its timestamp *or size* moves, so this
    guarantees the size moves: a bind-mounted Windows directory can report
    timestamps too coarse to tell two writes apart, and a same-size rewrite
    inside one tick would leave the old values cached with nothing to show for
    it. Restoring writes the original text back byte for byte, so the file the
    reader edits is never left with a test's numbers in it.
    """
    global _write_counter
    original = SETTINGS_FILE.read_text(encoding="utf-8")

    _write_counter += 1
    marked = _with_overrides(original, overrides) + f"\n# test override {'-' * _write_counter}\n"
    while len(marked.encode("utf-8")) == len(original.encode("utf-8")):
        marked += "-"

    try:
        SETTINGS_FILE.write_text(marked, encoding="utf-8")
        yield
    finally:
        SETTINGS_FILE.write_text(original, encoding="utf-8")


def _with_overrides(text: str, overrides: Dict[str, object]) -> str:
    """Replace the settings named, and append the ones the file doesn't carry."""
    lines = text.splitlines()
    remaining = dict(overrides)

    for index, line in enumerate(lines):
        name = line.split("=", 1)[0].strip()
        if name in remaining:
            lines[index] = f"{name}={remaining.pop(name)}"

    lines += [f"{name}={value}" for name, value in remaining.items()]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------- reading an EPUB

def epub_documents(path) -> Dict[str, str]:
    """Every XHTML document in the archive, by the name it is stored under."""
    with zipfile.ZipFile(path) as archive:
        return {
            name: archive.read(name).decode("utf-8", "replace")
            for name in archive.namelist()
            if name.lower().endswith((".xhtml", ".html", ".htm"))
        }


def epub_members(path) -> List[str]:
    """Every member, duplicates included — which is the point of asking."""
    with zipfile.ZipFile(path) as archive:
        return [info.filename for info in archive.infolist()]


def duplicate_members(path) -> List[str]:
    """
    Names stored more than once.

    A zip may hold two members under one name and most readers show the first,
    so a book that duplicates its nav document looks fine until something strict
    opens it. `EpubWriter._ensure_navigation` exists to prevent exactly this.
    """
    seen, duplicates = set(), []
    for name in epub_members(path):
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    return duplicates


def opf_metadata(path) -> Dict[str, List[str]]:
    """
    The titles and languages the package document declares, in the order it
    declares them.

    Lists rather than single values, because an EPUB may carry several of each
    and a reader shows the first — so "what language is this book in" is
    answered by position, not by whether the right value is in there somewhere.
    """
    with zipfile.ZipFile(path) as archive:
        opf_name = next((n for n in archive.namelist() if n.lower().endswith(".opf")), None)
        if not opf_name:
            return {'titles': [], 'languages': []}
        opf = archive.read(opf_name).decode("utf-8", "replace")

    def field(tag: str) -> List[str]:
        return [value.strip()
                for value in re.findall(rf"<dc:{tag}[^>]*>(.*?)</dc:{tag}>", opf, re.S)]

    return {'titles': field("title"), 'languages': field("language")}


def is_epub(content: bytes) -> bool:
    """Whether these bytes are an archive that opens."""
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            return archive.testzip() is None
    except (zipfile.BadZipFile, OSError):
        return False


def download_filename(response) -> str:
    """
    The name a browser would save a download under.

    Starlette percent-encodes the name into `filename*=utf-8''…` as soon as it
    holds anything that needs quoting — a space and a bracket are enough — so
    the header has to be decoded before it can be read.
    """
    disposition = response.headers.get("content-disposition", "")
    match = re.search(r"filename\*=utf-8''([^;]+)", disposition) or \
        re.search(r'filename="([^"]+)"', disposition)
    return unquote(match.group(1)) if match else ""


def visible_text(html: str) -> str:
    """The words a reader would see, with the markup taken out."""
    soup = BeautifulSoup(html, "html.parser")
    return (soup.body or soup).get_text(" ", strip=True)


def script_ratio(text: str, script: str) -> float:
    """
    Share of the letters in `text` that belong to `script`.

    Counted here rather than imported: this is the measure a translated chapter
    is judged by, so it has to be independent of the code that decided the
    chapter was translated.
    """
    ranges = SCRIPTS[script]
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return 0.0

    inside = sum(1 for character in letters
                 if any(low <= ord(character) <= high for low, high in ranges))
    return inside / len(letters)


def chapter_texts(path) -> Dict[str, str]:
    """
    Every document in the book that carries words, by member name.

    The cover page and the navigation are dropped: neither is a chapter, and
    both would drag a script ratio towards whichever language wrote the markup.
    """
    texts = {}
    for name, html in epub_documents(path).items():
        if "nav" in Path(name).name.lower() or "cover" in Path(name).name.lower():
            continue
        text = visible_text(html)
        if text:
            texts[name] = text
    return texts


def patches_by_rule(token_counts: List[int], limit: int) -> List[List[int]]:
    """
    How `packing.py` says these chapters should be grouped.

    A restatement of the documented rule — 500 tokens of room for the prompt, 50
    for each separator, and an oversized chapter gets a patch to itself — so the
    request count a reader is shown can be checked against the rule instead of
    against the implementation of it.
    """
    patches: List[List[int]] = []
    current: List[int] = []
    total = PATCH_OVERHEAD_TOKENS

    for tokens in token_counts:
        separator = CHAPTER_SEPARATOR_TOKENS if current else 0
        if current and total + separator + tokens > limit:
            patches.append(current)
            current, total = [tokens], PATCH_OVERHEAD_TOKENS + tokens
        else:
            current.append(tokens)
            total += separator + tokens

    if current:
        patches.append(current)
    return patches


# ------------------------------------------------------------------- the tally

class Report:
    """
    What passed, what failed, and what is only worrying.

    A soft check is one whose answer comes from the model rather than from the
    code — whether it honoured a glossary term, say. Those are worth watching
    and not worth failing a build over, so they are counted apart.
    """

    def __init__(self, title: str):
        self.title = title
        self.entries: List[Dict] = []
        self.started = time.time()
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")

    def section(self, name: str):
        print(f"\n-- {name} " + "-" * max(0, 72 - len(name)))

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        return self._record("PASS" if passed else "FAIL", name, detail)

    def soft(self, name: str, passed: bool, detail: str = "") -> bool:
        return self._record("PASS" if passed else "WARN", name, detail)

    def note(self, name: str, detail: str = ""):
        self._record("INFO", name, detail)

    def broke(self, name: str, error: BaseException):
        self._record("FAIL", name, f"{type(error).__name__}: {error}")
        traceback.print_exc()

    def _record(self, outcome: str, name: str, detail: str) -> bool:
        self.entries.append({'outcome': outcome, 'name': name, 'detail': detail})
        print(f"  [{outcome}] {name}" + (f"  -- {detail}" if detail else ""))
        return outcome == "PASS"

    def counts(self) -> Dict[str, int]:
        tally = {'PASS': 0, 'FAIL': 0, 'WARN': 0, 'INFO': 0}
        for entry in self.entries:
            tally[entry['outcome']] += 1
        return tally

    def summary(self) -> int:
        tally = self.counts()
        elapsed = time.time() - self.started
        print(f"\n{'=' * 78}")
        print(f"{self.title}: {tally['PASS']} passed, {tally['FAIL']} failed, "
              f"{tally['WARN']} warnings, in {elapsed:.0f}s")

        for entry in self.entries:
            if entry['outcome'] in ("FAIL", "WARN"):
                print(f"  [{entry['outcome']}] {entry['name']}"
                      + (f"  -- {entry['detail']}" if entry['detail'] else ""))
        print("=" * 78)

        return tally['FAIL']

    def save(self, name: str) -> Path:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / name
        path.write_text(json.dumps({
            'title': self.title,
            'at': time.strftime("%Y-%m-%d %H:%M:%S"),
            'seconds': round(time.time() - self.started, 1),
            'counts': self.counts(),
            'entries': self.entries,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        return path


def run_tests(report: Report, tests: List[Callable], *arguments) -> None:
    """
    Run each test, and let a failing one fail alone.

    A test that raises has still told us something, and the ones after it are
    usually about something else entirely — so the traceback is recorded as a
    failure and the suite carries on.
    """
    for test in tests:
        try:
            test(*arguments)
        except Exception as error:
            report.broke(f"{test.__name__} raised", error)
        sys.stdout.flush()
