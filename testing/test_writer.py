"""
What comes back out of the archive, checked without translating anything.

`EpubWriter` is asked to save chapters it was just handed, unchanged. Nothing
here calls the API — a book saved without a single word being translated should
come back byte-comparable in everything but its metadata, so anything that
differs is the writer's doing and can be found for free.

This is the one suite that reaches inside the app rather than talking to it over
HTTP: the writer is where a run's output is decided, and a defect here shows up
in every book the server produces.
"""

import sys
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup

from harness import ROOT, Api, Report, RESULTS_DIR, opf_metadata

# Python puts this folder on the path, not the project above it, and only this
# suite needs the app itself rather than the server in front of it.
sys.path.insert(0, str(ROOT))
from epub_translate.book import EpubWriter, SourceBook  # noqa: E402


def _round_trip(source_path, target_language: str = "English", saves: int = 1) -> Path:
    """
    Read a book, hand its chapters straight back to the writer, and save.

    `saves` repeats the save the way a run does — the book is rewritten after
    every patch — because saving twice is what turns a duplicated manifest entry
    into a duplicated zip member.
    """
    source = SourceBook(str(source_path))
    chapters, _language = source.chapters("auto")
    writer = EpubWriter(source, target_language)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / f"roundtrip-{Path(source_path).stem}.epub"
    for _ in range(saves):
        writer.save(chapters, str(output))

    return output


def _documents(path) -> dict:
    with zipfile.ZipFile(path) as archive:
        return {Path(name).name: archive.read(name).decode("utf-8", "replace")
                for name in archive.namelist()
                if name.lower().endswith((".xhtml", ".html"))}


def test_nothing_is_lost_from_the_archive(api: Api, report: Report, books):
    report.section("A book saved without being translated")

    source_path = books['styled']
    output = _round_trip(source_path)

    with zipfile.ZipFile(source_path) as archive:
        before = {Path(name).name for name in archive.namelist()}
    with zipfile.ZipFile(output) as archive:
        after = {Path(name).name for name in archive.namelist()}

    report.check("every file that went in comes out", before <= after,
                 f"missing: {sorted(before - after)}")
    report.check("the stylesheet is still in the book", "main.css" in after)
    report.check("so is the image", "pic.png" in after)
    report.check("nothing is stored twice",
                 len(zipfile.ZipFile(output).namelist())
                 == len(set(zipfile.ZipFile(output).namelist())))


def test_chapters_keep_their_head(api: Api, report: Report, books):
    report.section("What survives in a chapter that was not even translated")

    output = _round_trip(books['styled'])
    documents = _documents(output)
    chapters = {name: html for name, html in documents.items() if name.startswith("ch")}

    report.check("both chapters are there", len(chapters) == 2, str(sorted(chapters)))

    linked = {name: "main.css" in html for name, html in chapters.items()}
    report.check("each chapter still links the book's stylesheet", all(linked.values()),
                 f"lost from: {sorted(name for name, has in linked.items() if not has)}")

    titled = {name: "<title>" in html for name, html in chapters.items()}
    report.check("each chapter still has a title element", all(titled.values()),
                 f"lost from: {sorted(name for name, has in titled.items() if not has)}")

    image = BeautifulSoup(chapters.get('ch1.xhtml', ""), "html.parser").find("img")
    report.check("an image in the text still points where it did",
                 image is not None and image.get("src") == "../Images/pic.png",
                 str(image and image.get("src")))

    body_text = BeautifulSoup(chapters['ch2.xhtml'], "html.parser").get_text(" ", strip=True)
    report.check("the words are unchanged", "布包里是半块玉佩" in body_text,
                 body_text[:60])
    report.check("and so is the markup around them",
                 'class="first"' in chapters['ch2.xhtml'],
                 f"{chapters['ch2.xhtml'].count('class=')} class attributes kept")


def test_metadata_says_one_thing(api: Api, report: Report, books):
    report.section("What the translated book says it is")

    output = _round_trip(books['styled'], target_language="English")
    metadata = opf_metadata(output)

    report.check("the book declares one language", len(metadata['languages']) == 1,
                 str(metadata['languages']))
    report.check("and it is the language it was translated into",
                 metadata['languages'][:1] == ["en"], str(metadata['languages']))
    report.check("the book declares one title", len(metadata['titles']) == 1,
                 str(metadata['titles']))
    report.check("and it is the translated one",
                 metadata['titles'][:1] == ["青石镇的铁匠 (排版版) (Translated)"],
                 str(metadata['titles']))


def test_saving_again_changes_nothing(api: Api, report: Report, books):
    report.section("Saving after every patch, as a run does")

    once = _round_trip(books['styled'], saves=1)
    once_members = zipfile.ZipFile(once).namelist()

    thrice = _round_trip(books['styled'], saves=3)
    thrice_members = zipfile.ZipFile(thrice).namelist()

    report.check("three saves leave the same members as one",
                 thrice_members == once_members,
                 f"{len(thrice_members)} members against {len(once_members)}")
    report.check("and none of them twice",
                 len(thrice_members) == len(set(thrice_members)),
                 str(sorted({name for name in thrice_members
                             if thrice_members.count(name) > 1})))


def test_navigation_is_not_doubled(api: Api, report: Report, books):
    report.section("Navigation, in books that have it and books that don't")

    with_nav = zipfile.ZipFile(_round_trip(books['styled'], saves=2)).namelist()
    report.check("a book that brought a nav document keeps exactly one",
                 sum(1 for name in with_nav if Path(name).name == "nav.xhtml") == 1,
                 str([name for name in with_nav if "nav" in name.lower()]))
    report.check("and is given the NCX it did not have",
                 sum(1 for name in with_nav if name.lower().endswith(".ncx")) == 1,
                 str([name for name in with_nav if name.lower().endswith(".ncx")]))

    without = zipfile.ZipFile(_round_trip(books['zh_plain'], saves=2)).namelist()
    report.check("a book that brought neither is given one of each",
                 sum(1 for name in without if Path(name).name == "nav.xhtml") == 1
                 and sum(1 for name in without if name.lower().endswith(".ncx")) == 1,
                 str([name for name in without if "nav" in name.lower()
                      or name.lower().endswith(".ncx")]))

    both = zipfile.ZipFile(_round_trip(books['zh_small'], saves=2)).namelist()
    report.check("a book that brought both keeps one of each",
                 sum(1 for name in both if Path(name).name == "nav.xhtml") == 1
                 and sum(1 for name in both if name.lower().endswith(".ncx")) == 1,
                 str([name for name in both if "nav" in name.lower()
                      or name.lower().endswith(".ncx")]))


def test_the_contents_are_rebuilt_from_the_chapters(api: Api, report: Report, books):
    report.section("The table of contents")

    output = _round_trip(books['styled'])
    with zipfile.ZipFile(output) as archive:
        ncx = next((archive.read(name).decode("utf-8", "replace")
                    for name in archive.namelist() if name.lower().endswith(".ncx")), "")

    listed = [title for title in ("青石镇的铁匠", "布包里的东西") if title in ncx]
    report.check("every chapter is listed in it", len(listed) == 2, f"{len(listed)} of 2")
    report.check("and the navigation document is not listed as a chapter",
                 "Contents" not in ncx,
                 "the nav document's own title" if "Contents" in ncx else "chapters only")


TESTS = [
    test_nothing_is_lost_from_the_archive,
    test_chapters_keep_their_head,
    test_metadata_says_one_thing,
    test_saving_again_changes_nothing,
    test_navigation_is_not_doubled,
    test_the_contents_are_rebuilt_from_the_chapters,
]
