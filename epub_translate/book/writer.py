"""Writing the translated EPUB back out."""

import os
import tempfile
from typing import Dict, List

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub

from ..language import language_to_code
from .chapter import Chapter
from .reader import SourceBook


class EpubWriter:
    """
    Writes the translated EPUB, preserving the original's structure and metadata.

    Works on the SourceBook's already-parsed archive rather than reading the
    file again, which is what keeps a run's repeated progress saves from
    re-unzipping every image, font and stylesheet each time. Making one takes
    that book over: from here on its documents are being replaced by
    translations, so chapters have to have been extracted first.
    """

    def __init__(self, source: SourceBook, target_language: str = "English"):
        self.book = source.book

        # Cleared first, both of them: ebooklib's setters *add* a title and a
        # language rather than replacing one, so a book read in and written back
        # out declares its original before the new one — and a reader shows the
        # first, which left finished translations presenting themselves as the
        # book they were translated from.
        _forget_metadata(self.book, 'title')
        self.book.set_title(f"{source.title} (Translated)" if source.title else "Translated Book")
        _forget_metadata(self.book, 'language')
        self.book.set_language(language_to_code(target_language))

        _ensure_navigation(self.book)
        _keep_document_heads(self.book)

        # The soup last written into each document. The book is saved after
        # every patch, and a patch translates a handful of chapters, so without
        # this each save re-serializes the several hundred that did not change —
        # and does it holding the save lock.
        self._written: Dict[str, BeautifulSoup] = {}

    def save(self, chapters: List[Chapter], output_path: str):
        """Fold the current state of `chapters` into the book and write it to disk."""
        translated_soups = {chapter.id: chapter.soup for chapter in chapters}
        titles = {chapter.id: chapter.display_title for chapter in chapters}

        for item in self.book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            item_id = item.get_id()
            soup = translated_soups.get(item_id)

            # Compared by identity, because that is exactly what changes: the
            # worker rebinds `chapter.soup` to the parsed translation and leaves
            # every untouched chapter pointing at the soup it was read with.
            if soup is not None and self._written.get(item_id) is not soup:
                item.content = str(_as_html_document(soup)).encode('utf-8')
                # The head is rebuilt from this rather than from the content
                # above; see `_keep_document_heads`.
                item.title = titles[item_id]
                self._written[item_id] = soup

        self.book.toc = [
            epub.Link(chapter.file_name or f"{chapter.id}.html",
                      chapter.display_title,
                      chapter.id)
            for chapter in chapters
        ]

        _write_in_one_move(self.book, output_path)


def _forget_metadata(book, name: str):
    """
    Drop a Dublin Core field, so the value set next is the only one.

    `set_title` and `set_language` append — ebooklib supports a book declaring
    several of each — and nothing here wants the second one.
    """
    book.metadata.get(epub.NAMESPACES['DC'], {}).pop(name, None)


def _keep_document_heads(book):
    """
    Carry each document's `<head>` onto the item ebooklib will write it from.

    ebooklib does not write out the document it is given: it rebuilds one, and
    the `<head>` of that rebuild holds `item.title` and `item.links` and nothing
    else. Reading a book fills in neither, so without this every chapter loses
    its `<title>` and — the part a reader sees — the `<link>` to the book's
    stylesheet, leaving the CSS in the archive with nothing pointing at it and
    the translation rendering unstyled.

    Done once, here, because this is the last moment every document in the book
    is still the one that arrived: from the first save onwards they are being
    replaced by translations.
    """
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        content = item.content
        if not content:
            # The nav document `_ensure_navigation` just added, whose content
            # ebooklib generates at write time and which has no head to keep.
            continue
        if isinstance(content, bytes):
            content = content.decode('utf-8', 'replace')

        head = BeautifulSoup(content, 'html.parser').head
        if not head:
            continue

        item.links = [_attributes(link) for link in head.find_all('link')]
        if head.title and head.title.get_text(strip=True):
            item.title = head.title.get_text(strip=True)


def _attributes(tag) -> Dict[str, str]:
    """
    A tag's attributes, every value a string.

    BeautifulSoup hands back a list for the attributes HTML allows several
    values in — `rel="stylesheet"` arrives as `['stylesheet']` — and the writer
    these are going to builds elements with lxml, which takes strings only.
    """
    return {name: " ".join(value) if isinstance(value, list) else value
            for name, value in tag.attrs.items()}


def _write_in_one_move(book, output_path: str):
    """
    Write the book beside its destination, then rename it into place.

    The server serves this same path while the run that is writing it is still
    going, and `write_epub` rewrites the archive from the beginning: a download
    landing inside a save was getting however much of the zip had been flushed
    by then. A rename is atomic, so a reader gets either the previous book or
    the new one — the same trick that keeps `glossary/storage.py` honest, and
    for the same reason.
    """
    directory = os.path.dirname(os.path.abspath(output_path))
    handle, temp_path = tempfile.mkstemp(dir=directory, suffix='.epub.tmp')
    os.close(handle)

    try:
        epub.write_epub(temp_path, book, {})
        os.replace(temp_path, output_path)
    except Exception:
        os.unlink(temp_path)
        raise


def _ensure_navigation(book):
    """
    Give the book a table of contents and an NCX, unless it brought its own.

    Adding them unconditionally is the obvious thing and it is wrong: a book that
    already has them ends up with two manifest entries under one id and two zip
    members under one name — `zipfile` says so on every save, `Duplicate name:
    'EPUB/nav.xhtml'` — and duplicate ids make the EPUB invalid. Most EPUB3 books
    have both already.
    """
    present = {type(item) for item in book.get_items()}

    if epub.EpubNcx not in present:
        book.add_item(epub.EpubNcx())
    if epub.EpubNav not in present:
        book.add_item(epub.EpubNav())


def _as_html_document(soup: BeautifulSoup) -> BeautifulSoup:
    """Wrap a bare fragment in a minimal XHTML document; full documents pass through."""
    if soup.html:
        return soup

    return BeautifulSoup(f"""<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>Chapter</title>
    <meta charset="utf-8"/>
</head>
<body>
{soup}
</body>
</html>""", 'html.parser')
