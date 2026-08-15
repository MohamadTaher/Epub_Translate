"""Reading the source EPUB: its chapters, and the metadata shown alongside them."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub

from .. import language
from .chapter import Chapter


@dataclass
class BookInfo:
    """
    Title, author and cover art, for showing a reader the book they uploaded
    rather than the filename they happened to give it.

    Every field is optional: a book with no metadata and no cover is still
    perfectly translatable.
    """
    title: Optional[str] = None
    author: Optional[str] = None
    cover_bytes: Optional[bytes] = None
    cover_media_type: Optional[str] = None


def extract_chapter_title(soup: BeautifulSoup) -> str:
    """Extract a meaningful title from chapter content."""
    if soup.title and soup.title.get_text(strip=True):
        return soup.title.get_text(strip=True)

    for tag_name in ['h1', 'h2', 'h3']:
        heading = soup.find(tag_name)
        if heading and heading.get_text(strip=True):
            return heading.get_text(strip=True)

    first_p = soup.find('p')
    if first_p:
        text = first_p.get_text(strip=True)
        if text and len(text) < 100:
            return text

    return "Untitled Chapter"


class SourceBook:
    """
    An EPUB opened once and held.

    `epub.read_epub` unzips the archive and parses every image, font and
    stylesheet in it, so a book is read once per run and then shared: the
    chapters, the cover, and the writer that produces the translated copy all
    come off this one object. Constructing it raises if the file is not a
    readable EPUB — the one moment where that is worth failing on.
    """

    def __init__(self, path: str):
        self.path = str(path)
        self.book = epub.read_epub(self.path)

        # Read now and kept, because the writer works on this same book object
        # and retitles it. Whoever asks afterwards still wants the title the
        # book arrived with.
        self.title = _first_metadata(self.book, 'DC', 'title')
        self.author = _first_metadata(self.book, 'DC', 'creator')

    def info(self) -> BookInfo:
        """
        The book's own description of itself. Cosmetic, so nothing here raises:
        a cover that cannot be read costs a thumbnail, not a translation.
        """
        info = BookInfo(title=self.title, author=self.author)

        try:
            cover = _find_cover_item(self.book)
            if cover is not None:
                info.cover_bytes = cover.get_content()
                info.cover_media_type = getattr(cover, 'media_type', None) or "image/jpeg"
        except Exception:
            pass

        return info

    def chapters(self, source_language: str = "auto") -> Tuple[List[Chapter], str]:
        """
        Every chapter in the book, already translated ones included, alongside
        the source language they were judged against — which differs from
        `source_language` when it was "auto".

        Fresh Chapter objects each call, deliberately: a run translates them in
        place, so a caller re-planning a different selection has to be handed
        untouched ones. Only the parsing of the archive is shared.
        """
        chapters = []
        chapter_counter = 1

        for item in self.book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            content = item.get_content()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            soup = BeautifulSoup(content, 'html.parser')

            if not (soup.body and soup.body.get_text(strip=True)):
                continue

            if _is_navigation(item, soup):
                continue

            title = extract_chapter_title(soup)
            if title == "Untitled Chapter":
                title = f"Chapter {chapter_counter}"

            chapters.append(Chapter(
                id=item.get_id() or f"chapter_{chapter_counter}",
                title=title,
                file_name=item.get_name(),
                soup=soup,
            ))
            chapter_counter += 1

        resolved_language = source_language
        if language.normalize(source_language) == "auto":
            resolved_language = language.detect(_language_sample(chapters))

        _mark_already_translated(chapters, resolved_language)

        return chapters, resolved_language


def _is_navigation(item, soup: BeautifulSoup) -> bool:
    """
    Whether this document is the table of contents rather than a chapter.

    It is a document like any other, so nothing else here would tell it apart —
    it has a body, it has text, and its <title> is the book's own, which is why
    it turns up in the chapter list wearing the book's name. Translating it
    spends a request on a list of links and hands the model markup whose hrefs
    have to survive intact, so it is left alone; `EpubWriter` rebuilds the
    contents from the translated chapter titles anyway.

    ebooklib gives an EpubNav only when the manifest marks the item
    `properties="nav"`; the element itself catches the books that don't.
    """
    return isinstance(item, epub.EpubNav) or soup.find('nav', attrs={'epub:type': True}) is not None


def _language_sample(chapters: List[Chapter]) -> str:
    """Enough of the book's text to recognise its script, and no more."""
    return "".join(chapter.soup.get_text(" ", strip=True)[:2000] for chapter in chapters[:20])


def _mark_already_translated(chapters: List[Chapter], source_language: str):
    for chapter in chapters:
        chapter.already_translated = language.looks_translated(
            chapter.soup.get_text(" ", strip=True), source_language
        )
        chapter.skip_reason = (
            f"No {source_language.title()} characters detected"
            if chapter.already_translated else None
        )


def _first_metadata(book, namespace: str, name: str) -> Optional[str]:
    """First value of a metadata field, or None when the book doesn't carry it."""
    try:
        entries = book.get_metadata(namespace, name)
    except Exception:
        return None

    for value, _attributes in entries or []:
        if value and value.strip():
            return value.strip()
    return None


def _find_cover_item(book):
    """
    Locate the cover image, trying the three ways EPUBs record one.

    EPUB3 marks it in the manifest, EPUB2 points at it from a <meta> tag, and
    plenty of books do neither — hence the guess at the end, and the None.
    """
    covers = list(book.get_items_of_type(ebooklib.ITEM_COVER))
    if covers:
        return covers[0]

    for _value, attributes in book.get_metadata('OPF', 'cover') or []:
        item = book.get_item_with_id(attributes.get('content', ''))
        if item is not None:
            return item

    images = list(book.get_items_of_type(ebooklib.ITEM_IMAGE))
    for item in images:
        if 'cover' in item.get_name().lower():
            return item

    return images[0] if images else None
