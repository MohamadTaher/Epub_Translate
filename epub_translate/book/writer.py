"""Writing the translated EPUB back out."""

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
        self.book.set_title(f"{source.title} (Translated)" if source.title else "Translated Book")
        self.book.set_language(language_to_code(target_language))
        self.book.add_item(epub.EpubNcx())
        self.book.add_item(epub.EpubNav())

        # The soup last written into each document. The book is saved after
        # every batch, and a batch translates a handful of chapters, so without
        # this each save re-serializes the several hundred that did not change —
        # and does it holding the save lock.
        self._written: Dict[str, BeautifulSoup] = {}

    def save(self, chapters: List[Chapter], output_path: str):
        """Fold the current state of `chapters` into the book and write it to disk."""
        translated_soups = {chapter.id: chapter.soup for chapter in chapters}

        for item in self.book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            item_id = item.get_id()
            soup = translated_soups.get(item_id)

            # Compared by identity, because that is exactly what changes: the
            # worker rebinds `chapter.soup` to the parsed translation and leaves
            # every untouched chapter pointing at the soup it was read with.
            if soup is not None and self._written.get(item_id) is not soup:
                item.content = str(_as_html_document(soup)).encode('utf-8')
                self._written[item_id] = soup

        self.book.toc = [
            epub.Link(chapter.file_name or f"{chapter.id}.html",
                      chapter.display_title,
                      chapter.id)
            for chapter in chapters
        ]

        epub.write_epub(output_path, self.book, {})


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
