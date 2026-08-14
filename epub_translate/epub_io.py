from typing import Dict, List, Optional, Tuple

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub

from .logging_utils import logger
from .tokens import count_tokens

# ISO 639-1 codes for languages this tool is commonly used with. Falls back to
# "en" for anything not listed so EPUB metadata is never left blank.
LANGUAGE_CODES = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "japanese": "ja",
    "korean": "ko",
    "chinese": "zh",
    "russian": "ru",
    "arabic": "ar",
    "hebrew": "he",
    "greek": "el",
    "thai": "th",
    "hindi": "hi",
}

# Unicode ranges of each language's script, used to tell whether a chapter is
# still in the source language. Only languages written in a non-Latin script can
# be detected this way; see `source_script_ratio`.
SCRIPT_RANGES = {
    "chinese":  [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
    "japanese": [(0x3040, 0x309F), (0x30A0, 0x30FF), (0x4E00, 0x9FFF)],
    "korean":   [(0xAC00, 0xD7AF), (0x1100, 0x11FF)],
    "russian":  [(0x0400, 0x04FF)],
    "arabic":   [(0x0600, 0x06FF)],
    "hebrew":   [(0x0590, 0x05FF)],
    "greek":    [(0x0370, 0x03FF)],
    "thai":     [(0x0E00, 0x0E7F)],
    "hindi":    [(0x0900, 0x097F)],
}

# Below this share of source-script characters, a chapter is considered
# translated already. Generous, because a translated chapter often retains a few
# proper nouns in the original script.
TRANSLATED_SCRIPT_RATIO = 0.05


def language_to_code(language_name: str) -> str:
    """Map a human-readable language name (or existing code) to an ISO 639-1 code."""
    normalized = language_name.strip().lower()
    if len(normalized) == 2:
        return normalized
    return LANGUAGE_CODES.get(normalized, "en")


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


def source_script_ratio(text: str, source_language: str) -> float:
    """
    Share of `text` written in the source language's script, ignoring whitespace,
    digits and punctuation.

    Raises KeyError if the language has no detectable script; callers should use
    `script_is_detectable` first.
    """
    ranges = SCRIPT_RANGES[source_language.strip().lower()]

    considered = 0
    in_script = 0
    for char in text:
        if not char.isalpha():
            continue
        considered += 1
        code_point = ord(char)
        if any(low <= code_point <= high for low, high in ranges):
            in_script += 1

    if not considered:
        return 0.0
    return in_script / considered


def script_is_detectable(source_language: str) -> bool:
    """
    Whether a chapter's language can be judged from its characters alone.

    Only true for non-Latin scripts. Spanish and English share an alphabet, so
    no amount of character inspection distinguishes a translated chapter from an
    untranslated one.
    """
    return source_language.strip().lower() in SCRIPT_RANGES


def detect_source_language(chapters: List[Dict]) -> str:
    """
    Guess the source language from the characters used across the book.

    Returns a language name from SCRIPT_RANGES, or "auto" when no non-Latin
    script accounts for a meaningful share of the text (which includes every
    Latin-script book).
    """
    sample = "".join(ch['soup'].get_text(" ", strip=True)[:2000] for ch in chapters[:20])
    if not sample:
        return "auto"

    best_language = max(SCRIPT_RANGES, key=lambda lang: source_script_ratio(sample, lang))
    if source_script_ratio(sample, best_language) < 0.2:
        return "auto"
    return best_language


def is_chapter_already_translated(soup: BeautifulSoup, source_language: str) -> bool:
    """
    Check whether a chapter has already been translated out of `source_language`,
    by looking at how much of its text is still in the source script.

    Languages whose script can't be detected are never reported as translated:
    re-translating a chapter wastes tokens, but wrongly skipping one silently
    leaves the book half translated.
    """
    if not script_is_detectable(source_language):
        return False

    text = soup.get_text(" ", strip=True)
    if not text:
        return False

    return source_script_ratio(text, source_language) < TRANSLATED_SCRIPT_RATIO


def extract_chapters(epub_path: str, source_language: str = "auto") -> Tuple[List[Dict], str]:
    """
    Extract all chapters from the EPUB, including already translated ones.

    Returns the chapters alongside the source language actually used to judge
    them, which differs from `source_language` when it was "auto".
    """
    chapters = []
    book = epub.read_epub(epub_path)
    chapter_counter = 1

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        content = item.get_content()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        soup = BeautifulSoup(content, 'html.parser')

        if soup.body and soup.body.get_text(strip=True):
            chapter_title = extract_chapter_title(soup)
            if not chapter_title or chapter_title == "Untitled Chapter":
                chapter_title = f"Chapter {chapter_counter}"

            chapters.append({
                'id': item.get_id() or f"chapter_{chapter_counter}",
                'title': chapter_title,
                'soup': soup,
                'file_name': item.get_name(),
            })
            chapter_counter += 1

    resolved_language = source_language
    if source_language.strip().lower() == "auto":
        resolved_language = detect_source_language(chapters)

    detectable = script_is_detectable(resolved_language)
    for chapter in chapters:
        chapter['already_translated'] = is_chapter_already_translated(chapter['soup'], resolved_language)
        chapter['skip_reason'] = (
            f"No {resolved_language.title()} characters detected"
            if chapter['already_translated'] else None
        )

    already_translated_count = sum(1 for ch in chapters if ch['already_translated'])

    logger.system(f"Extracted {len(chapters)} total chapters")
    if not detectable:
        logger.system(
            f"Source language '{resolved_language}' uses the Latin alphabet, so already-translated "
            f"chapters can't be detected. All chapters will be translated."
        )
    elif already_translated_count > 0:
        logger.system(f"Found {already_translated_count} already translated chapters (will be preserved)")

    return chapters, resolved_language


PATCH_OVERHEAD_TOKENS = 500      # room for the prompt wrapped around every patch
PATCH_SEPARATOR_TOKENS = 50      # the marker joining two chapters within a patch


def pack_by_tokens(items: List[Tuple[object, int]], max_tokens_per_patch: int) -> List[List[object]]:
    """
    Group (item, token count) pairs into patches that stay under the token limit.

    Kept separate from `create_patches` so the server can work out how many API
    requests a selection would take without touching chapter objects or re-reading
    the book. Both paths must pack identically, or the count shown to a visitor
    would not be the count they are charged for.

    An item larger than the limit still gets a patch of its own; splitting a
    chapter mid-way would break the translation, so an oversized request is the
    lesser problem.
    """
    patches: List[List[object]] = []
    current_patch: List[object] = []
    current_tokens = PATCH_OVERHEAD_TOKENS

    for item, item_tokens in items:
        separator_tokens = PATCH_SEPARATOR_TOKENS if current_patch else 0

        if current_patch and current_tokens + separator_tokens + item_tokens > max_tokens_per_patch:
            patches.append(current_patch)
            current_patch = [item]
            current_tokens = PATCH_OVERHEAD_TOKENS + item_tokens
        else:
            current_patch.append(item)
            current_tokens += separator_tokens + item_tokens

    if current_patch:
        patches.append(current_patch)

    return patches


def count_chapter_tokens(chapters: List[Dict]):
    """
    Cache each chapter's pre-translation HTML and token count on the chapter dict.

    Done for every chapter rather than only the ones about to be translated, so a
    visitor deselecting chapters can be shown an accurate request count without
    the book being parsed again.
    """
    for chapter in chapters:
        if 'source_tokens' not in chapter:
            chapter['source_html'] = str(chapter['soup'])
            chapter['source_tokens'] = count_tokens(chapter['source_html'])


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


def read_book_info(epub_path: str) -> Dict:
    """
    Title, author and cover art, for showing the visitor the book they uploaded
    rather than the filename they happened to give it.

    Every field is optional. A book with no metadata and no cover is still
    perfectly translatable, so nothing here is allowed to raise.
    """
    info = {'title': None, 'author': None, 'cover_bytes': None, 'cover_media_type': None}

    try:
        book = epub.read_epub(epub_path)
    except Exception:
        return info

    info['title'] = _first_metadata(book, 'DC', 'title')
    info['author'] = _first_metadata(book, 'DC', 'creator')

    try:
        cover = _find_cover_item(book)
        if cover is not None:
            info['cover_bytes'] = cover.get_content()
            info['cover_media_type'] = getattr(cover, 'media_type', None) or "image/jpeg"
    except Exception:
        pass    # a missing or unreadable cover is cosmetic

    return info


def create_patches(chapters: List[Dict], max_tokens_per_patch: int = 15000,
                   include_translated: bool = False) -> List[List[Dict]]:
    """
    Group chapters into patches based on token limits, excluding already translated chapters.
    This batching improves API efficiency while staying under token limits.

    `include_translated` overrides the exclusion, for when the caller would rather
    re-translate everything than trust the detection.
    """
    if include_translated:
        untranslated_chapters = list(chapters)
    else:
        untranslated_chapters = [ch for ch in chapters if not ch.get('already_translated', False)]

    count_chapter_tokens(untranslated_chapters)

    return pack_by_tokens(
        [(ch, ch['source_tokens']) for ch in untranslated_chapters],
        max_tokens_per_patch,
    )


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


class EpubWriter:
    """
    Writes the translated EPUB, preserving the original's structure and metadata.

    The source book is read and parsed once and kept in memory, so the repeated
    progress saves during a run don't re-unzip and re-parse every image, font and
    stylesheet in the archive each time.
    """

    def __init__(self, original_book_path: str, target_language: str = "English"):
        self.book = epub.read_epub(original_book_path)

        original_title = self.book.get_metadata('DC', 'title')
        self.book.set_title(f"{original_title[0][0]} (Translated)" if original_title else "Translated Book")
        self.book.set_language(language_to_code(target_language))
        self.book.add_item(epub.EpubNcx())
        self.book.add_item(epub.EpubNav())

    def save(self, chapters: List[Dict], output_path: str):
        """Fold the current state of `chapters` into the book and write it to disk."""
        translated_soups = {ch['id']: ch['soup'] for ch in chapters}

        for item in self.book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = translated_soups.get(item.get_id())
            if soup is not None:
                item.content = str(_as_html_document(soup)).encode('utf-8')

        self.book.toc = [
            epub.Link(chapter.get('file_name', f"{chapter['id']}.html"),
                      chapter.get('translated_title', chapter['title']),
                      chapter['id'])
            for chapter in chapters
        ]

        epub.write_epub(output_path, self.book, {})
