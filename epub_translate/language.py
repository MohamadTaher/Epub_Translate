"""
Languages, their ISO codes, and judging from the characters alone whether a
piece of text has already been translated out of one.

Nothing here knows about EPUBs: it all works on plain text, so the same rules
answer "what language is this book in" and "has this chapter been done already".
"""

from typing import Dict, List, Tuple

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

# Unicode ranges of each language's script, used to tell whether text is still
# in the source language. Only languages written in a non-Latin script can be
# detected this way; see `is_detectable`.
SCRIPT_RANGES: Dict[str, List[Tuple[int, int]]] = {
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

# Below this share of source-script characters, text is considered translated
# already. Generous, because a translated chapter often keeps a few proper nouns
# in the original script.
TRANSLATED_SCRIPT_RATIO = 0.05

# Above this share, a book is taken to be written in that script. Well clear of
# the ratio above, so the two questions can never be answered "yes" at once.
DETECTED_SCRIPT_RATIO = 0.2


def normalize(language_name: str) -> str:
    """The form language names are compared and looked up in: stripped, lowercase."""
    return language_name.strip().lower()


def language_to_code(language_name: str) -> str:
    """Map a human-readable language name (or existing code) to an ISO 639-1 code."""
    normalized = normalize(language_name)
    if len(normalized) == 2:
        return normalized
    return LANGUAGE_CODES.get(normalized, "en")


def is_detectable(language_name: str) -> bool:
    """
    Whether text in this language can be recognised from its characters alone.

    Only true for non-Latin scripts. Spanish and English share an alphabet, so
    no amount of character inspection distinguishes a translated chapter from an
    untranslated one.
    """
    return normalize(language_name) in SCRIPT_RANGES


def script_ratio(text: str, language_name: str) -> float:
    """
    Share of `text` written in that language's script, ignoring whitespace,
    digits and punctuation.

    Raises KeyError for a language with no detectable script; callers should ask
    `is_detectable` first.
    """
    ranges = SCRIPT_RANGES[normalize(language_name)]

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


def detect(text: str) -> str:
    """
    Guess which language `text` is written in.

    Returns a language name from SCRIPT_RANGES, or "auto" when no non-Latin
    script accounts for a meaningful share of it — which includes every
    Latin-script book, since this cannot tell those apart.
    """
    if not text:
        return "auto"

    ratios = {language: script_ratio(text, language) for language in SCRIPT_RANGES}
    best = max(ratios, key=ratios.get)

    return best if ratios[best] >= DETECTED_SCRIPT_RATIO else "auto"


def looks_translated(text: str, source_language: str) -> bool:
    """
    Whether `text` has already been translated out of `source_language`.

    Languages whose script can't be detected are never reported as translated:
    re-translating a chapter wastes tokens, but wrongly skipping one silently
    leaves the book half translated.
    """
    if not is_detectable(source_language) or not text:
        return False

    return script_ratio(text, source_language) < TRANSLATED_SCRIPT_RATIO
